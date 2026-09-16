#!/usr/bin/env python3
"""Small independent whole-cell oracle. No performance measurements or large grids.

Run in a fresh scratch directory with the matching build's MPI modules loaded:
  python verify.py --exe /scratch/.../build-cpu/src/athena --backend cpu --output ...
Uses explicit shell masks, rather than the calculator's radius-to-bin formula.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess

import numpy as np

RTOL, ATOL = 1e-11, 1e-12
SOURCE = Path(__file__).resolve().parents[2]
TEMPLATE = (SOURCE / 'inputs/radial_profile/athinput.density').read_text()


def parse(path):
    lines = path.read_text().splitlines()
    dx, nr, requested, edge, count = map(float, lines[1].split())
    nr, count = int(nr), int(count)
    centers, data = [], []
    pos = 2
    for _ in range(count):
        fields = lines[pos].split()
        centers.append((int(fields[2]), *map(float, fields[3:])))
        data.append(np.array([list(map(float, row.split()))
                              for row in lines[pos+1:pos+nr+1]]))
        pos += nr+1
    assert pos == len(lines), path
    return dx, nr, requested, edge, centers, data


def check_run(directory, n, varying, requested, bounds=(-2., 2.)):
    length = bounds[1]-bounds[0]
    dx = length/n
    axis = bounds[0] + (np.arange(n)+0.5)*dx
    z, y, x = np.meshgrid(axis, axis, axis, indexing='ij')
    rho = (1 + 0.1*np.cos(np.pi*x/2) + 0.05*np.sin(np.pi*y/2)
           + 0.025*np.cos(np.pi*z/2)) if varying else np.ones_like(x)
    edges = (np.arange(n)+0.5)*dx
    # Strict inclusion of complete candidate shells; no near-edge snapping.
    complete = edges <= requested
    nr = np.count_nonzero(complete)
    midpoint = (n//2 + n*(n//2 + n*(n//2)), axis[n//2], axis[n//2], axis[n//2])
    expected = {
        'single': [midpoint], 'repeat': [midpoint],
        'three': [midpoint, (n-1, bounds[0], bounds[0], bounds[0]),
                  (0, axis[0], axis[0], axis[0])],
        'reordered': [(n-1, bounds[0], bounds[0], bounds[0]),
                      (0, axis[0], axis[0], axis[0]), midpoint],
        'shrink': [midpoint], 'initial_empty': [], 'empty': [], 'restored': [midpoint],
    }
    expected['device_three'] = expected['three']
    expected['device_reordered'] = expected['reordered']
    maximum_error = 0.
    for label, centers in expected.items():
        actual_dx, actual_nr, actual_requested, edge, got_centers, data = parse(
            directory / f'profile_{label}.txt')
        assert actual_nr == nr, (label, actual_nr, nr)
        np.testing.assert_allclose([actual_dx, actual_requested, edge],
                                   [dx, requested, edges[nr-1]], rtol=RTOL, atol=ATOL)
        assert len(got_centers) == len(centers)
        for center, got, profile in zip(centers, got_centers, data):
            assert got[0] == center[0], (got, center)
            np.testing.assert_allclose(got[1:], center[1:], rtol=0, atol=ATOL)
            # Modulo minimum-image calculation is independent of C++ round().
            length = bounds[1]-bounds[0]
            distances = [(coord-c+length/2) % length-length/2
                         for coord, c in zip((x, y, z), center[1:])]
            radius = np.sqrt(sum(d*d for d in distances))
            reference = []
            for shell in range(nr):
                low, high = max(0, (shell-0.5)*dx), (shell+0.5)*dx
                mask = (radius >= low) & (radius < high)
                volume = np.count_nonzero(mask)*dx**3
                density = rho[mask].mean() if volume else np.nan
                reference.append([shell*dx, density, volume])
            reference = np.array(reference)
            np.testing.assert_allclose(profile, reference, rtol=RTOL, atol=ATOL,
                                       equal_nan=True, err_msg=f'{directory.name}/{label}')
            maximum_error = max(maximum_error, float(np.nanmax(abs(profile-reference))))
    # Synthetic provider arithmetic, with >2^32 ID, without a giant mesh allocation.
    large = (directory / 'large_id.txt').read_text().split()
    assert int(large[0]) == 32768 + 65536*(32768+65536*2)
    np.testing.assert_allclose(list(map(float, large[1:])),
        [bounds[0]+(32768.5)*length/65536]*2+[bounds[0]+2.5*length/4], rtol=0, atol=ATOL)
    assert not list(directory.glob('timings*')), 'correctness mode collected timing data'
    assert not list(directory.glob('memory*')), 'correctness mode collected memory baseline'
    return maximum_error


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--exe', type=Path, required=True)
    parser.add_argument('--backend', choices=['cpu', 'gpu'], required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--compare', type=Path)
    args = parser.parse_args()
    root = args.output.resolve()
    allowed = Path('/scratch/gpfs/sm69/onthefly-rprof-test')
    assert allowed in root.parents, 'all generated test artifacts must be in scratch'
    root.mkdir(parents=True, exist_ok=True)
    results = []
    env = os.environ.copy()
    env.update(OMP_NUM_THREADS='2', OMP_PROC_BIND='false', PYTHONDONTWRITEBYTECODE='1')

    def run(name, ranks=1, n=16, block=8, varying=True, rmax=2., mhd=False,
            overrides=(), error=None, bounds=(-2., 2.)):
        directory = root / name
        directory.mkdir(exist_ok=False)
        template = TEMPLATE.replace('<hydro>', '<mhd>') if mhd else TEMPLATE
        (directory / 'input').write_text(template)
        options = [f'mesh/nx{a}={n}' for a in (1, 2, 3)]
        options += [f'meshblock/nx{a}={block}' for a in (1, 2, 3)]
        options += [f'mesh/x{a}min={bounds[0]}' for a in (1, 2, 3)]
        options += [f'mesh/x{a}max={bounds[1]}' for a in (1, 2, 3)]
        options += [f'problem/varying={str(varying).lower()}', f'problem/rmax={rmax}']
        command = ['mpiexec', '-n', str(ranks), '--bind-to', 'none', str(args.exe.resolve()),
                   '-i', 'input', *options, *overrides]
        (directory / 'command.json').write_text(json.dumps(command))
        with (directory / 'run.log').open('w') as log:
            done = subprocess.run(command, cwd=directory, env=env,
                                  stdout=log, stderr=subprocess.STDOUT, timeout=120)
        log = (directory / 'run.log').read_text()
        if error:
            assert done.returncode != 0 and error in log, (name, done.returncode, log[-2500:])
            result = {'case': name, 'diagnostic': error}
        else:
            assert done.returncode == 0, (name, log[-4000:])
            result = {'case': name, 'max_absolute_error': check_run(
                directory, n, varying, rmax, bounds)}
            placement = []
            for rank in range(ranks):
                record = dict(line.split(maxsplit=1) for line in
                    (directory/f'placement_rank{rank}.txt').read_text().splitlines())
                placement.append(record)
            assert sum(int(p['blocks']) for p in placement) == (n//block)**3
            if args.backend == 'gpu':
                assert len({p['pci_bus'] for p in placement}) == ranks, placement
            result['placement'] = placement
        results.append(result)
        print('PASS', name, flush=True)

    run('uniform', varying=False)
    run('varying')
    run('two_rank', ranks=2)
    run('one_block', block=16)
    run('mhd', mhd=True)
    run('mhd_two_rank', ranks=2, mhd=True)
    run('inward', n=4, block=4)
    run('exact_edge', rmax=1.125)
    run('near_edge', rmax=np.nextafter(1.125, 0.))
    run('above_edge', rmax=np.nextafter(1.125, np.inf))
    run('below_edge', rmax=1.125-1e-10)
    run('smallest', rmax=0.125)
    run('smallest_two_rank', ranks=2, rmax=0.125)
    run('decimal_dx', n=12, block=6, rmax=7/6)
    # Valid finite extents and supported geometry are caller preconditions.
    # Explicit decomposition and backend comparisons as well as independent oracles.
    comparisons = []
    for other in (root/'two_rank', root/'one_block', root/'mhd', root/'mhd_two_rank'):
        for profile in (root/'varying').glob('profile_*.txt'):
            base, got = parse(profile), parse(other/profile.name)
            assert base[:5] == got[:5]
            np.testing.assert_allclose(base[5], got[5], rtol=RTOL, atol=ATOL, equal_nan=True)
        comparisons.append(str(other))
    if args.compare:
        for case in results:
            if 'diagnostic' in case:
                continue
            for profile in (root/case['case']).glob('profile_*.txt'):
                base, got = parse(profile), parse(args.compare/case['case']/profile.name)
                assert base[:5] == got[:5]
                np.testing.assert_allclose(base[5], got[5], rtol=RTOL, atol=ATOL,
                                           equal_nan=True)
        comparisons.append(str(args.compare))
    record = dict(backend=args.backend, rtol=RTOL, atol=ATOL, cases=results,
                  comparisons=comparisons, performance_run=False)
    (root/'verification.json').write_text(json.dumps(record, indent=2)+'\n')
    print(f'{len(results)} cases passed; no performance baseline collected', flush=True)


if __name__ == '__main__':
    main()
