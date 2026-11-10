"""Focused double-precision restart checks for the UCT extraction.

Only uniform, pure isothermal MHD shared-file restarts are supported. This is
test infrastructure, not a general AthenaK analysis reader.
"""
import argparse
import json
import os
import shlex
from pathlib import Path
import struct
import subprocess

import numpy as np

ATOL = RTOL = 1.0e-11


def read_restart(path):
    """Read active conserved cells and staggered faces without precision conversion."""
    raw = Path(path).read_bytes()
    end = raw.index(b"<par_end>\n") + len(b"<par_end>\n")
    params = {}
    block = None
    for line in raw[:end].decode().splitlines():
        line = line.split("#", 1)[0].strip()
        if line.startswith("<"):
            block = line.strip("<>")
        elif "=" in line:
            key, value = line.split("=", 1)
            params[block, key.strip()] = value.strip()
    assert params["mhd", "eos"] == "isothermal"
    assert int(params.get(("mhd", "nscalars"), "0")) == 0
    assert not any(b in {"hydro", "particles", "radiation", "z4c", "adm",
                         "turb_driving", "cyclic_zoom"} for b, _ in params)
    offset = end

    def take(dtype, count):
        nonlocal offset
        dtype = np.dtype(dtype)
        nbytes = count * dtype.itemsize
        if offset + nbytes > len(raw):
            raise ValueError("Truncated restart")
        result = np.frombuffer(raw, dtype=dtype, count=count, offset=offset).copy()
        offset += nbytes
        return result

    nmb, root = take("=i4", 2)
    size = take("=f8", 9)
    mesh = take("=i4", 19)
    mb = take("=i4", 19)
    time, dt = take("=f8", 2)
    cycle = int(take("=i4", 1)[0])
    locs = take("=i4", 4 * nmb).reshape(nmb, 4)
    take("=f4", nmb)  # load-balancing costs
    record = int(take("=u8", 1)[0])
    assert np.all(locs[:, 3] == root), "Only uniform meshes are supported"
    nx, ny, nz = map(int, mesh[1:4])
    bx, by, bz = map(int, mb[1:4])
    ng = int(mb[0])
    n1, n2, n3 = bx + 2*ng, by + 2*ng if by > 1 else 1, bz + 2*ng if bz > 1 else 1
    shapes = [(4, n3, n2, n1), (n3, n2, n1+1),
              (n3, n2+1, n1), (n3+1, n2, n1)]
    counts = [int(np.prod(shape)) for shape in shapes]
    assert record == 8 * sum(counts), "Unexpected precision or restart record layout"
    assert len(raw) == offset + nmb * record, "Unexpected state sections or file size"
    arrays = [np.full((4, nz, ny, nx), np.nan),
              np.full((nz, ny, nx+1), np.nan),
              np.full((nz, ny+1, nx), np.nan),
              np.full((nz+1, ny, nx), np.nan)]
    starts = [int(mb[8]), int(mb[6]), int(mb[4])]  # k,j,i
    active = [bz, by, bx]
    density_byte = offset + 8*((starts[0]*n2+starts[1])*n1+starts[2])
    for loc in locs:
        base = [int(loc[2])*bz, int(loc[1])*by, int(loc[0])*bx]
        values = [take("=f8", count).reshape(shape)
                  for count, shape in zip(counts, shapes)]
        for f, (target, value) in enumerate(zip(arrays, values)):
            lengths = list(active)
            if f:
                lengths[3-f] += 1
            source_slices = tuple(slice(a, a+n) for a, n in zip(starts, lengths))
            target_slices = tuple(slice(a, a+n) for a, n in zip(base, lengths))
            if f == 0:
                source_slices = (slice(None),) + source_slices
                target_slices = (slice(None),) + target_slices
            src = value[source_slices]
            dst = target[target_slices]
            mask = np.isfinite(dst)
            np.testing.assert_allclose(dst[mask], src[mask], rtol=RTOL, atol=ATOL)
            target[target_slices] = src
    for field in arrays:
        assert np.isfinite(field).all()
    for f, field in enumerate(arrays[1:]):
        axis = 2-f
        np.testing.assert_allclose(np.take(field, 0, axis=axis),
                                   np.take(field, -1, axis=axis),
                                   rtol=RTOL, atol=ATOL)
    assert (arrays[0][0] > 0).all()
    return dict(fields=arrays, time=float(time), dt=float(dt), cycle=cycle,
                spacing=size[6:9], shape=(nx, ny, nz), params=params, raw=raw, density_byte=density_byte)


def compare(left, right, require_epoch=True):
    """Compare every conserved component and every physical staggered face."""
    if require_epoch:
        assert left["cycle"] == right["cycle"]
        np.testing.assert_allclose(left["time"], right["time"], rtol=RTOL, atol=ATOL)
    maxima = []
    for a, b in zip(left["fields"], right["fields"]):
        np.testing.assert_allclose(a, b, rtol=RTOL, atol=ATOL)
        maxima.append(float(np.max(np.abs(a-b))))
    return maxima


def cell_fields(state):
    u, b1, b2, b3 = state["fields"]
    return np.concatenate((u, np.stack((0.5*(b1[..., 1:]+b1[..., :-1]),
                                        0.5*(b2[:, 1:]+b2[:, :-1]),
                                        0.5*(b3[1:]+b3[:-1])))), axis=0)


def check_divergence(state):
    _, b1, b2, b3 = state["fields"]
    dx, dy, dz = state["spacing"]
    div = np.diff(b1, axis=2)/dx
    if state["shape"][1] > 1:
        div += np.diff(b2, axis=1)/dy
    if state["shape"][2] > 1:
        div += np.diff(b3, axis=0)/dz
    scale = max(float(np.max(np.abs(b))) for b in (b1, b2, b3))
    value = float(np.max(np.abs(div))) * min(dx, dy, dz) / scale
    assert value < 1.0e-10, value
    return value


def check_errors(initial, final, path):
    data = np.loadtxt(path, ndmin=2)[-1]
    labels = Path(path).read_text().splitlines()[0].split()[1:]
    assert len(labels) == len(data) == 13
    assert labels[-3:] == ["B1_L1", "B2_L1", "B3_L1"]
    assert "E_L1" not in labels
    exact = np.mean(np.abs(cell_fields(final)-cell_fields(initial)), axis=(1, 2, 3))
    np.testing.assert_allclose(data[6:], exact, rtol=1.0e-5, atol=1.0e-12)
    np.testing.assert_allclose(data[4], np.linalg.norm(exact), rtol=1.0e-5, atol=1.0e-12)
    return float(data[4])


def write_input(template, destination, overrides):
    """Materialize keys before startup; AthenaK CLI overrides cannot add keys."""
    blocks = {}
    block = None
    for line in Path(template).read_text().splitlines():
        line = line.split("#", 1)[0].strip()
        if line.startswith("<"):
            block = line.strip("<>")
            blocks.setdefault(block, {})
        elif "=" in line:
            key, value = line.split("=", 1)
            blocks[block][key.strip()] = value.strip()
    for key, value in overrides.items():
        block, name = key.split("/", 1)
        values = blocks.setdefault(block, {})
        if value is None:
            values.pop(name, None)
        else:
            values[name] = str(value)
    text = "\n".join("<%s>\n%s\n" % (block, "\n".join(
        "%s = %s" % item for item in values.items())) for block, values in blocks.items())
    Path(destination).write_text(text)


def run_case(binary, input_file, directory, overrides=None, launcher=None, success=True):
    """Run in a new directory and retain logs/checkpoints on failure."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    case_input = directory/"input.athinput"
    write_input(input_file, case_input, dict(overrides or {}))
    if launcher is None:
        launcher = shlex.split(os.environ.get("UCT_LAUNCH", ""))
    command = list(launcher) + [str(binary), "-i", str(case_input.resolve())]
    result = subprocess.run(command, cwd=directory, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True)
    (directory/"run.log").write_text(result.stdout)
    if success:
        assert result.returncode == 0, result.stdout[-6000:]
        initial = read_restart(directory/"rst/uct.00000.rst")
        final = read_restart(directory/"rst/uct.final.rst")
        check_divergence(initial)
        check_divergence(final)
        return initial, final
    assert result.returncode != 0, "Unsupported configuration was accepted"
    assert "FATAL ERROR" in result.stdout
    return result.stdout


def dimensions(nx=32, ndim=3, split=False):
    shape = (nx, nx//2 if ndim > 1 else 1, nx//2 if ndim > 2 else 1)
    blocks = (16, 8 if ndim > 1 else 1, 8 if ndim > 2 else 1) if split else shape
    return {"mesh/nx%d" % (i+1): n for i, n in enumerate(shape)} | {
        "meshblock/nx%d" % (i+1): n for i, n in enumerate(blocks)}


def run_regressions(binary, input_file, root, full=True, baseline=None, donor=None):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    evidence = {}
    small = {}
    for method in ("ct_contact", "uct_hlld", "uct"):
        opts = dimensions()
        opts.update({"mhd/emf": method, "time/nlim": 2, "time/tlim": 100})
        small[method] = run_case(binary, input_file, root/("path_"+method), opts)
    compare(small["uct_hlld"][1], small["uct"][1])
    a, b = small["ct_contact"][1]["fields"], small["uct_hlld"][1]["fields"]
    distinction = max(float(np.max(np.abs(x-y)/(ATOL+RTOL*np.abs(y))))
                      for x, y in zip(a[1:], b[1:]))
    assert distinction > 100, distinction
    evidence["method_distinction_allowance_units"] = distinction
    for label, executable, method in (("baseline", baseline, "ct_contact"),
                                      ("donor", donor, "uct_hlld")):
        if executable:
            opts = dimensions()
            opts.update({"time/nlim": 2, "time/tlim": 100})
            if label == "donor":
                opts.update({"mhd/emf": method, "mhd/mood": "false",
                             "mhd/well_balanced": "false", "mhd/uct_diag": "false"})
            ref = run_case(executable, input_file, root/label, opts)
            evidence[label+"_maxima"] = compare(small[method][1], ref[1])
    omitted = run_case(binary, input_file, root/"omitted", dimensions() |
                       {"time/nlim": 2, "time/tlim": 100, "mhd/emf": None})
    compare(omitted[1], small["ct_contact"][1])
    # The actual tests are multidimensional, so valid UCT calls enter CornerE.
    for axis in range(3):
        for velocity in (0, 0.25):
            for method in ("ct_contact", "uct_hlld"):
                opts = dimensions(16)
                opts.update({"mesh/nx2": 16, "mesh/nx3": 16,
                             "meshblock/nx2": 16, "meshblock/nx3": 16,
                             "problem/b_perp": 0, "problem/v_par": velocity,
                             "time/nlim": 10, "time/tlim": 100, "mhd/emf": method})
                opts.update({"problem/along_x%d" % (d+1): str(d == axis).lower()
                             for d in range(3)})
                name = "uniform_%s_%s_%s" % (axis, velocity, method)
                initial, final = run_case(binary, input_file, root/name, opts)
                compare(initial, final, require_epoch=False)
    # Verify that the checkpoint reader retains sub-float32 changes in active cells.
    raw = bytearray(initial["raw"])
    old = struct.unpack_from("=d", raw, initial["density_byte"])[0]
    struct.pack_into("=d", raw, initial["density_byte"], old+1.0e-12)
    precision_file = root/"reader_precision.rst"
    precision_file.write_bytes(raw)
    precise = read_restart(precision_file)
    assert precise["fields"][0][0,0,0,0] != old
    assert precise["fields"][0][0,0,0,0] == old+1.0e-12
    assert np.float32(old+1.0e-12) == np.float32(old)
    bad_cases = {"emf": {"mhd/emf": "invalid"}, "solver": {"mhd/rsolver": "hlle"},
                 "recon": {"mhd/reconstruct": "ppm4"}, "eos": {"mhd/eos": "ideal"},
                 "rk3": {"time/integrator": "rk3"}, "fofc": {"mhd/fofc": "true"},
                 "ghost": {"mesh/nghost": 2}, "scalar": {"mhd/nscalars": 1},
                 "diffusion": {"mhd/nu_iso": 0.01},
                 "boundary": {"mesh/ix1_bc": "outflow"},
                 "refinement": {"mesh_refinement/refinement": "static"},
                 "sts": {"time/sts_integrator": "rkl2"}}
    for name, bad in bad_cases.items():
        opts = dimensions(16, 2) | {"mhd/emf": "uct_hlld", "time/nlim": 0} | bad
        run_case(binary, input_file, root/("reject_"+name), opts, success=False)
    # Even a genuine 1D call must preserve the original face-to-edge copy.
    states = [run_case(binary, input_file, root/("one_d_"+method),
                       dimensions(32, 1) | {"mhd/emf": method, "problem/along_x1": "true"})
              for method in ("ct_contact", "uct_hlld")]
    compare(states[0][1], states[1][1])
    if full:
        evidence["errors"] = {}
        for ndim in (2, 3):
            for method in ("ct_contact", "uct_hlld"):
                errors = []
                for n in (32, 64, 128):
                    name = "wave_%dd_%s_%d" % (ndim, method, n)
                    initial, final = run_case(binary, input_file, root/name,
                                             dimensions(n, ndim) | {"mhd/emf": method})
                    errors.append(check_errors(initial, final, root/name/"uct-errs.dat"))
                    if n == 64:
                        split = run_case(binary, input_file, root/(name+"_split"),
                                         dimensions(n, ndim, True) | {"mhd/emf": method})
                        evidence[name+"_split_maxima"] = compare(final, split[1])
                    if ndim == 3:
                        assert (np.mean(np.abs(cell_fields(final)[4:] -
                                               cell_fields(initial)[4:]), axis=(1, 2, 3)) > 0).all()
                assert errors[2] < 1e-2 and errors[2]/errors[1] < .4, errors
                assert errors[0] > errors[1] > errors[2], errors
                evidence["errors"]["%dd_%s" % (ndim, method)] = errors
        rotated = {}
        for ndim in (2, 3):
            for axis in range(ndim):
                for method in ("ct_contact", "uct_hlld"):
                    opts = dimensions(32 if ndim == 2 else 16, ndim)
                    if ndim == 3:
                        opts.update({"mesh/nx2": 16, "mesh/nx3": 16,
                                     "meshblock/nx2": 16, "meshblock/nx3": 16})
                        opts.update({"mesh/x1max": 1})
                    opts.update({"mhd/emf": method})
                    opts.update({"problem/along_x%d" % (d+1): str(d == axis).lower()
                                 for d in range(3)})
                    name = "axis_%dd_%s_%d" % (ndim, method, axis)
                    initial, final = run_case(binary, input_file, root/name, opts)
                    check_errors(initial, final, root/name/"uct-errs.dat")
                    for field in final["fields"]:
                        # Include all conserved components, not just density.
                        for spatial in ([field] if field.ndim == 3 else field):
                            for d in range(ndim):
                                if d != axis:
                                    assert np.max(np.ptp(spatial, axis=2-d)) < ATOL
                    if ndim == 3:
                        # Undo the proper rotations used in cpaw.cpp.
                        order, signs = ((0,1,2),(1,1,1)) if axis == 0 else (
                            ((1,0,2),(1,-1,1)) if axis == 1 else ((2,0,1),(1,-1,-1)))
                        u, *faces = final["fields"]
                        def profile(f):
                            idx = [0,0,0]
                            idx[2-axis] = slice(None)
                            return f[tuple(idx)]
                        profiles = [profile(u[0])]
                        profiles += [signs[d]*profile(u[1+order[d]]) for d in range(3)]
                        profiles += [signs[d]*profile(faces[order[d]]) for d in range(3)]
                        if axis == 0:
                            rotated[method] = profiles
                        else:
                            for a, b in zip(rotated[method], profiles):
                                np.testing.assert_allclose(a,b,rtol=RTOL,atol=ATOL)
    (root/"summary.json").write_text(json.dumps(evidence, indent=2))
    return evidence


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", type=Path)
    parser.add_argument("--input", type=Path)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--donor", type=Path)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    print(json.dumps(run_regressions(args.binary.resolve(), args.input.resolve(), args.root,
                                    not args.smoke, args.baseline, args.donor), indent=2))


if __name__ == "__main__":
    main()
