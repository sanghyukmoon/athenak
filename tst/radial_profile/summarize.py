#!/usr/bin/env python3
"""Summarize an approved campaign after performance.slurm completes."""
import csv
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

root = Path(sys.argv[1]).resolve()
assert Path('/scratch/gpfs/sm69/onthefly-rprof-test') in root.parents
rows = []
for n in (128, 256):
    for block in (32, 64):
        path = root / f'N{n}-B{block}'
        for file in sorted(path.glob('timings_rank*.csv')):
            samples = np.genfromtxt(file, delimiter=',', names=True)
            assert len(samples) == 21
            assert np.array_equal(samples['call'], np.arange(21))
            phases = ['allocation', 'reset', 'accumulation',
                      'reduction', 'normalization']
            accounted = sum(samples[p] for p in phases)
            assert np.all(samples['total'] >= accounted-1e-9)
            for phase in ['setup', *phases, 'total']:
                rows.append(dict(n=n, block=block, rank=file.stem.removeprefix('timings_rank'),
                    phase=phase, first=samples[phase][0],
                    median=np.median(samples[phase][1:]), minimum=min(samples[phase][1:]),
                    maximum=max(samples[phase][1:])))
assert len(rows) == 28, 'expected four single-rank cases'
with (root/'phase_summary.csv').open('w') as out:
    writer = csv.DictWriter(out, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
fig, ax = plt.subplots(figsize=(6, 4), layout='constrained')
for block in (32, 64):
    values = [r for r in rows if r['block'] == block and r['phase'] == 'total']
    median = np.array([r['median'] for r in values])
    ax.errorbar([r['n'] for r in values], median,
        yerr=[median-[r['minimum'] for r in values], [r['maximum'] for r in values]-median],
        marker='o', capsize=4, label=f'{block}³ cells/block')
ax.set(xlabel='Cells per domain side', ylabel='Seconds per profile (20-call median)',
       xticks=[128, 256])
ax.legend()
fig.savefig(root/'density_performance.pdf')
