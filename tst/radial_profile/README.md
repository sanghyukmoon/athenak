# Density calculator checks and review-gated benchmark

`inputs/radial_profile/athinput.density` runs static isothermal hydro on [-2,2]^3.
Compile out of tree with `-DPROBLEM=radial_profile_benchmark` and
`-DAthena_ENABLE_MPI=ON`; add `-DAthena_ENABLE_OPENMP=ON` for CPU or
`-DKokkos_ENABLE_CUDA=ON` and the correct architecture for GPU. Never reuse a
CPU build directory for GPU. GPU MPI must support device buffers; this is an
execution prerequisite.
No host staging is implemented.

The Stellar-specific `build.sh cpu|gpu|a100 SCRATCH_DIRECTORY` reproduces these
module stacks and creates separate CPU, V100, or A100 build directories. It
only builds. Load the matching MPI/compiler modules, then run (using absolute paths):

    /home/sm69/miniforge3/envs/pyathena/bin/python \
      /home/sm69/athenak/tst/radial_profile/verify.py \
      --exe /scratch/gpfs/sm69/onthefly-rprof-test/BUILD/src/athena \
      --backend cpu --output /scratch/gpfs/sm69/onthefly-rprof-test/FRESH-CPU

For this Stellar CUDA MPI stack, set `OMPI_MCA_pml=ob1` and
`OMPI_MCA_btl=self,smcuda,tcp`, and `OMPI_MCA_coll=^hcoll`. The default system UCX transport failed in
AthenaK boundary exchange on CUDA pointers despite the positive CUDA support
query. This explicitly selects the installed CUDA shared-memory transport.
The installed HCOLL collective also failed on device buffers and is explicitly
disabled; OpenMPI handles the device-buffer reduction. The calculator still
passes device buffers directly to MPI. The positive runtime query alone does
not certify every installed transport/collective combination.

Run again with `--backend gpu` and a CUDA binary, adding `--compare FRESH-CPU`.
The verifier runs small MPI jobs with one/two ranks and uses explicit global
cell masks as an independent oracle (rtol 1e-11, atol 1e-12). It exercises
reused storage, three ordered centers, changed/empty center counts, periodic
boundaries, poisoned ghosts, large IDs, and strict inward whole-bin extents.
Device kernels also create and reorder prescribed centers directly, with no
intervening host copy before calculation. Diagnostics verify that Compute leaves
center IDs and coordinates unchanged. No performance output is enabled. Output
paths must be new.

The calculator consumes primitive density (`hydro/w0`, otherwise `mhd/w0`).
Every caller must ensure primitives are current. Providers return a managed
`DvceArray1D<RadialProfileCenter>` with all writes completed. `Compute` accepts
`const DvceArray1D<const RadialProfileCenter>&`, reads its device data directly,
and neither modifies nor retains it. The extent is the active center count.
Callers keep centers unchanged through Compute; all ranks supply identical
ordered centers. Equality is a caller precondition, not a collective check.

The CPU fixture provider uses `DualArray1D`: fill `view_host()`, call
`modify_host()` and `sync_device()`, and return `view_device()`. The returned
managed view keeps the allocation alive. Future GPU minima providers can return
a device array directly; minima discovery and global-list assembly are deferred.
Diagnostic copies of centers/results to the host occur outside calculation timing.

Supported geometry and valid finite extents are caller preconditions: uniform
3D Cartesian meshes with cubic cells, periodic boundaries, no refinement, and
`dx/2 <= rmax <= Lmin/2`. The complete reduction buffer must fit an MPI integer
count. Removed constructor checks are not exercised as rejection tests.

`result(center,field,bin)` is valid globally only on
rank 0 and is ready on return. Fields are `density` and `sampled_volume`.
Empty shells have NaN density and zero volume. IDs are metadata, not array
indices. `dr=dx`, radii are `bin*dr`, and the requested `rmax` is an outer edge;
complete half-open shells are retained with strict inward rounding, without
near-edge snapping. Example L4/dx1/rmax2 yields radii0,1 and edge1.5.

Do not run `performance.slurm` until explicit user code-review approval.
After approval, compile for the allocated GPU (A100: `Kokkos_ARCH_AMPERE80=ON`),
create a new scratch campaign directory, and submit the script with
`--review-approved /absolute/path/to/athena`. It requests one GPU and runs
N128/N256 with block32/block64, first call plus twenty unchanged-data repeats.
Inspect the job at startup and at least every 30 minutes. No campaign has been
run as part of preparing these scripts.

Each rank records constructor setup separately from Compute allocation,
reset, accumulation (including ScatterView contribution),
blocking device MPI reduction, root normalization, and total Compute time.
Phases use device fences without inter-rank barriers. MPI time can include
arrival imbalance. Provider calls (including center allocation, preparation and
transfer),
initialization, diagnostic copies, memory queries and writes are outside Compute
timing. The timing CSV and reporting script use `reset`, replacing the previous
`reset_transfer` phase; no performance campaign has used the previous schema.
First-call calculation and
setup must be reported separately from the 20-call median/min/max; seconds
per profile equal seconds per calculation for this one-center fixture.
`memory_rank*.txt` records lifetime peak host RSS and current total used GPU
memory (device-wide, not a per-calculator allocation or peak device metric).
Run `summarize.py CAMPAIGN` for phase CSV and PDF plots after the campaign.
Persistent radial output, reader integration, subcells, fields beyond density,
and minima discovery are deferred.

The 5D `par_for` defaults to signed `int` indexing with an explicit Kokkos
`IndexType` policy. Use `par_for<std::int64_t>` when the combined iteration count
needs 64 bits; each individual dimension and callback coordinate must still fit int. Accumulation
uses this wide wrapper over `(center, MeshBlock, k, j, i)`; normalization uses
2D `par_for` under the MPI buffer-count precondition. Cell coordinates stay
zero-based until primitive-array access adds the active-cell offsets.
