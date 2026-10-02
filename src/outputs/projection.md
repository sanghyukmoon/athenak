# Projection output

Use a scheduled output block to write full-domain projections along x, y, and z:

    <output6>
    file_type = proj
    dt = 0.1
    id = proj

The optional `id` defaults to `proj`. No `variable` is required. Standard `dcycle`,
`file_number`, `last_time`, basename, initial/final output and restart behavior apply.
Files are `proj/<basename>.<id>.<number:05d>.proj`, with one file per output epoch.

Supported meshes are uniform 3D Newtonian Cartesian grids with one hydro or MHD
fluid. Non-cubic domains and unequal cell widths are supported. Refinement,
relativistic coordinates, two-fluid ion-neutral models, slices, selected MeshBlock
IDs and ghost-zone output are rejected.

Each image contains `surface_density`, `velocity_mass_weighted_x`,
`velocity_mass_weighted_y`, and `velocity_mass_weighted_z`, in that order. MHD adds
`bfield_mass_weighted_x`, `bfield_mass_weighted_y`, and `bfield_mass_weighted_z`.
Cartesian component names stay fixed for all viewing axes. Values use code units.
For a viewing axis with cell width dl, the fields are sums over active cells:

    surface_density = sum(rho * dl)
    velocity_mass_weighted_i = sum(momentum_i * dl) / surface_density
    bfield_mass_weighted_i = sum(rho * B_i * dl) / surface_density

Magnetic fields use cell-centered B. Accumulation and storage are float64, even
in a single-precision simulation. Normalization uses direct division; zero-density
columns therefore follow floating-point division behavior. One image buffer per
rank is reused across axes. MPI builds sum full device image buffers using in-place
MPI_Allreduce before rank 0 normalizes and writes.

## Version 1.0 binary format

The file starts with these newline-terminated ASCII lines (angle brackets denote
values, not literal characters):

    Athena projection output version=1.0
      size of preheader=11
      time=<17-significant-digit value>
      cycle=<integer>
      output number=<integer>
      size of variable=8
      byte order=<little or big>
      number of axes=3
      root grid size=<Nx> <Ny> <Nz>
      domain left edge=<x1min> <x2min> <x3min>
      domain right edge=<x1max> <x2max> <x3max>
      feature flags=<0 for hydro, 1 for MHD>
      number of variables=<4 or 7>
      variables: <space-separated field names>
      header offset=<parameter-dump byte length>

The preheader count includes its own line through `feature flags`. Immediately
after the last newline are exactly `header offset` bytes from ParameterDump.
This length is not an absolute file offset. Immediately after the dump are the
x, y, z image payloads, with no padding or per-axis records. Payloads are native
endian IEEE float64, in the byte order stated in the header.

Each axis is field-major with a contiguous fast pixel index. In C/NumPy order,
the x image has shape `[nfield, Nz, Ny]` (fast=y, slow=z), the y image has shape
`[nfield, Nz, Nx]` (fast=x, slow=z), and the z image has shape `[nfield, Ny, Nx]`
(fast=x, slow=y). Pixel coordinates are cell centers implied by domain edges and
root counts. The total payload size is `8*nfield*(Nz*Ny + Nz*Nx + Ny*Nx)` bytes.
No existing binary-volume reader is compatible with this new format; production
Python reader integration is separate work.
