# Radial-profile binary output

`RadialProfileOutput` connects the existing radial-profile calculator to AthenaK's
ordinary output dispatch. For a problem that enrolls `rprof_center_func`, add:

```ini
<output1>
file_type = rprof
dt = 0.01
# Alternatively use dcycle = 10 instead of dt.
# rmax defaults to half the shortest domain length.
# nsub = 4
# nbins_subcell_corrected = 4
```

Files are `rprof/<job/basename>.<file_number:05d>.rprof`. Neither `variable` nor
`id` is required or used. Mesh slicing, ghost-zone inclusion, and per-rank file
options do not alter the radial output. Existing generic input parsing still
applies; this output adds no option or duplicate-stream checks.

The problem callback supplies identical ordered `RadialProfileCenter` records on
every MPI rank, including current center velocities. Centers must satisfy the
calculator's existing assumptions: cell centers on uniform periodic Cartesian
meshes with cubic cells. The calculator enforces its existing radius and
subdivision checks. The default radius is half the shortest domain length.
The output owns one calculator from construction through destruction.
`LoadOutputData` invokes the callback and calculator on every rank at every output. The calculator
returns complete normalized profiles; rank zero prepares host views without
another profile reduction or gather. All ranks open and close the shared file
through `IOWrapper`; rank zero writes the header and complete payload using
noncollective writes. The destination is opened directly.

## Scheduling and scope

Initial, periodic, final, and restart dispatch are unchanged. The writer records
actual simulation time and cycle and the stream's native output number. After
closing it increments `file_number`; a negative `last_time` becomes the actual
time, otherwise `dt` is added. Both entries are saved in `ParameterInput`, just
as in `MeshBinaryOutput`.

An initial output can contain centers. Final dispatch always writes a file,
including when a periodic output was just written. Restart initialization skips
initial output and continues saved counters. These are AthenaK's existing
conventions; this component does not implement the separate Athena++/Python
production cadence contract or its initial empty-profile policy.

The caller remains responsible for center and field readiness. This writer does
not discover potential minima or change gravity scheduling. The current Python
reader accepts historical ATHRPRF version-1 files, not this text-signature format. Reader support, scientific
comparisons, production acceptance, and temporary-file publication are separate
work.

## Text header, version 1.0

The header uses ASCII text and follows the mesh binary writer's scientific time
precision (`std::numeric_limits<Real>::max_digits10 - 1`). The exact layout is:

```text
Athena radial profile output version=1.0
  size of preheader=9
  time=<actual simulation time>
  cycle=<cycle>
  output number=<native file number>
  feature flags=<integer bitmask>
  nsub=<subdivisions>
  nbins_subcell_corrected=<corrected-bin count>
  number of centers=<ncenter>
  number of radial bins=<nr>
  number of variables=<nvariable>
  variables:  <ordered variable names separated by spaces>
```

The preheader count includes its own line and the eight metadata lines before
`number of variables`, matching the mesh reader's convention. Feature bit 0 is
MHD and bit 1 is gravity. `ncenter` and `nr` come from calculator result extents;
`nvariable` is three plus the number of selected radial fields. Binary data begins
immediately after the newline terminating `variables:`. There is no parameter
dump, geometry header, stored offset, dtype declaration or descriptor table.

## Sequential binary payload

Integers and real values use native byte order, without byte-order detection or
swapping. Center IDs are unsigned 64-bit integers. Every coordinate and field
value is an eight-byte double, independent of simulation `Real` precision.
Values are copied individually or as double arrays, without struct padding.
Starting at the first byte after the text header, read:

| Order | Type | Elements | Meaning |
| ---: | --- | ---: | --- |
| 1 | uint64 | `ncenter` | Center IDs in provider order |
| 2 | double | `3*ncenter` | Interleaved center positions `(x1,x2,x3)` |
| 3 | double | `nr` | Radial coordinates `r[k]=k*dx` |
| 4 | double | `ncenter` per variable | Three origin-velocity arrays |
| 5 | double | `ncenter*nr` per variable | Selected radial-field arrays |

The first three named variables are `velocity_x_origin`, `velocity_y_origin`,
and `velocity_z_origin`. Their center-only classification is part of the format
contract. Every subsequent named variable is radial, with center-major rows and
radius varying fastest. Names match `RadialProfile::Field` in enum order. Without
MHD, fields 39–50 and 58–62 are omitted; without gravity, fields 51–57 are omitted.
Variable counts are 51 (hydro), 58 (hydro+gravity), 68 (MHD), and 75 (MHD+gravity).
All nine `mass_flux_ij` transport-tensor components are stored unchanged, with
the velocity index first and radial-direction index second.

Provider ordering is preserved, without sorting or deduplication. The last
stored radius need not equal the requested `rmax`. The total payload byte count is
`8*(7*ncenter + nr + (nvariable-3)*ncenter*nr)`. Empty-center files retain the text
header and radial coordinates; all center-dependent arrays have zero length.

The text signature distinguishes this format from historical descriptor-based
`ATHRPRF` files. Those files and the existing production reader remain unchanged.
The text `version=1.0` belongs to this new signature; it does not imply
compatibility with the historical binary version-1 schema. Reader implementation
and `athinput.runtime` work remain deferred.
