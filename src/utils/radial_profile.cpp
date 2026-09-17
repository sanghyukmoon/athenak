#include "utils/radial_profile.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <limits>

#include "coordinates/coordinates.hpp"
#include "coordinates/cell_locations.hpp"
#include "globals.hpp"
#include "hydro/hydro.hpp"
#include "mesh/mesh.hpp"
#include "mhd/mhd.hpp"

#if MPI_PARALLEL_ENABLED
#include <mpi.h>
#endif

#if MPI_PARALLEL_ENABLED && defined(KOKKOS_ENABLE_CUDA) && defined(OPEN_MPI)
#include <mpi-ext.h>
#endif

namespace {
[[noreturn]] void Fail(const char *message) {
  std::cerr << "### FATAL ERROR in RadialProfile: " << message << std::endl;
#if MPI_PARALLEL_ENABLED
  MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
#endif
  std::exit(EXIT_FAILURE);
}

// Consider:
// -----------------------------------------------------
// 0   |   1   |   2   | ... |   n-1   |     n     | ...
//  0.5*dr  1.5*dr  2.5*dr      (n-0.5)*dr  (n+0.5)*dr
// -----------------------------------------------------
// If (n-0.5)*dr <= r < (n+0.5)*dr, then bin index = n
KOKKOS_INLINE_FUNCTION
int BinIndex(Real r, Real dr) {
  return static_cast<int>(std::floor(r/dr + 0.5));
}
}  // namespace

RadialProfile::RadialProfile(Mesh *mesh, Real requested_rmax)
    : bin_width(mesh->mesh_size.dx1),
      num_bins(BinIndex(requested_rmax, bin_width)),
      mesh_(mesh) {
}


//----------------------------------------------------------------------------------------
// \!fn void RadialProfile::Compute()
// \brief

DvceArray3D<Real> RadialProfile::Compute(
    const DvceArray1D<const RadialProfileCenter>& centers) {
  // Local timer lambda
  Kokkos::Timer clock;
  double start = 0;
  auto phase = [&](double &seconds) {
    if (measure_time) {
      Kokkos::fence();
      const double now = clock.seconds();
      seconds = now - start;
      start = now;
    }
  };
  timings = {}; // clear previous timings
  if (measure_time) {
    Kokkos::fence();
    clock.reset();
  }
  const std::uint64_t ncenter = centers.extent(0);
  auto rprof = DvceArray3D<Real>("radial_profile", ncenter, nfields, num_bins);
  const auto scatter = Kokkos::Experimental::create_scatter_view(rprof);
  phase(timings.allocation);
  if (ncenter == 0) {
    Kokkos::fence();
    if (measure_time) timings.total = clock.seconds();
    return rprof;
  }

  // Capture variables
  DvceArray5D<Real> u0;
  auto *pack = mesh_->pmb_pack;
  if (pack->phydro != nullptr) {
    u0 = pack->phydro->u0;
  } else if (pack->pmhd != nullptr) {
    u0 = pack->pmhd->u0;
  }
  const auto &mesh_size = mesh_->mesh_size;
  const Real dvol = mesh_size.dx1*mesh_size.dx2*mesh_size.dx3;
  const Real lx1 = mesh_size.x1max - mesh_size.x1min;
  const Real lx2 = mesh_size.x2max - mesh_size.x2min;
  const Real lx3 = mesh_size.x3max - mesh_size.x3min;
  const auto meshblock_sizes = mesh_->pmb_pack->pmb->mb_size.d_view;
  const auto indcs = mesh_->mb_indcs;
  const std::uint64_t ncells = std::uint64_t(indcs.nx1)*indcs.nx2*indcs.nx3;
  const std::uint64_t nmb = mesh_->pmb_pack->nmb_thispack;
  if (ncenter*nmb*ncells > std::numeric_limits<std::int64_t>::max()) {
    Fail("cell loop overflow");
  }
  const Real dr = bin_width;
  const int nbins = num_bins;
  par_for<std::int64_t>("radial_profile_accumulate", DevExeSpace(),
      0, static_cast<int>(ncenter)-1, 0, static_cast<int>(nmb)-1,
      0, indcs.nx3-1, 0, indcs.nx2-1, 0, indcs.nx1-1,
      KOKKOS_LAMBDA(int c, int m, int k, int j, int i) {
    const auto &block_size = meshblock_sizes(m);
    const auto &center = centers(c);
    Real x = CellCenterX(i, indcs.nx1, block_size.x1min, block_size.x1max) - center.x1;
    Real y = CellCenterX(j, indcs.nx2, block_size.x2min, block_size.x2max) - center.x2;
    Real z = CellCenterX(k, indcs.nx3, block_size.x3min, block_size.x3max) - center.x3;
    x -= lx1*round(x/lx1);
    y -= lx2*round(y/lx2);
    z -= lx3*round(z/lx3);
    const int bin = BinIndex(sqrt(x*x+y*y+z*z), dr);
    if (bin < nbins) {
      auto sum = scatter.access();
      sum(c, shell_volume, bin) += dvol;
      sum(c, shell_mass, bin) += u0(m, IDN, k+indcs.ks, j+indcs.js, i+indcs.is)*dvol;
      // TODO Add more fields...
    }
  });
  Kokkos::Experimental::contribute(rprof, scatter);
  Kokkos::fence();  // complete device writes before MPI reads this memory
  phase(timings.accumulation);
  // Reduce across MPI ranks, if any.
  // Only rank 0's result is globally normalized and ready on return.
#if MPI_PARALLEL_ENABLED
  const int count = static_cast<int>(ncenter*nfields*num_bins);
  const int status = MPI_Reduce(
      (global_variable::my_rank == 0) ? MPI_IN_PLACE : rprof.data(), // send buffer
      (global_variable::my_rank == 0) ? rprof.data() : nullptr,      // recv buffer
      count, MPI_ATHENA_REAL, MPI_SUM, 0, MPI_COMM_WORLD
  );
  if (status != MPI_SUCCESS) Fail("device-buffer MPI_Reduce failed");
#endif
  phase(timings.reduction);
  if (global_variable::my_rank == 0) {
    par_for("radial_profile_normalize", DevExeSpace(),
        0, static_cast<int>(ncenter)-1, 0, nbins-1,
        KOKKOS_LAMBDA(int c, int bin) {
      const Real vshell = rprof(c, shell_volume, bin);
      rprof(c, density, bin) = rprof(c, shell_mass, bin)/vshell;
      // TODO Add more fields...
    });
  }
  Kokkos::fence();  // results ready for caller, including asynchronous GPU normalization
  phase(timings.normalization);
  if (measure_time) timings.total = clock.seconds();

  return rprof;
}
