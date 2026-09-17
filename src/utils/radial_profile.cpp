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

struct SubcellParent {
  int c, m, k, j, i;
};

int CheckedProduct(int left, int right) {
  if (left > std::numeric_limits<int>::max()/right) {
    Fail("subcell list or iteration count exceeds int range");
  }
  return left*right;
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

RadialProfile::RadialProfile(Mesh *mesh, Real requested_rmax,
                             int corrected_bins, int subdivisions)
    : bin_width(mesh->mesh_size.dx1),
      num_bins(BinIndex(requested_rmax, bin_width)),
      nbins_subcell(corrected_bins),
      nsub(subdivisions),
      mesh_(mesh) {
  if (nbins_subcell < 0) Fail("nbins_subcell must be nonnegative");
  if (nsub < 1) Fail("nsub must be positive");
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
  const int corrected_bins = nbins_subcell;
  const int subdivisions = nsub;
  DvceArray1D<SubcellParent> parents;
  DvceArray1D<int> parent_count;
  if (corrected_bins > 0 && ncenter > 0) {
    const int max_int = std::numeric_limits<int>::max();
    if (corrected_bins > (max_int-1)/2 || ncenter > max_int) {
      Fail("subcell list capacity exceeds int range");
    }
    const int side = 2*corrected_bins+1;
    int capacity = static_cast<int>(ncenter);
    for (int axis=0; axis<3; ++axis) capacity = CheckedProduct(capacity, side);
    parents = DvceArray1D<SubcellParent>("subcell_parents", capacity);
    parent_count = DvceArray1D<int>("subcell_parent_count", 1);
    Kokkos::deep_copy(parent_count, 0);
  }
  phase(timings.allocation);
  if (ncenter == 0) {
    Kokkos::fence();
    if (measure_time) timings.total = clock.seconds();
    return rprof;
  }

  // Capture variables
  DvceArray5D<Real> u0, w0;
  auto *pack = mesh_->pmb_pack;
  if (pack->phydro != nullptr) {
    u0 = pack->phydro->u0;
    w0 = pack->phydro->w0;
  } else if (pack->pmhd != nullptr) {
    u0 = pack->pmhd->u0;
    w0 = pack->pmhd->w0;
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
      indcs.ks, indcs.ke, indcs.js, indcs.je, indcs.is, indcs.ie,
      KOKKOS_LAMBDA(int c, int m, int k, int j, int i) {
    const auto &block_size = meshblock_sizes(m);
    const auto &center = centers(c);
    Real x = CellCenterX(i-indcs.is, indcs.nx1, block_size.x1min, block_size.x1max) - center.x1;
    Real y = CellCenterX(j-indcs.js, indcs.nx2, block_size.x2min, block_size.x2max) - center.x2;
    Real z = CellCenterX(k-indcs.ks, indcs.nx3, block_size.x3min, block_size.x3max) - center.x3;
    x -= lx1*round(x/lx1);
    y -= lx2*round(y/lx2);
    z -= lx3*round(z/lx3);
    const int bin = BinIndex(sqrt(x*x+y*y+z*z), dr);
    if (corrected_bins <= bin && bin < nbins) {
      auto sum = scatter.access();
      sum(c, shell_volume, bin) += dvol;
      sum(c, shell_mass, bin) += u0(m, IDN, k, j, i)*dvol;
      sum(c, velocity_x, bin) += w0(m, IVX, k, j, i)*dvol;
      sum(c, velocity_y, bin) += w0(m, IVY, k, j, i)*dvol;
      sum(c, velocity_z, bin) += w0(m, IVZ, k, j, i)*dvol;
      sum(c, velocity_mass_weighted_x, bin) += u0(m, IM1, k, j, i)*dvol;
      sum(c, velocity_mass_weighted_y, bin) += u0(m, IM2, k, j, i)*dvol;
      sum(c, velocity_mass_weighted_z, bin) += u0(m, IM3, k, j, i)*dvol;
      // TODO Add more fields...
    }
    // Include the next parent shell: its subcells can enter a corrected bin.
    if (corrected_bins > 0 && bin <= corrected_bins) {
      const int entry = Kokkos::atomic_fetch_add(&parent_count(0), 1);
      parents(entry) = {c, m, k, j, i};
    }
  });
  if (corrected_bins > 0) {
    const auto count_host = Kokkos::create_mirror_view_and_copy(
        Kokkos::HostSpace(), parent_count);
    const int count = count_host(0);
    if (count > 0) {
      int iterations = count;
      for (int axis=0; axis<3; ++axis) {
        iterations = CheckedProduct(iterations, subdivisions);
      }
      const Real dx1 = mesh_size.dx1, dx2 = mesh_size.dx2, dx3 = mesh_size.dx3;
      const Real subcell_volume = dvol/subdivisions/subdivisions/subdivisions;
      par_for("radial_profile_subcells", DevExeSpace(), 0, count-1,
          0, subdivisions-1, 0, subdivisions-1, 0, subdivisions-1,
          KOKKOS_LAMBDA(int entry, int ksub, int jsub, int isub) {
        const auto &parent = parents(entry);
        const int c = parent.c, m = parent.m;
        const int k = parent.k, j = parent.j, i = parent.i;
        const auto &block_size = meshblock_sizes(m);
        const auto &center = centers(c);
        Real x = CellCenterX(i-indcs.is, indcs.nx1,
                            block_size.x1min, block_size.x1max) - center.x1
                 + ((isub+0.5)/subdivisions-0.5)*dx1;
        Real y = CellCenterX(j-indcs.js, indcs.nx2,
                            block_size.x2min, block_size.x2max) - center.x2
                 + ((jsub+0.5)/subdivisions-0.5)*dx2;
        Real z = CellCenterX(k-indcs.ks, indcs.nx3,
                            block_size.x3min, block_size.x3max) - center.x3
                 + ((ksub+0.5)/subdivisions-0.5)*dx3;
        x -= lx1*round(x/lx1);
        y -= lx2*round(y/lx2);
        z -= lx3*round(z/lx3);
        const int bin = BinIndex(sqrt(x*x+y*y+z*z), dr);
        if (bin < corrected_bins && bin < nbins) {
          auto sum = scatter.access();
          sum(c, shell_volume, bin) += subcell_volume;
          sum(c, shell_mass, bin) += u0(m, IDN, k, j, i)*subcell_volume;
          sum(c, velocity_x, bin) += w0(m, IVX, k, j, i)*subcell_volume;
          sum(c, velocity_y, bin) += w0(m, IVY, k, j, i)*subcell_volume;
          sum(c, velocity_z, bin) += w0(m, IVZ, k, j, i)*subcell_volume;
          sum(c, velocity_mass_weighted_x, bin) += u0(m, IM1, k, j, i)*subcell_volume;
          sum(c, velocity_mass_weighted_y, bin) += u0(m, IM2, k, j, i)*subcell_volume;
          sum(c, velocity_mass_weighted_z, bin) += u0(m, IM3, k, j, i)*subcell_volume;
        }
      });
    }
  }
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
      const Real mshell = rprof(c, shell_mass, bin);
      rprof(c, density, bin) = rprof(c, shell_mass, bin)/vshell;
      rprof(c, velocity_x, bin) = rprof(c, velocity_x, bin)/vshell;
      rprof(c, velocity_y, bin) = rprof(c, velocity_y, bin)/vshell;
      rprof(c, velocity_z, bin) = rprof(c, velocity_z, bin)/vshell;
      rprof(c, velocity_mass_weighted_x, bin) = rprof(c, velocity_mass_weighted_x, bin)/mshell;
      rprof(c, velocity_mass_weighted_y, bin) = rprof(c, velocity_mass_weighted_y, bin)/mshell;
      rprof(c, velocity_mass_weighted_z, bin) = rprof(c, velocity_mass_weighted_z, bin)/mshell;
      // TODO Add more fields...
    });
  }
  Kokkos::fence();  // results ready for caller, including asynchronous GPU normalization
  phase(timings.normalization);
  if (measure_time) timings.total = clock.seconds();

  return rprof;
}
