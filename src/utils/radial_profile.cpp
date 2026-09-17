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
                             int corrected_bins, int num_subcells)
    : bin_width(mesh->mesh_size.dx1),
      num_bins(BinIndex(requested_rmax, bin_width)),
      num_bins_subcell(corrected_bins),
      nsub(num_subcells),
      mesh_(mesh) {
  if (num_bins_subcell < 0) Fail("num_bins_subcell must be nonnegative");
  if (nsub < 1) Fail("nsub must be positive");
  if (num_bins_subcell > num_bins) {
    Fail("num_bins_subcell must be <= num_bins");
  }
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
  const int ncenter = centers.extent(0);
  auto rprof = DvceArray3D<Real>("radial_profile", ncenter, nfields, num_bins);
  const auto scatter = Kokkos::Experimental::create_scatter_view(rprof);
  const int side = 2*num_bins_subcell+1;
  DvceArray1D<SubcellParent> parent_cells = DvceArray1D<SubcellParent>(
      "subcell_parents", ncenter*side*side*side
  );
  Kokkos::DualView<int> parent_count("subcell_parent_count");
  auto parent_count_dview = parent_count.view_device();
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
  if (static_cast<std::uint64_t>(ncenter)*nmb*ncells > std::numeric_limits<std::int64_t>::max()) {
    Fail("cell loop overflow");
  }
  const Real dr = bin_width;
  const int nbins = num_bins;
  const int nbins_sub = num_bins_subcell;
  const Real dx1 = mesh_size.dx1, dx2 = mesh_size.dx2, dx3 = mesh_size.dx3;
  const Real dvol_subcell = dvol/nsub/nsub/nsub;
  const int nsubcells = nsub;

  // Kokkos lambda function to find bin and dump the cell data into that bin
  // Negative isub indicates that this is a parent cell, not a subcell.
  const auto DumpCellToBin = KOKKOS_LAMBDA(
      int c, int m, int k, int j, int i, int ksub, int jsub, int isub) -> int {
    const bool is_subcell = isub >= 0;
    const auto &block_size = meshblock_sizes(m);
    const auto &center = centers(c);
    Real x = CellCenterX(i-indcs.is, indcs.nx1, block_size.x1min, block_size.x1max) - center.x1;
    Real y = CellCenterX(j-indcs.js, indcs.nx2, block_size.x2min, block_size.x2max) - center.x2;
    Real z = CellCenterX(k-indcs.ks, indcs.nx3, block_size.x3min, block_size.x3max) - center.x3;
    if (is_subcell) {
      x += ((isub+0.5)/nsubcells-0.5)*dx1;
      y += ((jsub+0.5)/nsubcells-0.5)*dx2;
      z += ((ksub+0.5)/nsubcells-0.5)*dx3;
    }
    x -= lx1*round(x/lx1);
    y -= lx2*round(y/lx2);
    z -= lx3*round(z/lx3);
    const int bin = BinIndex(sqrt(x*x+y*y+z*z), dr);
    const bool is_valid_bin = is_subcell ? bin < nbins_sub
                                  : nbins_sub <= bin && bin < nbins;
    if (is_valid_bin) {
      const Real volume = is_subcell ? dvol_subcell : dvol;
      auto sum = scatter.access();
      sum(c, shell_volume, bin) += volume;
      sum(c, shell_mass, bin) += u0(m, IDN, k, j, i)*volume;
      sum(c, velocity_x, bin) += w0(m, IVX, k, j, i)*volume;
      sum(c, velocity_y, bin) += w0(m, IVY, k, j, i)*volume;
      sum(c, velocity_z, bin) += w0(m, IVZ, k, j, i)*volume;
      sum(c, velocity_mass_weighted_x, bin) += u0(m, IM1, k, j, i)*volume;
      sum(c, velocity_mass_weighted_y, bin) += u0(m, IM2, k, j, i)*volume;
      sum(c, velocity_mass_weighted_z, bin) += u0(m, IM3, k, j, i)*volume;
      // TODO Add more fields...
    }
    return bin;
  };
  // END_KOKKOS_LAMBDA

  // Now, perform the actual calculation
  par_for<std::int64_t>("radial_binning", DevExeSpace(),
      0, static_cast<int>(ncenter)-1, 0, static_cast<int>(nmb)-1,
      indcs.ks, indcs.ke, indcs.js, indcs.je, indcs.is, indcs.ie,
      KOKKOS_LAMBDA(int c, int m, int k, int j, int i) {
    const int bin = DumpCellToBin(c, m, k, j, i, -1, -1, -1);
    // Record parent cell information needed for the subsequent subcell corection.
    if (nbins_sub > 0 && bin <= nbins_sub) {
      const int idx = Kokkos::atomic_fetch_add(&parent_count_dview(), 1);
      parent_cells(idx) = {c, m, k, j, i};
    }
  });
  parent_count.modify_device();
  parent_count.sync_host();
  par_for("subcell_correction", DevExeSpace(), 0, parent_count.view_host()()-1,
      0, nsub-1, 0, nsub-1, 0, nsub-1,
      KOKKOS_LAMBDA(int idx, int ksub, int jsub, int isub) {
    const auto &parent = parent_cells(idx);
    DumpCellToBin(parent.c, parent.m, parent.k, parent.j, parent.i, ksub, jsub, isub);
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

  // Normalize radial profiles
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
