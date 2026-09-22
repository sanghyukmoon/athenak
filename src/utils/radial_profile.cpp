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

KOKKOS_INLINE_FUNCTION
void CartesianToSpherical(Real vx, Real vy, Real vz,
		Real cos_th, Real sin_th, Real cos_ph, Real sin_ph,
    Real &vr, Real &vtheta, Real &vphi) {
  const Real v_cyl = vx*cos_ph + vy*sin_ph;
  vr     = v_cyl*sin_th + vz*cos_th;
  vtheta = v_cyl*cos_th - vz*sin_th;
  vphi   = -vx*sin_ph + vy*cos_ph;
}
}  // namespace

RadialProfile::RadialProfile(Mesh *mesh, Real requested_rmax,
                             int corrected_bins, int num_subcells, bool mpi_allreduce)
    : bin_width(mesh->mesh_size.dx1),
      num_bins(BinIndex(requested_rmax, bin_width)),
      num_bins_subcell(corrected_bins),
      nsub(num_subcells),
      use_allreduce(mpi_allreduce),
      mesh_(mesh) {
  if (nsub % 2 != 0) Fail("nsub must be even to avoid singularity at the center");
  if (num_bins_subcell < 1) Fail("At least one subcell bin is needed to avoid singularity");
  if (nsub < 1) Fail("nsub must be positive");
  if (num_bins_subcell > num_bins) Fail("num_bins_subcell must be <= num_bins");
}


//----------------------------------------------------------------------------------------
// \!fn void RadialProfile::Compute()
// \brief
//
// The volume-weighted shell-average of a quantity Q is computed as
// <Q>_bin = sum_{ijk \in bin} Q_{ijk}*dV_{ijk} / sum_{ijk \in bin} dV_{ijk}
//         = sum_{ijk \in bin} Q_{ijk} / sum_{ijk \in bin} 1.0
//
// The mass-weighted shell-average of a quantity Q is computed as
// <Q>_bin,mw = sum_{ijk \in bin} Q_{ijk}*\rho_{ijk} / sum_{ijk \in bin} \rho_{ijk}
//
// Hence, no need to multiply cell volume here. In the normalization step, we
// simply divide each bin by "sample count" and "density sum", respectively.
// Note that the "sample count" and "density sum" are simply the shell_volume and
// shell_mass before the normalization step.
//
// Because dV_{ijk} cancels, we need not worry about the different between the
// parent cell volume and the subcell volume. The only exceptions are the actual
// shell volume and shell mass, because
//     shell_volume = sum_{ijk \in bin} dV_{ijk}
//     shell_mass   = sum_{ijk \in bin} \rho_{ijk}*dV_{ijk}
// However, because
//     dV_{ijk} = dvol             bin >= nbins_sub
//              = dvol_subcell     bin < nbins_sub
// is constant within a given bin, the dV_{ijk} factor can be factored out such that
//     shell_volume = sample_count * volume_element
//     shell_mass   = density_sum * volume_element,
// where
//     volume_element = dvol             bin >= nbins_sub
//                    = dvol_subcell     bin < nbins_sub

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
    Real rsph = sqrt(x*x+y*y+z*z);
    const int bin = BinIndex(rsph, dr);
    const bool is_valid_bin = is_subcell ? bin < nbins_sub
                                  : nbins_sub <= bin && bin < nbins;
    if (is_valid_bin) {
      const Real vx = w0(m, IVX, k, j, i);
      const Real vy = w0(m, IVY, k, j, i);
      const Real vz = w0(m, IVZ, k, j, i);
      const Real px = u0(m, IM1, k, j, i);
      const Real py = u0(m, IM2, k, j, i);
      const Real pz = u0(m, IM3, k, j, i);

      // Cartesian-to-spherical transformation for vector quantities
      // The basis vectors \hat{\theta} and \hat{\phi} are undefined at R=0; for those
      // cells, we subdivide the parent cell into nquad^3 subcells, and average the
      const bool on_axis = !is_subcell && std::abs(x) < 0.5*dx1 && std::abs(y) < 0.5*dx2;
      const int nquad = on_axis ? nsubcells : 1;
      Real v_sph[3] = {}, v_sph_sq[3] = {};
      Real p_sph[3] = {}, pv_sph[3] = {};
      for (int kk = 0; kk < nquad; ++kk) {
        for (int jj = 0; jj < nquad; ++jj) {
          for (int ii = 0; ii < nquad; ++ii) {
            Real xq = x, yq = y, zq = z;
            if (on_axis) {
              xq += ((ii+0.5)/nquad-0.5)*dx1;
              yq += ((jj+0.5)/nquad-0.5)*dx2;
              zq += ((kk+0.5)/nquad-0.5)*dx3;
              xq -= lx1*round(xq/lx1);
              yq -= lx2*round(yq/lx2);
              zq -= lx3*round(zq/lx3);
            }
            rsph = sqrt(xq*xq+yq*yq+zq*zq);
            const Real rcyl = sqrt(xq*xq+yq*yq);
            const Real cos_th = zq/rsph;
            const Real sin_th = rcyl/rsph;
            const Real cos_ph = xq/rcyl;
            const Real sin_ph = yq/rcyl;
            Real v[3], p[3];
            CartesianToSpherical(vx, vy, vz, cos_th, sin_th, cos_ph, sin_ph,
                                 v[0], v[1], v[2]);
            CartesianToSpherical(px, py, pz, cos_th, sin_th, cos_ph, sin_ph,
                                 p[0], p[1], p[2]);
            for (int a = 0; a < 3; ++a) {
              v_sph[a] += v[a];
              v_sph_sq[a] += v[a]*v[a];
              p_sph[a] += p[a];
              pv_sph[a] += p[a]*v[a];
            }
          }
        }
      }
      if (on_axis) {
        const Real samples = static_cast<Real>(nquad)*nquad*nquad;
        for (int a = 0; a < 3; ++a) {
          v_sph[a] /= samples;
          v_sph_sq[a] /= samples;
          p_sph[a] /= samples;
          pv_sph[a] /= samples;
        }
      }
      const Real &v1 = v_sph[0], &v2 = v_sph[1], &v3 = v_sph[2];
      const Real &v1_sq = v_sph_sq[0], &v2_sq = v_sph_sq[1], &v3_sq = v_sph_sq[2];
      const Real &p1 = p_sph[0], &p2 = p_sph[1], &p3 = p_sph[2];
      const Real &pv1 = pv_sph[0], &pv2 = pv_sph[1], &pv3 = pv_sph[2];

      // Add fields to the bin
      auto sum = scatter.access();
      sum(c, shell_volume, bin) += 1.0;
      sum(c, shell_mass, bin) += u0(m, IDN, k, j, i);
      sum(c, velocity_x, bin) += vx;
      sum(c, velocity_y, bin) += vy;
      sum(c, velocity_z, bin) += vz;
      sum(c, velocity_mass_weighted_x, bin) += px;
      sum(c, velocity_mass_weighted_y, bin) += py;
      sum(c, velocity_mass_weighted_z, bin) += pz;
      sum(c, velocity_x_sq, bin) += vx*vx;
      sum(c, velocity_y_sq, bin) += vy*vy;
      sum(c, velocity_z_sq, bin) += vz*vz;
      sum(c, velocity_mass_weighted_x_sq, bin) += px*vx;
      sum(c, velocity_mass_weighted_y_sq, bin) += py*vy;
      sum(c, velocity_mass_weighted_z_sq, bin) += pz*vz;
      sum(c, velocity_1, bin) += v1;
      sum(c, velocity_2, bin) += v2;
      sum(c, velocity_3, bin) += v3;
      sum(c, velocity_mass_weighted_1, bin) += p1;
      sum(c, velocity_mass_weighted_2, bin) += p2;
      sum(c, velocity_mass_weighted_3, bin) += p3;
      sum(c, velocity_1_sq, bin) += v1_sq;
      sum(c, velocity_2_sq, bin) += v2_sq;
      sum(c, velocity_3_sq, bin) += v3_sq;
      sum(c, velocity_mass_weighted_1_sq, bin) += pv1;
      sum(c, velocity_mass_weighted_2_sq, bin) += pv2;
      sum(c, velocity_mass_weighted_3_sq, bin) += pv3;
      // TODO Add more fields...
    }
    return bin;
  };
  // END_KOKKOS_LAMBDA

  // =======================================================
  // Step 1. Perform the radial binning
  // =======================================================
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
  int status;
  if (use_allreduce) {
    status = MPI_Allreduce(
        MPI_IN_PLACE, rprof.data(), count, MPI_ATHENA_REAL, MPI_SUM, MPI_COMM_WORLD
    );
  } else {
    status = MPI_Reduce(
        (global_variable::my_rank == 0) ? MPI_IN_PLACE : rprof.data(), // send buffer
        (global_variable::my_rank == 0) ? rprof.data() : nullptr,      // recv buffer
        count, MPI_ATHENA_REAL, MPI_SUM, 0, MPI_COMM_WORLD
    );
  }
  if (status != MPI_SUCCESS) Fail("device-buffer MPI reduction failed");
#endif
  phase(timings.reduction);

  // =======================================================
  // Step 2. Normalize the radial profiles
  // =======================================================
  if (global_variable::my_rank == 0) {
    par_for("radial_profile_normalize", DevExeSpace(),
        0, static_cast<int>(ncenter)-1, 0, nbins-1,
        KOKKOS_LAMBDA(int c, int bin) {
      const Real volume_element = bin < nbins_sub ? dvol_subcell : dvol;
      const Real sample_count = rprof(c, shell_volume, bin);
      const Real density_sum = rprof(c, shell_mass, bin);
      rprof(c, shell_volume, bin) = sample_count*volume_element;
      rprof(c, shell_mass, bin) = density_sum*volume_element;
      rprof(c, density, bin) = density_sum/sample_count;

      constexpr Field volume_weighted_fields[] = {
        velocity_x, velocity_y, velocity_z,
        velocity_x_sq, velocity_y_sq, velocity_z_sq,
        velocity_1, velocity_2, velocity_3,
        velocity_1_sq, velocity_2_sq, velocity_3_sq
      };
      constexpr Field mass_weighted_fields[] = {
        velocity_mass_weighted_x,
        velocity_mass_weighted_y,
        velocity_mass_weighted_z,
        velocity_mass_weighted_x_sq,
        velocity_mass_weighted_y_sq,
        velocity_mass_weighted_z_sq,
        velocity_mass_weighted_1,
        velocity_mass_weighted_2,
        velocity_mass_weighted_3,
        velocity_mass_weighted_1_sq,
        velocity_mass_weighted_2_sq,
        velocity_mass_weighted_3_sq
      };
      for (Field f : volume_weighted_fields) {
        rprof(c, f, bin) /= sample_count;
      }
      for (Field f : mass_weighted_fields) {
        rprof(c, f, bin) /= density_sum;
      }

      // TODO Add more fields...
    });
  }
  Kokkos::fence();  // results ready for caller, including asynchronous GPU normalization
  phase(timings.normalization);
  if (measure_time) timings.total = clock.seconds();

  return rprof;
}
