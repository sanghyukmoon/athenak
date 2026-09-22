#include "utils/radial_profile.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <limits>

#include "coordinates/coordinates.hpp"
#include "coordinates/cell_locations.hpp"
#include "globals.hpp"
#include "gravity/gravity.hpp"
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
  DvceArray5D<Real> u0, w0, bcc0, phi;
  const auto *pack = mesh_->pmb_pack;
  const bool mhd = pack->pmhd != nullptr;
  const bool gravity = pack->pgrav != nullptr;
  if (mhd) {
    u0 = pack->pmhd->u0;
    w0 = pack->pmhd->w0;
    bcc0 = pack->pmhd->bcc0;
  } else if (pack->phydro != nullptr) {
    u0 = pack->phydro->u0;
    w0 = pack->phydro->w0;
  } else {
    Fail("Radial profile requires either hydro or MHD turned on");
  }
  if (gravity) phi = pack->pgrav->phi;
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
      // Prepare fields
      const Real rho = u0(m, IDN, k, j, i);
      const Real vx = w0(m, IVX, k, j, i) - center.vx;
      const Real vy = w0(m, IVY, k, j, i) - center.vy;
      const Real vz = w0(m, IVZ, k, j, i) - center.vz;
      const Real px = rho*vx;
      const Real py = rho*vy;
      const Real pz = rho*vz;
      Real phi_c, gx, gy, gz;
      if (gravity) {
        phi_c = phi(m, 0, k, j, i);
        gx = -(phi(m, 0, k, j, i+1) - phi(m, 0, k, j, i-1))/(2.0*dx1);
        gy = -(phi(m, 0, k, j+1, i) - phi(m, 0, k, j-1, i))/(2.0*dx2);
        gz = -(phi(m, 0, k+1, j, i) - phi(m, 0, k-1, j, i))/(2.0*dx3);
      }
      Real bx, by, bz;
      if (mhd) {
        bx = bcc0(m, IBX, k, j, i);
        by = bcc0(m, IBY, k, j, i);
        bz = bcc0(m, IBZ, k, j, i);
      }

      // Prepare fields involving spherical vector components
      // The basis vectors \hat{\theta} and \hat{\phi} are undefined at R=0; for those
      // cells, average sample projections and nonlinear moments before scattering.
      const bool on_axis = !is_subcell && std::abs(x) < 0.5*dx1 && std::abs(y) < 0.5*dx2;
      const int nquad = on_axis ? nsubcells : 1;
      Real v_sph[3] = {}, v_sph_sq[3] = {};
      Real p_sph[3] = {}, pv_sph[3] = {};
      Real inward_mass_flux = 0.0, outward_mass_flux = 0.0;
      Real g1 = 0.0, neg_gr_flag = 0.0;
      Real b_sph[3] = {}, b_sph_sq[3] = {};
      for (int kk = 0; kk < nquad; ++kk) {
        for (int jj = 0; jj < nquad; ++jj) {
          for (int ii = 0; ii < nquad; ++ii) {
            Real xq = x, yq = y, zq = z;
            if (on_axis) {
              xq += ((ii+0.5)/nquad-0.5)*dx1;
              yq += ((jj+0.5)/nquad-0.5)*dx2;
              zq += ((kk+0.5)/nquad-0.5)*dx3;
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
            inward_mass_flux += rho*Kokkos::fmax(-v[0], 0.0);
            outward_mass_flux += rho*Kokkos::fmax(v[0], 0.0);
            if (mhd) {
              Real b[3];
              CartesianToSpherical(bx, by, bz, cos_th, sin_th, cos_ph, sin_ph,
                                   b[0], b[1], b[2]);
              for (int ax = 0; ax < 3; ++ax) {
                b_sph[ax] += b[ax];
                b_sph_sq[ax] += b[ax]*b[ax];
              }
            }
            if (gravity) {
              const Real gr = (gx*xq + gy*yq + gz*zq)/rsph;
              g1 += gr;
              neg_gr_flag += gr < 0.0 ? 1.0 : 0.0;
            }
            for (int ax = 0; ax < 3; ++ax) {
              v_sph[ax] += v[ax];
              v_sph_sq[ax] += v[ax]*v[ax];
              p_sph[ax] += p[ax];
              pv_sph[ax] += p[ax]*v[ax];
            }
          }
        }
      }
      if (on_axis) {
        const Real samples = static_cast<Real>(nquad)*nquad*nquad;
        inward_mass_flux /= samples;
        outward_mass_flux /= samples;
        g1 /= samples;
        neg_gr_flag /= samples;
        for (int ax = 0; ax < 3; ++ax) {
          b_sph[ax] /= samples;
          b_sph_sq[ax] /= samples;
          v_sph[ax] /= samples;
          v_sph_sq[ax] /= samples;
          p_sph[ax] /= samples;
          pv_sph[ax] /= samples;
        }
      }
      // Add fields to the bin
      auto sum = scatter.access();
      sum(c, shell_volume, bin) += 1.0;
      sum(c, shell_mass, bin) += rho;
      sum(c, density_sq, bin) += rho*rho;
      sum(c, velocity_x, bin) += vx;
      sum(c, velocity_y, bin) += vy;
      sum(c, velocity_z, bin) += vz;
      sum(c, velocity_xy, bin) += vx*vy;
      sum(c, velocity_xz, bin) += vx*vz;
      sum(c, velocity_yz, bin) += vy*vz;
      sum(c, velocity_x_sq, bin) += vx*vx;
      sum(c, velocity_y_sq, bin) += vy*vy;
      sum(c, velocity_z_sq, bin) += vz*vz;
      sum(c, velocity_mass_weighted_x, bin) += px;
      sum(c, velocity_mass_weighted_y, bin) += py;
      sum(c, velocity_mass_weighted_z, bin) += pz;
      sum(c, velocity_mass_weighted_xy, bin) += px*vy;
      sum(c, velocity_mass_weighted_xz, bin) += px*vz;
      sum(c, velocity_mass_weighted_yz, bin) += py*vz;
      sum(c, velocity_mass_weighted_x_sq, bin) += px*vx;
      sum(c, velocity_mass_weighted_y_sq, bin) += py*vy;
      sum(c, velocity_mass_weighted_z_sq, bin) += pz*vz;
      sum(c, angular_momentum_density_x, bin) += y*pz - z*py;
      sum(c, angular_momentum_density_y, bin) += z*px - x*pz;
      sum(c, angular_momentum_density_z, bin) += x*py - y*px;
      // Spherical components
      sum(c, velocity_1, bin) += v_sph[0];
      sum(c, velocity_2, bin) += v_sph[1];
      sum(c, velocity_3, bin) += v_sph[2];
      sum(c, velocity_mass_weighted_1, bin) += p_sph[0];
      sum(c, velocity_mass_weighted_2, bin) += p_sph[1];
      sum(c, velocity_mass_weighted_3, bin) += p_sph[2];
      sum(c, velocity_1_sq, bin) += v_sph_sq[0];
      sum(c, velocity_2_sq, bin) += v_sph_sq[1];
      sum(c, velocity_3_sq, bin) += v_sph_sq[2];
      sum(c, velocity_mass_weighted_1_sq, bin) += pv_sph[0];
      sum(c, velocity_mass_weighted_2_sq, bin) += pv_sph[1];
      sum(c, velocity_mass_weighted_3_sq, bin) += pv_sph[2];
      sum(c, mass_flux_in, bin) += inward_mass_flux;
      sum(c, mass_flux_out, bin) += outward_mass_flux;
      if (mhd) {
        sum(c, bfield_x, bin) += bx;
        sum(c, bfield_y, bin) += by;
        sum(c, bfield_z, bin) += bz;
        sum(c, bfield_x_sq, bin) += bx*bx;
        sum(c, bfield_y_sq, bin) += by*by;
        sum(c, bfield_z_sq, bin) += bz*bz;
        sum(c, bfield_1, bin) += b_sph[0];
        sum(c, bfield_2, bin) += b_sph[1];
        sum(c, bfield_3, bin) += b_sph[2];
        sum(c, bfield_1_sq, bin) += b_sph_sq[0];
        sum(c, bfield_2_sq, bin) += b_sph_sq[1];
        sum(c, bfield_3_sq, bin) += b_sph_sq[2];
      }
      if (gravity) {
        sum(c, potential_mass_weighted, bin) += rho*phi_c;
        sum(c, gravity_1, bin) += g1;
        sum(c, gravity_mass_weighted_1, bin) += rho*g1;
        sum(c, fraction_negative_gravity_1, bin) += neg_gr_flag;
        sum(c, rhoxgx, bin) += rho*x*gx;
        sum(c, rhoygy, bin) += rho*y*gy;
        sum(c, rhozgz, bin) += rho*z*gz;
      }
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
  constexpr Field volume_weighted_fields[] = {
    density_sq,
    velocity_x, velocity_y, velocity_z,
    velocity_xy, velocity_xz, velocity_yz,
    velocity_x_sq, velocity_y_sq, velocity_z_sq,
    angular_momentum_density_x, angular_momentum_density_y, angular_momentum_density_z,
    velocity_1, velocity_2, velocity_3,
    velocity_1_sq, velocity_2_sq, velocity_3_sq,
    mass_flux_in, mass_flux_out,
    bfield_x, bfield_y, bfield_z,
    bfield_x_sq, bfield_y_sq, bfield_z_sq,
    bfield_1, bfield_2, bfield_3,
    bfield_1_sq, bfield_2_sq, bfield_3_sq,
    gravity_1, fraction_negative_gravity_1
  };
  constexpr Field mass_weighted_fields[] = {
    velocity_mass_weighted_x, velocity_mass_weighted_y, velocity_mass_weighted_z,
    velocity_mass_weighted_xy, velocity_mass_weighted_xz, velocity_mass_weighted_yz,
    velocity_mass_weighted_x_sq, velocity_mass_weighted_y_sq, velocity_mass_weighted_z_sq,
    velocity_mass_weighted_1, velocity_mass_weighted_2, velocity_mass_weighted_3,
    velocity_mass_weighted_1_sq, velocity_mass_weighted_2_sq, velocity_mass_weighted_3_sq,
    potential_mass_weighted, gravity_mass_weighted_1,
    rhoxgx, rhoygy, rhozgz
  };
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
      for (Field f : volume_weighted_fields) {
        rprof(c, f, bin) /= sample_count;
      }
      for (Field f : mass_weighted_fields) {
        rprof(c, f, bin) /= density_sum;
      }
    });
  }
  Kokkos::fence();  // results ready for caller, including asynchronous GPU normalization
  phase(timings.normalization);
  if (measure_time) timings.total = clock.seconds();

  return rprof;
}
