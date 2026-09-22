#ifndef UTILS_RADIAL_PROFILE_HPP_
#define UTILS_RADIAL_PROFILE_HPP_

#include <cstdint>

#include "athena.hpp"
#include "Kokkos_ScatterView.hpp"

class Mesh;

struct RadialProfileCenter {
  std::uint64_t id;
  Real x1, x2, x3;
  Real vx, vy, vz;
};


// Radial shell averages on uniform, cubic-cell, periodic Cartesian meshes.
// The caller supplies identical ordered centers, including current center-cell
// velocities, on every MPI rank. Only rank 0's
// returned view is globally normalized and ready on return. Returned views own
// their data and remain valid across subsequent Compute calls.
// Centers must be ready before Compute and unchanged until it returns. Their view
// extent is the active count; the calculator neither modifies nor retains them.
//
// Profile centers must be mesh cell centers. Even nsub avoids singular sample
// positions in the central subcell region. Outer polar-axis cells use the same
// subdivision to average spherical projections and their moments within each
// cell, retaining the parent radial bin and weight. The caller must ensure
// rmax <= half the shortest box length and synchronized fluid, bcc0 and potential
// data, including valid first-layer potential ghosts. Compute never solves gravity.
// Position moments and transport use sample positions; this mixed central/outer
// estimator is not an exact nonoverlapping volume partition.
// Optional gravity/MHD slots are zero when absent; a future writer omits them.

class RadialProfile {
 public:
  enum Field {
    shell_volume = 0,
    shell_mass = 1,
    density = 2,
    density_sq = 3,
    velocity_x = 4,
    velocity_y = 5,
    velocity_z = 6,
    velocity_xy = 7,
    velocity_xz = 8,
    velocity_yz = 9,
    velocity_x_sq = 10,
    velocity_y_sq = 11,
    velocity_z_sq = 12,
    velocity_mass_weighted_x = 13,
    velocity_mass_weighted_y = 14,
    velocity_mass_weighted_z = 15,
    velocity_mass_weighted_xy = 16,
    velocity_mass_weighted_xz = 17,
    velocity_mass_weighted_yz = 18,
    velocity_mass_weighted_x_sq = 19,
    velocity_mass_weighted_y_sq = 20,
    velocity_mass_weighted_z_sq = 21,
    angular_momentum_density_x = 22,
    angular_momentum_density_y = 23,
    angular_momentum_density_z = 24,
    velocity_1 = 25,
    velocity_2 = 26,
    velocity_3 = 27,
    velocity_mass_weighted_1 = 28,
    velocity_mass_weighted_2 = 29,
    velocity_mass_weighted_3 = 30,
    velocity_1_sq = 31,
    velocity_2_sq = 32,
    velocity_3_sq = 33,
    velocity_mass_weighted_1_sq = 34,
    velocity_mass_weighted_2_sq = 35,
    velocity_mass_weighted_3_sq = 36,
    mass_flux_in = 37,
    mass_flux_out = 38,
    bfield_x = 39,
    bfield_y = 40,
    bfield_z = 41,
    bfield_x_sq = 42,
    bfield_y_sq = 43,
    bfield_z_sq = 44,
    bfield_1 = 45,
    bfield_2 = 46,
    bfield_3 = 47,
    bfield_1_sq = 48,
    bfield_2_sq = 49,
    bfield_3_sq = 50,
    potential_mass_weighted = 51,
    gravity_1 = 52,
    gravity_mass_weighted_1 = 53,
    fraction_negative_gravity_1 = 54,
    rhoxgx = 55,
    rhoygy = 56,
    rhozgz = 57,
    nfields = 58
  };
  RadialProfile(Mesh *mesh, Real rmax, int nbins_subcell = 4, int nsub = 4, bool mpi_allreduce = true);
  DvceArray3D<Real> Compute(const DvceArray1D<const RadialProfileCenter>& centers);

  // Optional fenced phase measurements. Disabled for ordinary/correctness calls.
  bool measure_time = false;
  struct Timings {
    double allocation = 0, accumulation = 0;
    double reduction = 0, normalization = 0, total = 0;
  } timings;

  const Real bin_width;
  const int num_bins;
  const int num_bins_subcell;
  const int nsub;
  bool use_allreduce;

 private:
  Mesh *mesh_;

};
#endif  // UTILS_RADIAL_PROFILE_HPP_
