#ifndef UTILS_RADIAL_PROFILE_HPP_
#define UTILS_RADIAL_PROFILE_HPP_

#include <cstdint>

#include "athena.hpp"
#include "Kokkos_ScatterView.hpp"

class Mesh;

struct RadialProfileCenter {
  std::uint64_t id;
  Real x1, x2, x3;
};


// Radial shell averages on uniform, cubic-cell, periodic Cartesian meshes.
// The caller supplies identical ordered centers on every MPI rank. Only rank 0's
// returned view is globally normalized and ready on return. Returned views own
// their data and remain valid across subsequent Compute calls.
// Centers must be ready before Compute and unchanged until it returns. Their view
// extent is the active count; the calculator neither modifies nor retains them.
//
// To avoid coordinate singularity, the central cell is subdivided into nsub^3 subcells,
// where nsub is even. If the radial profile center is at the cell center, no subcell
// center will coincide with the singularity. If the radial profile center is not at the
// cell center, the radial profile calculation may encounter the singularity.
//
// Along the z axis, the azimuth angle is undefined; we may choose to perform similar
// subcell correction there; for now, we take arbitrary phi=0 convention at z axis.
// Beyond the subcell correction region, the contribution of this on-axis cell will be
// small.

class RadialProfile {
 public:
   // TODO is it possible to get length of enum, instead of assigning nfield here?
  enum Field {
    shell_volume = 0,
    shell_mass = 1,
    density = 2,
    velocity_x = 3,
    velocity_y = 4,
    velocity_z = 5,
    velocity_mass_weighted_x = 6,
    velocity_mass_weighted_y = 7,
    velocity_mass_weighted_z = 8,
    velocity_x_sq = 9,
    velocity_y_sq = 10,
    velocity_z_sq = 11,
    velocity_mass_weighted_x_sq = 12,
    velocity_mass_weighted_y_sq = 13,
    velocity_mass_weighted_z_sq = 14,
    velocity_1 = 15,
    velocity_2 = 16,
    velocity_3 = 17,
    velocity_mass_weighted_1 = 18,
    velocity_mass_weighted_2 = 19,
    velocity_mass_weighted_3 = 20,
    velocity_1_sq = 21,
    velocity_2_sq = 22,
    velocity_3_sq = 23,
    velocity_mass_weighted_1_sq = 24,
    velocity_mass_weighted_2_sq = 25,
    velocity_mass_weighted_3_sq = 26,
    nfields = 27
  };
  RadialProfile(Mesh *mesh, Real rmax, int nbins_subcell = 4, int nsub = 4);
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

 private:
  Mesh *mesh_;

};
#endif  // UTILS_RADIAL_PROFILE_HPP_
