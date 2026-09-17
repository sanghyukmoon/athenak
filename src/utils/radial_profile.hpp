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

// Providers return a managed device view with all center writes completed.
using RadialProfileCenterFnPtr = DvceArray1D<RadialProfileCenter> (*)(Mesh *pm);

// Whole-cell density on uniform, cubic-cell, periodic Cartesian meshes.
// The caller supplies identical ordered centers on every MPI rank. Only rank 0's
// result is globally normalized and ready on return; it is valid until Compute.
// Centers must be ready before Compute and unchanged until it returns. Their view
// extent is the active count; the calculator neither modifies nor retains them.
class RadialProfile {
 public:
   // TODO is it possible to get length of enum, instead of assigning nfield here?
  enum Field {
    shell_volume = 0,
    shell_mass = 1,
    density = 2,
    nfields = 3};
  RadialProfile(Mesh *mesh, Real rmax);
  DvceArray3D<Real> Compute(const DvceArray1D<const RadialProfileCenter>& centers);

  // Optional fenced phase measurements. Disabled for ordinary/correctness calls.
  bool measure_time = false;
  struct Timings {
    double allocation = 0, accumulation = 0;
    double reduction = 0, normalization = 0, total = 0;
  } timings;

  const Real bin_width;
  const int num_bins;

 private:
  Mesh *mesh_;

};
#endif  // UTILS_RADIAL_PROFILE_HPP_
