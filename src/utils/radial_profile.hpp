#ifndef UTILS_RADIAL_PROFILE_HPP_
#define UTILS_RADIAL_PROFILE_HPP_

#include <cstdint>
#include <vector>

#include "athena.hpp"
#include "Kokkos_ScatterView.hpp"

class Mesh;

struct RadialProfileCenter {
  std::uint64_t id;
  Real x1, x2, x3;
};

using RadialProfileCenterFnPtr = std::vector<RadialProfileCenter> (*)(Mesh *pm);

// Whole-cell density on uniform, cubic-cell, periodic Cartesian meshes.
// The caller supplies identical ordered centers on every MPI rank. Only rank 0's
// result is globally normalized and ready on return; it is valid until Compute.
class RadialProfile {
 public:
  enum Field {density = 0, sampled_volume = 1, nfields = 2};
  RadialProfile(Mesh *mesh, Real rmax);
  void Compute(const std::vector<RadialProfileCenter>& centers);

  const Real dr, rmax;  // requested maximum OUTER edge, not last radial coordinate
  const int nr;
  const Real final_edge;
  DvceArray3D<Real> result;  // (center, field, radial_bin); empty for no centers

  // Optional fenced phase measurements. Disabled for ordinary/correctness calls.
  bool measure_time = false;
  struct Timings {
    double allocation = 0, reset_transfer = 0, accumulation = 0;
    double reduction = 0, normalization = 0, total = 0;
  } timings;

 private:
  using Scatter = Kokkos::Experimental::ScatterView<Real***, LayoutWrapper>;
  Mesh *mesh_;
  DvceArray1D<RadialProfileCenter> centers_;
  DvceArray1D<RadialProfileCenter>::HostMirror host_centers_;
  Scatter scatter_;
};
#endif  // UTILS_RADIAL_PROFILE_HPP_
