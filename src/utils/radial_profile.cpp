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

DvceArray5D<Real> Primitives(Mesh *mesh) {
  auto *pack = mesh->pmb_pack;
  DvceArray5D<Real> primitive;
  if (pack->phydro != nullptr) {
    primitive = pack->phydro->w0;
  } else if (pack->pmhd != nullptr) {
    primitive = pack->pmhd->w0;
  }
  const auto &b = mesh->mb_indcs;
  if (primitive.extent(0) < static_cast<std::size_t>(pack->nmb_thispack) ||
      primitive.extent(1) <= IDN || primitive.extent(2) <= b.ke ||
      primitive.extent(3) <= b.je || primitive.extent(4) <= b.ie) {
    Fail("an available hydro or MHD primitive density array is required");
  }
  return primitive;
}
}  // namespace

RadialProfile::RadialProfile(Mesh *mesh, Real requested_rmax)
    : dr(mesh->mesh_size.dx1),
      nr(static_cast<int>(std::floor((requested_rmax - 0.5*dr) / dr)) + 1),
      mesh_(mesh) {
}

void RadialProfile::Compute(const std::vector<RadialProfileCenter>& centers) {
  timings = {};
  Kokkos::Timer clock;
  double start = 0;

  // Local timer lambda
  auto phase = [&](double &seconds) {
    if (measure_time) {
      Kokkos::fence();
      const double now = clock.seconds();
      seconds = now - start;
      start = now;
    }
  };

  if (measure_time) { Kokkos::fence(); clock.reset(); }
  const std::uint64_t ncenter = centers.size();
  if (result.extent(0) != ncenter) {
    result = DvceArray3D<Real>("radial_profile", ncenter, nfields, nr);
    scatter_ = Scatter(result);
    centers_ = DvceArray1D<RadialProfileCenter>("radial_centers", ncenter);
    host_centers_ = Kokkos::create_mirror_view(centers_);
  }
  phase(timings.allocation);
  if (centers.empty()) {
    Kokkos::fence();
    if (measure_time) timings.total = clock.seconds();
    return;
  }
  for (std::size_t c=0; c<centers.size(); ++c) host_centers_(c) = centers[c];
  Kokkos::deep_copy(centers_, host_centers_);
  scatter_.reset();
  Kokkos::deep_copy(result, 0.0);  // reset target as well as duplicated CPU storage
  phase(timings.reset_transfer);

  const auto primitive = Primitives(mesh_);
  const auto size = mesh_->pmb_pack->pmb->mb_size.d_view;
  const auto b = mesh_->mb_indcs;
  const auto s = mesh_->mesh_size;
  const std::uint64_t nblock = mesh_->pmb_pack->nmb_thispack;
  const std::uint64_t cells_per_block = std::uint64_t(b.nx1)*b.nx2*b.nx3;
  if (nblock > 0 && ncenter > std::numeric_limits<std::uint64_t>::max() /
                              nblock / cells_per_block) Fail("cell loop overflow");
  const auto device_centers = centers_;
  const auto scatter = scatter_;
  const Real dx = dr, cell_volume = s.dx1*s.dx2*s.dx3;
  const Real length1 = s.x1max-s.x1min, length2 = s.x2max-s.x2min;
  const Real length3 = s.x3max-s.x3min;
  const int bins = nr;
  using Policy = Kokkos::RangePolicy<DevExeSpace, Kokkos::IndexType<std::uint64_t>>;
  Kokkos::parallel_for("radial_profile_accumulate",
      Policy(0, ncenter*nblock*cells_per_block), KOKKOS_LAMBDA(std::uint64_t index) {
    const int i = index % b.nx1; index /= b.nx1;
    const int j = index % b.nx2; index /= b.nx2;
    const int k = index % b.nx3; index /= b.nx3;
    const int m = index % nblock;
    const std::uint64_t c = index / nblock;
    const auto &block = size(m);
    const auto &center = device_centers(c);
    Real x = CellCenterX(i, b.nx1, block.x1min, block.x1max)-center.x1;
    Real y = CellCenterX(j, b.nx2, block.x2min, block.x2max)-center.x2;
    Real z = CellCenterX(k, b.nx3, block.x3min, block.x3max)-center.x3;
    x -= length1*round(x/length1);
    y -= length2*round(y/length2);
    z -= length3*round(z/length3);
    const Real radial_index = floor(sqrt(x*x+y*y+z*z)/dx + Real(0.5));
    if (radial_index < bins) {
      const int bin = static_cast<int>(radial_index);
      auto sum = scatter.access();
      sum(c, density, bin) += primitive(m, IDN, k+b.ks, j+b.js, i+b.is)
                              * cell_volume;
      sum(c, sampled_volume, bin) += cell_volume;
    }
  });
  Kokkos::Experimental::contribute(result, scatter_);
  Kokkos::fence();  // complete device writes before MPI reads this memory
  phase(timings.accumulation);
#if MPI_PARALLEL_ENABLED
  const int count = static_cast<int>(ncenter*nfields*nr);
  const int status = MPI_Reduce(global_variable::my_rank == 0 ? MPI_IN_PLACE : result.data(),
      result.data(), count, MPI_ATHENA_REAL, MPI_SUM, 0, MPI_COMM_WORLD);
  if (status != MPI_SUCCESS) Fail("device-buffer MPI_Reduce failed");
#endif
  phase(timings.reduction);
  if (global_variable::my_rank == 0) {
    const auto profile = result;
    const Real empty = std::numeric_limits<Real>::quiet_NaN();
    Kokkos::parallel_for("radial_profile_normalize", Policy(0, ncenter*nr),
        KOKKOS_LAMBDA(std::uint64_t index) {
      const auto c = index/bins;
      const int bin = index%bins;
      const Real volume = profile(c, sampled_volume, bin);
      profile(c, density, bin) = volume > 0 ? profile(c, density, bin)/volume : empty;
    });
  }
  Kokkos::fence();  // results ready for caller, including asynchronous GPU normalization
  phase(timings.normalization);
  if (measure_time) timings.total = clock.seconds();
}
