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
  const auto &indcs = mesh->mb_indcs;
  if (primitive.extent(0) < static_cast<std::size_t>(pack->nmb_thispack) ||
      primitive.extent(1) <= IDN || primitive.extent(2) <= indcs.ke ||
      primitive.extent(3) <= indcs.je || primitive.extent(4) <= indcs.ie) {
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

void RadialProfile::Compute(const DvceArray1D<const RadialProfileCenter>& centers) {
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
  const std::uint64_t ncenter = centers.extent(0);
  result = DvceArray3D<Real>("radial_profile", ncenter, nfields, nr);
  const auto scatter = Kokkos::Experimental::create_scatter_view(result);
  phase(timings.allocation);
  if (ncenter == 0) {
    Kokkos::fence();
    if (measure_time) timings.total = clock.seconds();
    return;
  }
  phase(timings.reset);

  const auto primitive = Primitives(mesh_);
  const auto mbsize = mesh_->pmb_pack->pmb->mb_size.d_view;
  const auto indcs = mesh_->mb_indcs;
  const auto &ms = mesh_->mesh_size;
  const Real dvol = ms.dx1*ms.dx2*ms.dx3;
  const Real lx1 = ms.x1max - ms.x1min;
  const Real lx2 = ms.x2max - ms.x2min;
  const Real lx3 = ms.x3max - ms.x3min;
  const std::uint64_t nmb = mesh_->pmb_pack->nmb_thispack;
  const std::uint64_t ncells = std::uint64_t(indcs.nx1)*indcs.nx2*indcs.nx3;
  if (ncenter*nmb*ncells > std::numeric_limits<std::int64_t>::max()) {
    Fail("cell loop overflow");
  }
  const Real dx = dr;
  const int bins = nr;
  par_for<std::int64_t>("radial_profile_accumulate", DevExeSpace(),
      0, static_cast<int>(ncenter)-1, 0, static_cast<int>(nmb)-1,
      0, indcs.nx3-1, 0, indcs.nx2-1, 0, indcs.nx1-1,
      KOKKOS_LAMBDA(int c, int m, int k, int j, int i) {
    const auto &block = mbsize(m);
    const auto &center = centers(c);
    Real x = CellCenterX(i, indcs.nx1, block.x1min, block.x1max)-center.x1;
    Real y = CellCenterX(j, indcs.nx2, block.x2min, block.x2max)-center.x2;
    Real z = CellCenterX(k, indcs.nx3, block.x3min, block.x3max)-center.x3;
    x -= lx1*round(x/lx1);
    y -= lx2*round(y/lx2);
    z -= lx3*round(z/lx3);
    const Real radial_index = floor(sqrt(x*x+y*y+z*z)/dx + Real(0.5));
    if (radial_index < bins) {
      const int bin = static_cast<int>(radial_index);
      auto sum = scatter.access();
      sum(c, density, bin) += primitive(m, IDN, k+indcs.ks, j+indcs.js, i+indcs.is)
                              * dvol;
      sum(c, sampled_volume, bin) += dvol;
    }
  });
  Kokkos::Experimental::contribute(result, scatter);
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
    par_for("radial_profile_normalize", DevExeSpace(),
        0, static_cast<int>(ncenter)-1, 0, bins-1,
        KOKKOS_LAMBDA(int c, int bin) {
      const Real volume = profile(c, sampled_volume, bin);
      profile(c, density, bin) = volume > 0 ? profile(c, density, bin)/volume : empty;
    });
  }
  Kokkos::fence();  // results ready for caller, including asynchronous GPU normalization
  phase(timings.normalization);
  if (measure_time) timings.total = clock.seconds();
}
