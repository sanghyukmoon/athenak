// Analytic static/dynamic fixture. Diagnostic text is for tests, not a persistent rprof format.
#include <sched.h>
#include <unistd.h>

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <type_traits>

#include "athena.hpp"
#include "coordinates/cell_locations.hpp"
#include "mesh/mesh.hpp"
#include "eos/eos.hpp"
#include "globals.hpp"
#include "hydro/hydro.hpp"
#include "mhd/mhd.hpp"
#include "pgen/pgen.hpp"
#include "utils/radial_profile.hpp"
#include "bvals/bvals.hpp"
#if MPI_PARALLEL_ENABLED
#include <mpi.h>
#endif


int num_centers = 100;
DvceArray1D<RadialProfileCenter> DomainCenter(Mesh *pm) {
  const auto &size = pm->mesh_size;
  const auto &indcs = pm->mesh_indcs;
  const std::uint64_t i = indcs.nx1/2, j = indcs.nx2/2, k = indcs.nx3/2;
  DvceArray1D<RadialProfileCenter> centers("domain_center", num_centers);
  RadialProfileCenter center = {
    i + indcs.nx1*(j + indcs.nx2*k),
    CellCenterX(i, indcs.nx1, size.x1min, size.x1max),
    CellCenterX(j, indcs.nx2, size.x2min, size.x2max),
    CellCenterX(k, indcs.nx3, size.x3min, size.x3max),
    0.0, 0.0, 0.0  // stationary uniform benchmark
  };
  Kokkos::deep_copy(centers, center);
  return centers;
}

void BenchmarkRadialProfile(ParameterInput *pin, Mesh *pm) {
  Real rmax = pin->GetReal("problem", "rmax");
  int nbins_subcell = pin->GetInteger("problem", "nbins_subcell");
  int nsub = pin->GetInteger("problem", "nsub");
  bool mpi_allreduce = pin->GetOrAddBoolean("problem", "mpi_allreduce", true);

  RadialProfile radial_profile(pm, rmax, nbins_subcell, nsub, mpi_allreduce);
  radial_profile.measure_time = true;
  const auto centers = DomainCenter(pm);
  Kokkos::fence();

  // Measure time
  #if MPI_PARALLEL_ENABLED
  MPI_Barrier(MPI_COMM_WORLD);
  #endif
  Kokkos::Timer clock;
  radial_profile.Compute(centers);
  Kokkos::fence();
  const double elapsed = clock.seconds();
  double max_elapsed = elapsed;
  #if MPI_PARALLEL_ENABLED
  if (mpi_allreduce) {
    MPI_Allreduce(&elapsed, &max_elapsed, 1, MPI_DOUBLE,
                  MPI_MAX, MPI_COMM_WORLD);
  } else {
    MPI_Reduce(&elapsed, &max_elapsed, 1, MPI_DOUBLE,
               MPI_MAX, 0, MPI_COMM_WORLD);
  }
  #endif
  std::uint64_t zonecycles = static_cast<uint64_t>(pm->nmb_total) * num_centers *
                             pm->NumberOfMeshBlockCells();
  double zcps = static_cast<double>(zonecycles) / max_elapsed;

  if (global_variable::my_rank == 0) {
    std::cout << "RadialProfile performance... " << std::endl;
    std::cout << "cpu time used  = " << max_elapsed << std::endl;
    std::cout << "zone-cycles/cpu_second = " << zcps << std::endl;
    const auto &t = radial_profile.timings;
    std::cout << "Rank 0 timings (seconds per center): allocation=" << t.allocation/num_centers
              << " accumulation=" << t.accumulation/num_centers
              << " reduction=" << t.reduction/num_centers
              << " normalization=" << t.normalization/num_centers
              << " total=" << t.total/num_centers << std::endl;
  }
}

void ProblemGenerator::UserProblem(ParameterInput *pin, const bool restart) {
  rprof_center_func = DomainCenter;
  pgen_final_func = BenchmarkRadialProfile;
  if (restart) return;
  auto *pack = pmy_mesh_->pmb_pack;
  auto u0 = pack->phydro != nullptr ? pack->phydro->u0 : pack->pmhd->u0;
  Kokkos::deep_copy(u0, 0.0);
  Kokkos::deep_copy(Kokkos::subview(u0, Kokkos::ALL(), static_cast<int>(IDN),
                    Kokkos::ALL(), Kokkos::ALL(), Kokkos::ALL()), 1.0);
  if (pack->pmhd != nullptr) {
    Kokkos::deep_copy(pack->pmhd->b0.x1f, 0.0);
    Kokkos::deep_copy(pack->pmhd->b0.x2f, 0.0);
    Kokkos::deep_copy(pack->pmhd->b0.x3f, 1.0);
  }
}
