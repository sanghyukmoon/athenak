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


DvceArray1D<RadialProfileCenter> DomainCenter(Mesh *pm) {
  const auto &size = pm->mesh_size;
  const auto &indcs = pm->mesh_indcs;
  const std::uint64_t i = indcs.nx1/2, j = indcs.nx2/2, k = indcs.nx3/2;
  DvceArray1D<RadialProfileCenter> centers("domain_center", 1);
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
  constexpr int num_repeats = 100;
  Real rmax = pin->GetReal("problem", "rmax");
  int nbins_subcell_corrected = pin->GetInteger("problem", "nbins_subcell_corrected");
  int nsub = pin->GetInteger("problem", "nsub");

  RadialProfile radial_profile(pm, rmax, nbins_subcell_corrected, nsub);
  radial_profile.measure_time = true;
  const auto centers = DomainCenter(pm);
  Kokkos::fence();

  RadialProfile::Timings total_timings;

  // Measure 100 successive single-center calculations, including per-call allocations.
  #if MPI_PARALLEL_ENABLED
  MPI_Barrier(MPI_COMM_WORLD);
  #endif
  Kokkos::Timer clock;
  for (int repeat = 0; repeat < num_repeats; ++repeat) {
    radial_profile.Compute(centers);
    const auto &t = radial_profile.timings;
    total_timings.allocation += t.allocation;
    total_timings.accumulation += t.accumulation;
    total_timings.normalization += t.normalization;
    total_timings.reduction += t.reduction;
    total_timings.flux += t.flux;
    total_timings.total += t.total;
  }
  Kokkos::fence();
  const double elapsed = clock.seconds();
  double max_elapsed = elapsed;
  #if MPI_PARALLEL_ENABLED
  MPI_Allreduce(&elapsed, &max_elapsed, 1, MPI_DOUBLE,
                MPI_MAX, MPI_COMM_WORLD);
  #endif
  std::uint64_t zonecycles = static_cast<uint64_t>(pm->nmb_total) * num_repeats *
                             pm->NumberOfMeshBlockCells();
  double zcps = static_cast<double>(zonecycles) / max_elapsed;

  if (global_variable::my_rank == 0) {
    std::cout << "RadialProfile performance... " << std::endl;
    std::cout << "cpu time used  = " << max_elapsed << std::endl;
    std::cout << "zone-cycles/cpu_second = " << zcps << std::endl;
    const auto &t = total_timings;
    std::cout << "Rank 0 timings (seconds per calculation): allocation=" << t.allocation/num_repeats
              << " accumulation=" << t.accumulation/num_repeats
              << " normalization=" << t.normalization/num_repeats
              << " reduction=" << t.reduction/num_repeats
              << " flux calculation=" << t.flux/num_repeats
              << " total=" << t.total/num_repeats << std::endl;
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
