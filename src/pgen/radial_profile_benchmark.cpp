// Analytic static/dynamic fixture. Diagnostic text is for tests, not a persistent rprof format.
#include <sched.h>
#include <unistd.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#include <type_traits>
#include <vector>

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
  constexpr int num_centers = 100;
  const std::uint64_t nx1 = indcs.nx1, nx2 = indcs.nx2, nx3 = indcs.nx3;
  const std::uint64_t num_cells = nx1*nx2*nx3;
  if (num_cells < num_centers) {
    std::cerr << "### FATAL ERROR in radial_profile_benchmark: "
              << "at least 100 active cells are required" << std::endl;
    #if MPI_PARALLEL_ENABLED
    MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    #endif
    std::exit(EXIT_FAILURE);
  }

  // Reproduce the same synthetic, uniformly distributed centers on every rank.
  std::mt19937_64 generator(0);
  std::uniform_int_distribution<std::uint64_t> cell_id(0, num_cells - 1);
  std::vector<std::uint64_t> center_ids;
  while (center_ids.size() < num_centers) {
    const auto id = cell_id(generator);
    if (std::find(center_ids.begin(), center_ids.end(), id) == center_ids.end()) {
      center_ids.push_back(id);
    }
  }
  std::sort(center_ids.begin(), center_ids.end());

  DvceArray1D<RadialProfileCenter> centers("domain_centers", num_centers);
  auto host_centers = Kokkos::create_mirror_view(centers);
  for (int n = 0; n < num_centers; ++n) {
    const auto id = center_ids[n];
    const std::uint64_t i = id%nx1, j = (id/nx1)%nx2, k = id/(nx1*nx2);
    host_centers(n) = {
      id,
      CellCenterX(i, indcs.nx1, size.x1min, size.x1max),
      CellCenterX(j, indcs.nx2, size.x2min, size.x2max),
      CellCenterX(k, indcs.nx3, size.x3min, size.x3max),
      0.0, 0.0, 0.0  // stationary uniform benchmark
    };
  }
  Kokkos::deep_copy(centers, host_centers);
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
  const int num_centers = centers.extent(0);
  Kokkos::fence();

  RadialProfile::Timings total_timings;

  // Repeat the fixed 100-center batch, including allocations and MPI on every call.
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
  const auto &indcs = pm->mesh_indcs;
  const std::uint64_t num_cells = static_cast<std::uint64_t>(indcs.nx1) *
                                  indcs.nx2 * indcs.nx3;
  const double seconds_per_batch = max_elapsed/num_repeats;
  const double seconds_per_center = seconds_per_batch/num_centers;
  const double zcps = static_cast<double>(num_cells)/seconds_per_center;

  if (global_variable::my_rank == 0) {
    std::cout << "RadialProfile performance... " << std::endl;
    std::cout << "centers = " << num_centers << " repeats = " << num_repeats << std::endl;
    std::cout << "total elapsed time (seconds) = " << max_elapsed << std::endl;
    std::cout << "average seconds per batch = " << seconds_per_batch << std::endl;
    std::cout << "average seconds per center = " << seconds_per_center << std::endl;
    std::cout << "zone-cycles/cpu_second = " << zcps << std::endl;
    const auto &t = total_timings;
    std::cout << "Rank 0 timings (seconds per 100-center calculation): allocation=" << t.allocation/num_repeats
              << " accumulation=" << t.accumulation/num_repeats
              << " normalization=" << t.normalization/num_repeats
              << " reduction=" << t.reduction/num_repeats
              << " flux calculation=" << t.flux/num_repeats
              << " total=" << t.total/num_repeats << std::endl;
  }
}

void ProblemGenerator::UserProblem(ParameterInput *pin, const bool restart) {
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
