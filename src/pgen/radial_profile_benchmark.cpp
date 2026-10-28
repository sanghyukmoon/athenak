// Static/dynamic radial-profile performance benchmark with synthetic centers.
#include <algorithm>
#include <cstdlib>
#include <cstdint>
#include <iostream>
#include <random>
#include <vector>

#include "athena.hpp"
#include "coordinates/cell_locations.hpp"
#include "mesh/mesh.hpp"
#include "globals.hpp"
#include "hydro/hydro.hpp"
#include "mhd/mhd.hpp"
#include "pgen/pgen.hpp"
#include "utils/radial_profile.hpp"
#if MPI_PARALLEL_ENABLED
#include <mpi.h>
#endif


DvceArray1D<RadialProfileCenter> DomainCenter(Mesh *pm, int num_centers) {
  const auto &size = pm->mesh_size;
  const auto &indcs = pm->mesh_indcs;
  const std::uint64_t nx1 = indcs.nx1, nx2 = indcs.nx2, nx3 = indcs.nx3;
  const std::uint64_t num_cells = nx1*nx2*nx3;
  if (num_centers <= 0 || static_cast<std::uint64_t>(num_centers) > num_cells) {
    std::cerr << "### FATAL ERROR in radial_profile_benchmark: "
              << "problem/num_centers must be positive and no larger than "
              << "the global active-cell count (" << num_cells << "), got "
              << num_centers << std::endl;
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
  const int num_centers = pin->GetInteger("problem", "num_centers");
  const int num_repeats = pin->GetInteger("problem", "num_repeats");
  Real rmax = pin->GetReal("problem", "rmax");
  int nbins_subcell_corrected = pin->GetInteger("problem", "nbins_subcell_corrected");
  int nsub = pin->GetInteger("problem", "nsub");

  RadialProfile radial_profile(pm, rmax, nbins_subcell_corrected, nsub);
  const auto centers = DomainCenter(pm, num_centers);

  // Warm up outside timing
  radial_profile.measure_time = false;
  radial_profile.Compute(centers);
  Kokkos::fence();

  // Time whole calls, including allocation, MPI, flux, and result destruction.
  // Phase instrumentation and center generation are excluded.
  #if MPI_PARALLEL_ENABLED
  MPI_Barrier(MPI_COMM_WORLD);
  #endif
  Kokkos::Timer run_time;
  for (int repeat = 0; repeat < num_repeats; ++repeat) {
    radial_profile.Compute(centers);
  }
  Kokkos::fence();
  float exe_time_thisrank = run_time.seconds();
  float exe_time = exe_time_thisrank;
  #if MPI_PARALLEL_ENABLED
  MPI_Allreduce(&exe_time_thisrank, &exe_time, 1, MPI_FLOAT, MPI_MAX, MPI_COMM_WORLD);
  #endif

  int nmb = pm->nmb_total;
  std::uint64_t zonecycles = num_repeats * nmb * static_cast<uint64_t>(pm->NumberOfMeshBlockCells());
  float zcps = static_cast<float>(zonecycles) / exe_time;

  if (global_variable::my_rank == 0) {
    std::cout << "RadialProfile performance for " << num_centers << " centers..." << std::endl;
    std::cout << "cpu time used  = " << exe_time << std::endl;
    std::cout << "equivalent zone-cycles/cpu_second = " << zcps << std::endl;
  }

  // Phase-separated diagnostics
  #if MPI_PARALLEL_ENABLED
  MPI_Barrier(MPI_COMM_WORLD);
  #endif
  radial_profile.measure_time = true;
  radial_profile.Compute(centers);
  Kokkos::fence();
  const auto &t = radial_profile.timings;
  const char *phase_names[] = {
    "allocation", "accumulation", "shell_reduction", "normalization",
    "magnetic_flux", "total"
  };
  const double exe_time_by_phase_local[] = {
    t.allocation, t.accumulation, t.reduction, t.normalization, t.flux, t.total
  };
  double exe_time_by_phase[6];
  #if MPI_PARALLEL_ENABLED
  MPI_Reduce(exe_time_by_phase_local, exe_time_by_phase, 6, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
  #else
  std::copy(exe_time_by_phase_local, exe_time_by_phase_local + 6, exe_time_by_phase);
  #endif

  if (global_variable::my_rank == 0) {
    std::cout << "Rprof performance phase by phase" << std::endl;
    for (int phase = 0; phase < 6; ++phase) {
      std::cout << "rprof.phase." << phase_names[phase] << ".max_seconds = "
                << exe_time_by_phase[phase] << std::endl;
    }
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
