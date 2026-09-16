// Static analytic fixture. Diagnostic text is for tests, not a persistent rprof format.
#include <sys/resource.h>

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
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

namespace {
RadialProfileCenter MidpointCell(const RegionSize &box, const RegionIndcs &grid) {
  const std::uint64_t nx = grid.nx1, ny = grid.nx2;
  const std::uint64_t i = nx/2, j = ny/2, k = std::uint64_t(grid.nx3)/2;
  return {i + nx*(j + ny*k),
          box.x1min + (i+0.5)*(box.x1max-box.x1min)/grid.nx1,
          box.x2min + (j+0.5)*(box.x2max-box.x2min)/grid.nx2,
          box.x3min + (k+0.5)*(box.x3max-box.x3min)/grid.nx3};
}

std::vector<RadialProfileCenter> DomainCenter(Mesh *pm) {
  return {MidpointCell(pm->mesh_size, pm->mesh_indcs)};
}

void WriteProfile(const std::string &label, const RadialProfile &profile,
                  const std::vector<RadialProfileCenter> &centers) {
  if (global_variable::my_rank != 0) return;
  const auto host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), profile.result);
  std::ofstream out("profile_"+label+".txt");
  out << std::setprecision(17);
  out << "# dr nr requested_rmax final_edge ncenter\n"
      << profile.dr << ' ' << profile.nr << ' ' << profile.rmax << ' '
      << profile.final_edge << ' ' << centers.size() << '\n';
  for (std::size_t c=0; c<centers.size(); ++c) {
    const auto &center = centers[c];
    out << "# center " << center.id << ' ' << center.x1 << ' ' << center.x2
        << ' ' << center.x3 << '\n';
    for (int bin=0; bin<profile.nr; ++bin) {
      out << bin*profile.dr << ' ' << host(c, RadialProfile::density, bin) << ' '
          << host(c, RadialProfile::sampled_volume, bin) << '\n';
    }
  }
  if (!out) Kokkos::abort("cannot write radial profile diagnostic");
}

void PoisonGhostDensity(Mesh *pm) {
  auto *pack = pm->pmb_pack;
  auto w = pack->phydro != nullptr ? pack->phydro->w0 : pack->pmhd->w0;
  const auto b = pm->mb_indcs;
  par_for("poison_rprof_ghosts", DevExeSpace(), 0, pack->nmb_thispack-1,
      0, static_cast<int>(w.extent(2))-1, 0, static_cast<int>(w.extent(3))-1,
      0, static_cast<int>(w.extent(4))-1, KOKKOS_LAMBDA(int m, int k, int j, int i) {
    if (i<b.is || i>b.ie || j<b.js || j>b.je || k<b.ks || k>b.ke) {
      w(m, IDN, k, j, i) = 1.0e20;
    }
  });
  Kokkos::fence();
}

void BenchmarkRadialProfile(ParameterInput *pin, Mesh *pm) {
  const auto &s = pm->mesh_size;
  const Real requested = pin->GetOrAddReal("problem", "rmax",
      0.5*std::min({s.x1max-s.x1min, s.x2max-s.x2min, s.x3max-s.x3min}));
  const bool performance = pin->GetOrAddBoolean("problem", "performance", false);
  const auto centers = pm->pgen->rprof_center_func(pm);  // provider excluded from timing
  Kokkos::fence();
  Kokkos::Timer setup;
  RadialProfile profile(pm, requested);
  Kokkos::fence();
  const double setup_seconds = performance ? setup.seconds() : 0;
  if (performance) {
    profile.measure_time = true;
    const int repeats = pin->GetOrAddInteger("problem", "repeats", 20);
    if (repeats < 1) Kokkos::abort("benchmark requires at least one repeat");
    std::vector<RadialProfile::Timings> times;
    times.reserve(repeats+1);
    for (int call=0; call<=repeats; ++call) {
      profile.Compute(centers);
      times.push_back(profile.timings);
    }
    // All host copies, memory queries, and writing occur outside measured regions.
    const std::string rank = std::to_string(global_variable::my_rank);
    std::ofstream out("timings_rank"+rank+".csv");
    out << std::setprecision(17)
        << "call,setup,allocation,reset_transfer,accumulation,reduction,normalization,total\n";
    for (std::size_t call=0; call<times.size(); ++call) {
      const auto &t = times[call];
      out << call << ',' << (call == 0 ? setup_seconds : 0) << ',' << t.allocation << ','
          << t.reset_transfer << ',' << t.accumulation << ',' << t.reduction << ','
          << t.normalization << ',' << t.total << '\n';
    }
    struct rusage usage;
    getrusage(RUSAGE_SELF, &usage);
    std::ofstream memory("memory_rank"+rank+".txt");
    memory << "peak_rss_kib " << usage.ru_maxrss << '\n';
#ifdef KOKKOS_ENABLE_CUDA
    std::size_t free_bytes, total_bytes;
    if (cudaMemGetInfo(&free_bytes, &total_bytes) != cudaSuccess) {
      Kokkos::abort("cudaMemGetInfo failed");
    }
    memory << "device_used_bytes " << total_bytes-free_bytes << '\n'
           << "device_total_bytes " << total_bytes << '\n';
#endif
    if (!out || !memory) Kokkos::abort("cannot write benchmark diagnostics");
    WriteProfile("performance", profile, centers);
    return;
  }

  std::ofstream placement("placement_rank"+std::to_string(global_variable::my_rank)+".txt");
  placement << "blocks " << pm->pmb_pack->nmb_thispack << '\n'
            << "execution_concurrency " << DevExeSpace().concurrency() << '\n';
#ifdef KOKKOS_ENABLE_CUDA
  int device;
  char bus[32];
  if (cudaGetDevice(&device) != cudaSuccess ||
      cudaDeviceGetPCIBusId(bus, sizeof(bus), device) != cudaSuccess) {
    Kokkos::abort("cannot inspect CUDA device placement");
  }
  placement << "cuda_device " << device << '\n' << "pci_bus " << bus << '\n';
#endif
  if (!placement) Kokkos::abort("cannot write device placement");
  PoisonGhostDensity(pm);
  profile.Compute({});
  if (profile.result.size() != 0) Kokkos::abort("initial empty result is not empty");
  WriteProfile("initial_empty", profile, {});
  profile.Compute(centers);
  WriteProfile("single", profile, centers);
  profile.Compute(centers);
  WriteProfile("repeat", profile, centers);
  // Deliberately non-sorted order; the calculator must preserve caller ordering.
  auto corner = RadialProfileCenter{0, s.x1min+0.5*s.dx1,
                                      s.x2min+0.5*s.dx2, s.x3min+0.5*s.dx3};
  auto boundary = RadialProfileCenter{std::uint64_t(pm->mesh_indcs.nx1)-1,
                                      s.x1min, s.x2min, s.x3min};
  std::vector<RadialProfileCenter> three{centers[0], boundary, corner};
  profile.Compute(three);
  WriteProfile("three", profile, three);
  std::rotate(three.begin(), three.begin()+1, three.end());
  profile.Compute(three);  // same allocation size, new center ordering
  WriteProfile("reordered", profile, three);
  std::rotate(three.begin(), three.end()-1, three.end());
  three.resize(1);
  profile.Compute(three);
  WriteProfile("shrink", profile, three);
  profile.Compute({});
  if (profile.result.size() != 0) Kokkos::abort("empty centers gave nonempty results");
  WriteProfile("empty", profile, {});
  profile.Compute({});
  profile.Compute(centers);
  WriteProfile("restored", profile, centers);

  RegionIndcs large = pm->mesh_indcs;
  large.nx1 = 65536; large.nx2 = 65536; large.nx3 = 4;
  const auto synthetic = MidpointCell(s, large);
  if (global_variable::my_rank == 0) {
    std::ofstream out("large_id.txt");
    out << std::setprecision(17) << synthetic.id << ' ' << synthetic.x1 << ' '
        << synthetic.x2 << ' ' << synthetic.x3 << '\n';
  }
}
}  // namespace

void ProblemGenerator::UserProblem(ParameterInput *pin, const bool restart) {
  rprof_center_func = DomainCenter;
  pgen_final_func = BenchmarkRadialProfile;
  if (restart) return;
  auto *pack = pmy_mesh_->pmb_pack;
  if (pin->GetString("time", "evolution") != "static" ||
      (pack->phydro == nullptr && pack->pmhd == nullptr)) {
    Kokkos::abort("radial_profile_benchmark requires static hydro or MHD");
  }
  auto *eos = pack->phydro != nullptr ? pack->phydro->peos : pack->pmhd->peos;
  if (eos->eos_data.is_ideal) Kokkos::abort("benchmark requires isothermal EOS");
  const bool varying = pin->GetOrAddBoolean("problem", "varying", true);
  const auto b = pmy_mesh_->mb_indcs;
  const auto size = pack->pmb->mb_size.d_view;
  auto u = pack->phydro != nullptr ? pack->phydro->u0 : pack->pmhd->u0;
  Kokkos::deep_copy(u, 0.0);
  if (pack->pmhd != nullptr) {
    Kokkos::deep_copy(pack->pmhd->b0.x1f, 0.0);
    Kokkos::deep_copy(pack->pmhd->b0.x2f, 0.0);
    Kokkos::deep_copy(pack->pmhd->b0.x3f, 0.0);
  }
  par_for("radial_profile_initial_density", DevExeSpace(), 0, pack->nmb_thispack-1,
      b.ks, b.ke, b.js, b.je, b.is, b.ie, KOKKOS_LAMBDA(int m, int k, int j, int i) {
    const auto &s = size(m);
    const Real x = CellCenterX(i-b.is, b.nx1, s.x1min, s.x1max);
    const Real y = CellCenterX(j-b.js, b.nx2, s.x2min, s.x2max);
    const Real z = CellCenterX(k-b.ks, b.nx3, s.x3min, s.x3max);
    u(m, IDN, k, j, i) = varying ?
        1.0 + 0.1*cos(M_PI*x/2) + 0.05*sin(M_PI*y/2) + 0.025*cos(M_PI*z/2) : 1.0;
  });
}
