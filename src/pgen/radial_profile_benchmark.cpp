// Analytic static/dynamic fixture. Diagnostic text is for tests, not a persistent rprof format.
#include <sys/resource.h>
#include <sched.h>
#include <unistd.h>

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>
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
#include "utils/benchmark_timer.hpp"
#include "bvals/bvals.hpp"

namespace {
int center_count = 1;
bool subcell_fixture = false;
RadialProfileCenter MidpointCell(const RegionSize &box, const RegionIndcs &grid) {
  const std::uint64_t nx = grid.nx1, ny = grid.nx2;
  const std::uint64_t i = nx/2, j = ny/2, k = std::uint64_t(grid.nx3)/2;
  return {i + nx*(j + ny*k),
          box.x1min + (i+0.5)*(box.x1max-box.x1min)/grid.nx1,
          box.x2min + (j+0.5)*(box.x2max-box.x2min)/grid.nx2,
          box.x3min + (k+0.5)*(box.x3max-box.x3min)/grid.nx3};
}

DvceArray1D<RadialProfileCenter> DomainCenter(Mesh *pm) {
  DualArray1D<RadialProfileCenter> centers("domain_center",
                                           subcell_fixture ? 3 : center_count);
  const auto &box = pm->mesh_size;
  const auto &grid = pm->mesh_indcs;
  const int side = center_count == 64 ? 4 : 2;
  if (subcell_fixture) {
    const auto midpoint = MidpointCell(box, grid);
    centers.view_host()(0) = midpoint;
    centers.view_host()(1) = {0, box.x1min, box.x2min, box.x3min};
    centers.view_host()(2) = {1, midpoint.x1+0.31*box.dx1,
                               midpoint.x2-0.27*box.dx2, midpoint.x3+0.43*box.dx3};
  } else if (center_count == 1) {
    centers.view_host()(0) = MidpointCell(box, grid);
  } else {
    for (int z=0, c=0; z<side; ++z) {
      for (int y=0; y<side; ++y) {
        for (int x=0; x<side; ++x, ++c) {
          const std::uint64_t i = (2*x+1)*grid.nx1/(2*side);
          const std::uint64_t j = (2*y+1)*grid.nx2/(2*side);
          const std::uint64_t k = (2*z+1)*grid.nx3/(2*side);
          centers.view_host()(c) = {i + std::uint64_t(grid.nx1)*(j+grid.nx2*k),
              box.x1min+(i+0.5)*box.dx1, box.x2min+(j+0.5)*box.dx2,
              box.x3min+(k+0.5)*box.dx3};
        }
      }
    }
  }
  centers.modify_host();
  centers.sync_device();
  return centers.view_device();
}

void WriteProfile(const std::string &label, const RadialProfile &profile,
                  const DvceArray3D<Real> &rprof,
                  const DvceArray1D<const RadialProfileCenter> &centers,
                  Real requested_rmax) {
  if (global_variable::my_rank != 0) return;
  const auto host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), rprof);
  const auto host_centers = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), centers);
  std::ofstream out("profile_"+label+".txt");
  out << std::setprecision(17);
  out << "# bin_width num_bins requested_rmax final_edge ncenter\n"
      << profile.bin_width << ' ' << profile.num_bins << ' ' << requested_rmax << ' '
      << (profile.num_bins-0.5)*profile.bin_width << ' ' << centers.extent(0) << '\n';
  out << "# nbins_subcell nsub\n" << "# " << profile.nbins_subcell << ' '
      << profile.nsub << '\n';
  for (std::size_t c=0; c<centers.extent(0); ++c) {
    const auto &center = host_centers(c);
    out << "# center " << center.id << ' ' << center.x1 << ' ' << center.x2
        << ' ' << center.x3 << '\n';
    for (int bin=0; bin<profile.num_bins; ++bin) {
      out << bin*profile.bin_width << ' ' << host(c, RadialProfile::density, bin) << ' '
          << host(c, RadialProfile::shell_volume, bin) << ' '
          << host(c, RadialProfile::shell_mass, bin) << ' '
          << host(c, RadialProfile::velocity_x, bin) << ' '
          << host(c, RadialProfile::velocity_y, bin) << ' '
          << host(c, RadialProfile::velocity_z, bin) << ' '
          << host(c, RadialProfile::velocity_mass_weighted_x, bin) << ' '
          << host(c, RadialProfile::velocity_mass_weighted_y, bin) << ' '
          << host(c, RadialProfile::velocity_mass_weighted_z, bin) << '\n';
    }
  }
  if (!out) Kokkos::abort("cannot write radial profile diagnostic");
}

void PoisonGhostDensity(Mesh *pm) {
  auto *pack = pm->pmb_pack;
  auto u0 = pack->phydro != nullptr ? pack->phydro->u0 : pack->pmhd->u0;
  const auto b = pm->mb_indcs;
  par_for("poison_rprof_ghosts", DevExeSpace(), 0, pack->nmb_thispack-1,
      0, static_cast<int>(u0.extent(2))-1, 0, static_cast<int>(u0.extent(3))-1,
      0, static_cast<int>(u0.extent(4))-1, KOKKOS_LAMBDA(int m, int k, int j, int i) {
    if (i<b.is || i>b.ie || j<b.js || j>b.je || k<b.ks || k>b.ke) {
      u0(m, IDN, k, j, i) = 1.0e20;
    }
  });
  Kokkos::fence();
}

void BenchmarkRadialProfile(ParameterInput *pin, Mesh *pm) {
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
  cpu_set_t affinity;
  CPU_ZERO(&affinity);
  sched_getaffinity(0, sizeof(affinity), &affinity);
  char hostname[256];
  gethostname(hostname, sizeof(hostname));
  placement << "hostname " << hostname << '\n' << "cpu_count " << CPU_COUNT(&affinity)
            << '\n' << "cpu_ids ";
  for (int cpu=0; cpu<CPU_SETSIZE; ++cpu) {
    if (CPU_ISSET(cpu, &affinity)) placement << cpu << ' ';
  }
  placement << '\n';
  if (!placement) Kokkos::abort("cannot write device placement");
  if (pin->GetOrAddBoolean("problem", "check_evolution", false)) {
    const auto *mhd = pm->pmb_pack->pmhd;
    const auto u = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), mhd->u0);
    const auto bcc = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), mhd->bcc0);
    const auto &b = pm->mb_indcs;
    double mass = 0, min_density = 1e100;
    bool finite = true;
    for (int m=0; m<pm->pmb_pack->nmb_thispack; ++m) {
      for (int k=b.ks; k<=b.ke; ++k) for (int j=b.js; j<=b.je; ++j) {
        for (int i=b.is; i<=b.ie; ++i) {
          mass += u(m, IDN, k, j, i)*pm->mesh_size.dx1*
                  pm->mesh_size.dx2*pm->mesh_size.dx3;
          min_density = std::min(min_density, double(u(m, IDN, k, j, i)));
          for (int n=0; n<4; ++n) finite &= std::isfinite(u(m,n,k,j,i));
          for (int n=0; n<3; ++n) finite &= std::isfinite(bcc(m,n,k,j,i));
        }
      }
    }
    std::ofstream state("state_rank"+std::to_string(global_variable::my_rank)+".txt");
    state << std::setprecision(17) << "mass " << mass << '\n'
          << "min_density " << min_density << '\n' << "finite " << finite << '\n'
          << "cycles " << pm->ncycle << '\n';
  }
  if (pm->pmb_pack->pmhd != nullptr) {
    const auto *m = pm->pmb_pack->pmhd;
    std::ofstream allocation("allocations_rank"+
                            std::to_string(global_variable::my_rank)+".csv");
    allocation << "category,array,bytes\n";
    auto record = [&](const char *category, const char *name, const auto &view) {
      using Value = typename std::decay_t<decltype(view)>::non_const_value_type;
      allocation << category << ',' << name << ',' << view.size()*sizeof(Value) << '\n';
    };
    record("state", "u0", m->u0);
    record("state", "w0", m->w0);
    record("state", "bcc0", m->bcc0);
    record("state", "b0.x1f", m->b0.x1f);
    record("state", "b0.x2f", m->b0.x2f);
    record("state", "b0.x3f", m->b0.x3f);
    record("integration", "u1", m->u1);
    record("integration", "b1.x1f", m->b1.x1f);
    record("integration", "b1.x2f", m->b1.x2f);
    record("integration", "b1.x3f", m->b1.x3f);
    record("integration", "uflx.x1f", m->uflx.x1f);
    record("integration", "uflx.x2f", m->uflx.x2f);
    record("integration", "uflx.x3f", m->uflx.x3f);
    record("integration", "efld.x1e", m->efld.x1e);
    record("integration", "efld.x2e", m->efld.x2e);
    record("integration", "efld.x3e", m->efld.x3e);
    record("integration", "e3x1", m->e3x1);
    record("integration", "e2x1", m->e2x1);
    record("integration", "e1x2", m->e1x2);
    record("integration", "e3x2", m->e3x2);
    record("integration", "e2x3", m->e2x3);
    record("integration", "e1x3", m->e1x3);
    // These three private arrays have the same cell shape as e1x2 (mhd.cpp).
    allocation << "integration,private_cell_E_source_shape,"
               << 3*m->e1x2.size()*sizeof(Real) << '\n';
    record("reconstruction", "wl3d", m->wl3d);
    record("reconstruction", "wr3d", m->wr3d);
    record("reconstruction", "bl3d", m->bl3d);
    record("reconstruction", "br3d", m->br3d);
#if MPI_PARALLEL_ENABLED
    for (auto *boundary : {static_cast<MeshBoundaryValues*>(m->pbval_u),
                           static_cast<MeshBoundaryValues*>(m->pbval_b)}) {
      record("communication", "rank_send_vars", boundary->rank_sendbuf_vars_);
      record("communication", "rank_recv_vars", boundary->rank_recvbuf_vars_);
      record("communication", "send_offsets", boundary->send_agg_offset_);
      record("communication", "recv_offsets", boundary->recv_agg_offset_);
      record("host_communication", "send_headers", boundary->rank_sendhdr_vars_);
      record("host_communication", "recv_headers", boundary->rank_recvhdr_vars_);
    }
#endif
    for (int n=0; n<56; ++n) {
      for (auto *boundary : {static_cast<MeshBoundaryValues*>(m->pbval_u),
                             static_cast<MeshBoundaryValues*>(m->pbval_b)}) {
        record("communication", "send_vars", boundary->sendbuf[n].vars);
        record("communication", "recv_vars", boundary->recvbuf[n].vars);
        record("communication", "send_flux", boundary->sendbuf[n].flux);
        record("communication", "recv_flux", boundary->recvbuf[n].flux);
      }
    }
  }
  if (!pin->GetOrAddBoolean("problem", "profiles", true)) return;
  const auto &s = pm->mesh_size;
  const Real requested = pin->GetOrAddReal("problem", "rmax",
      0.5*std::min({s.x1max-s.x1min, s.x2max-s.x2min, s.x3max-s.x3min}));
  const bool performance = pin->GetOrAddBoolean("problem", "performance", false);
  const auto centers = pm->pgen->rprof_center_func(pm);  // provider excluded from timing
  if (!performance) {
    const auto host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), centers);
    std::ofstream metadata("centers_rank"+std::to_string(global_variable::my_rank)+".txt");
    metadata << std::setprecision(17);
    for (std::size_t c=0; c<centers.extent(0); ++c) {
      metadata << host(c).id << ' ' << host(c).x1 << ' ' << host(c).x2 << ' '
               << host(c).x3 << '\n';
    }
  }
  Kokkos::fence();
  RadialProfile profile(pm, requested,
      pin->GetOrAddInteger("problem", "nbins_subcell", 4),
      pin->GetOrAddInteger("problem", "nsub", 4));
  Kokkos::fence();
  profile.measure_time = performance &&
      pin->GetOrAddBoolean("problem", "phase_timings", false);
  DvceArray3D<Real> rprof;
  if (performance) {
    BenchmarkTimer timer(pin->GetOrAddInteger("problem", "repeats", 20),
                         pin->GetOrAddReal("problem", "warmup_seconds", 0.0));
    std::vector<RadialProfile::Timings> phases;
    while (!timer.Done()) {
      rprof = {};  // release previous result before fences, barrier and timer
      timer.Start();
      rprof = profile.Compute(centers);
      timer.Stop();  // includes Compute return and local ScatterView destruction
      phases.push_back(profile.timings);
    }
    timer.Write("profile_timings", rprof.size()*sizeof(Real));
    if (profile.measure_time) {
      std::ofstream out("phases_rank"+std::to_string(global_variable::my_rank)+".csv");
      out << std::setprecision(17)
          << "sample,allocation,accumulation,reduction,normalization,total\n";
      for (std::size_t i=0; i<phases.size(); ++i) {
        const auto &t = phases[i];
        out << i << ',' << t.allocation << ',' << t.accumulation << ','
            << t.reduction << ',' << t.normalization << ',' << t.total << '\n';
      }
      if (!out) Kokkos::abort("cannot write profile phases");
    }
    return;
  }

  if (pin->GetOrAddBoolean("problem", "accuracy_only", false)) {
    rprof = profile.Compute(centers);
    WriteProfile("single", profile, rprof, centers, requested);
    return;
  }
  PoisonGhostDensity(pm);
  rprof = profile.Compute({});
  if (rprof.size() != 0) Kokkos::abort("initial empty result is not empty");
  WriteProfile("initial_empty", profile, rprof, {}, requested);
  rprof = profile.Compute(centers);
  WriteProfile("single", profile, rprof, centers, requested);
  const auto retained_single = rprof;
  rprof = profile.Compute(centers);
  WriteProfile("repeat", profile, rprof, centers, requested);
  // Deliberately non-sorted order; the calculator must preserve caller ordering.
  const auto midpoint = MidpointCell(s, pm->mesh_indcs);
  const auto corner = RadialProfileCenter{0, s.x1min+0.5*s.dx1,
                                           s.x2min+0.5*s.dx2, s.x3min+0.5*s.dx3};
  const auto boundary = RadialProfileCenter{std::uint64_t(pm->mesh_indcs.nx1)-1,
                                            s.x1min, s.x2min, s.x3min};
  DualArray1D<RadialProfileCenter> three("three_centers", 3);
  const auto host_three = three.view_host();
  host_three(0) = midpoint;
  host_three(1) = boundary;
  host_three(2) = corner;
  three.modify_host();
  three.sync_device();
  rprof = profile.Compute(three.view_device());
  WriteProfile("three", profile, rprof, three.view_device(), requested);
  std::rotate(host_three.data(), host_three.data()+1, host_three.data()+3);
  three.modify_host();
  three.sync_device();
  rprof = profile.Compute(three.view_device());  // same allocation size, new center ordering
  WriteProfile("reordered", profile, rprof, three.view_device(), requested);
  // A later call changes the first center; the retained result must not change.
  WriteProfile("retained_single", profile, retained_single, centers, requested);

  // Exercise future GPU-provider data flow with prescribed centers, not minima.
  const DvceArray1D<RadialProfileCenter> device_three("device_three_centers", 3);
  par_for("generate_rprof_centers", DevExeSpace(), 0, 0,
      KOKKOS_LAMBDA(int) {
    device_three(0) = midpoint;
    device_three(1) = boundary;
    device_three(2) = corner;
  });
  Kokkos::fence();  // provider contract: centers ready before Compute
  rprof = profile.Compute(device_three);
  WriteProfile("device_three", profile, rprof, device_three, requested);
  par_for("reorder_rprof_centers", DevExeSpace(), 0, 0,
      KOKKOS_LAMBDA(int) {
    const auto first = device_three(0);
    device_three(0) = device_three(1);
    device_three(1) = device_three(2);
    device_three(2) = first;
  });
  Kokkos::fence();
  rprof = profile.Compute(device_three);
  WriteProfile("device_reordered", profile, rprof, device_three, requested);

  rprof = profile.Compute(centers);
  WriteProfile("shrink", profile, rprof, centers, requested);
  rprof = profile.Compute({});
  if (rprof.size() != 0) Kokkos::abort("empty centers gave nonempty results");
  WriteProfile("empty", profile, rprof, {}, requested);
  rprof = profile.Compute({});
  rprof = profile.Compute(centers);
  WriteProfile("restored", profile, rprof, centers, requested);

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
  subcell_fixture = pin->GetOrAddBoolean("problem", "subcell_fixture", false);
  center_count = pin->GetOrAddInteger("problem", "center_count", 1);
  if (center_count != 1 && center_count != 8 && center_count != 64) {
    Kokkos::abort("center_count must be 1, 8 or 64");
  }
  rprof_center_func = DomainCenter;
  pgen_final_func = BenchmarkRadialProfile;
  if (restart) return;
  auto *pack = pmy_mesh_->pmb_pack;
  if (pack->phydro == nullptr && pack->pmhd == nullptr) {
    Kokkos::abort("radial_profile_benchmark requires hydro or MHD");
  }
  auto *eos = pack->phydro != nullptr ? pack->phydro->peos : pack->pmhd->peos;
  if (eos->eos_data.is_ideal) Kokkos::abort("benchmark requires isothermal EOS");
  const bool varying = pin->GetOrAddBoolean("problem", "varying", true);
  const bool varying_velocity = pin->GetOrAddBoolean("problem", "varying_velocity", false);
  const auto b = pmy_mesh_->mb_indcs;
  const auto size = pack->pmb->mb_size.d_view;
  auto u = pack->phydro != nullptr ? pack->phydro->u0 : pack->pmhd->u0;
  Kokkos::deep_copy(u, 0.0);
  if (pack->pmhd != nullptr) {
    Kokkos::deep_copy(pack->pmhd->b0.x1f, 1.0);
    Kokkos::deep_copy(pack->pmhd->b0.x2f, 0.5);
    Kokkos::deep_copy(pack->pmhd->b0.x3f, 0.25);
  }
  par_for("radial_profile_initial_density", DevExeSpace(), 0, pack->nmb_thispack-1,
      b.ks, b.ke, b.js, b.je, b.is, b.ie, KOKKOS_LAMBDA(int m, int k, int j, int i) {
    const auto &s = size(m);
    const Real x = CellCenterX(i-b.is, b.nx1, s.x1min, s.x1max);
    const Real y = CellCenterX(j-b.js, b.nx2, s.x2min, s.x2max);
    const Real z = CellCenterX(k-b.ks, b.nx3, s.x3min, s.x3max);
    const Real density = varying ?
        1.0 + 0.1*cos(M_PI*x/2) + 0.05*sin(M_PI*y/2) + 0.025*cos(M_PI*z/2) : 1.0;
    u(m, IDN, k, j, i) = density;
    if (varying_velocity) {
      u(m, IM1, k, j, i) = density*(0.3 + 0.2*cos(M_PI*x/2));
      u(m, IM2, k, j, i) = density*(-0.2 + 0.15*sin(M_PI*y/2));
      u(m, IM3, k, j, i) = density*(0.1 + 0.1*cos(M_PI*z/2));
    }
  });
}
