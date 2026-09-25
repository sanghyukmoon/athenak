//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file gmtf.cpp
//! \brief Problem generator for gravo-magneto-turbulent fragmentation

// C headers

// C++ headers
#include <algorithm>
#include <cstdint>
#include <cmath>
#include <random>
#include <vector>
#include <iostream> // cout

#include "athena.hpp"
#include "parameter_input.hpp"
#include "coordinates/cell_locations.hpp"
#include "mesh/mesh.hpp"
#include "eos/eos.hpp"
#include "globals.hpp"
#include "gravity/gravity.hpp"
#include "hydro/hydro.hpp"
#include "mhd/mhd.hpp"
#include "pgen.hpp"
#include "utils/radial_profile.hpp"

#if MPI_PARALLEL_ENABLED
#include <mpi.h>
#endif

namespace {
DvceArray1D<RadialProfileCenter> FindPotentialMinima(Mesh *pm) {
  // GMTF starts at uniform density; roundoff in the initial potential is not structure.
  if (pm->ncycle == 0) {
    return DvceArray1D<RadialProfileCenter>("potential_minima", 0);
  }

  // Capture variables for kernel
  auto *pack = pm->pmb_pack;
  const auto phi = pack->pgrav->phi;
  const auto &indcs = pm->mb_indcs;
  const int &nx1 = indcs.nx1, &nx2 = indcs.nx2, &nx3 = indcs.nx3;
  const int &is = indcs.is, &js = indcs.js, &ks = indcs.ks;
  const int &ie = indcs.ie, &je = indcs.je, &ke = indcs.ke;
  const int cells_per_block = pm->NumberOfMeshBlockCells();
  const int ncells_thispack = cells_per_block*pack->nmb_thispack;
  DvceArray1D<std::uint8_t> is_minima("minimum_flags", ncells_thispack);
  int ncenters_thisrank = 0;
  // Flag minima and count them simultaneously.
  Kokkos::parallel_reduce("find_potential_minima", Kokkos::RangePolicy<DevExeSpace>(0, ncells_thispack),
      KOKKOS_LAMBDA(const int cell, int &count) {
    const int m = cell/cells_per_block;
    const auto active = cell%cells_per_block;
    const int i = is + active%nx1;
    const int j = js + (active/nx1)%nx2;
    const int k = ks + active/(nx1*nx2);
    const Real phi_c = phi(m, 0, k, j, i);
    for (int dk = -1; dk <= 1; ++dk) {
      for (int dj = -1; dj <= 1; ++dj) {
        for (int di = -1; di <= 1; ++di) {
          if (di == 0 && dj == 0 && dk == 0) continue;
          if (!(phi_c < phi(m, 0, k+dk, j+dj, i+di))) return;
        }
      }
    }
    is_minima(cell) = 1;
    ++count;
  }, ncenters_thisrank);

  // Capture variables
  DvceArray1D<LogicalLocation> lloc("minimum_block_locations", pack->nmb_thispack);
  auto lloc_host = Kokkos::create_mirror_view(lloc);
  for (int m = 0; m < pack->nmb_thispack; ++m) {
    lloc_host(m) = pm->lloc_eachmb[pack->gids + m];
  }
  Kokkos::deep_copy(lloc, lloc_host);
  const std::uint64_t gnx1 = pm->mesh_indcs.nx1;
  const std::uint64_t gnx2 = pm->mesh_indcs.nx2;
  const auto w0 = pack->phydro != nullptr ? pack->phydro->w0 : pack->pmhd->w0;
  const auto meshblock_sizes = pack->pmb->mb_size.d_view;
  DvceArray1D<RadialProfileCenter> local_centers("local_potential_minima", ncenters_thisrank);
  Kokkos::View<int, DevMemSpace> center_count("minimum_center_count");
  par_for("compact_potential_minima", DevExeSpace(),
      0, pack->nmb_thispack-1, ks, ke, js, je, is, ie,
      KOKKOS_LAMBDA(int m, int k, int j, int i) {
    const int cell = m*cells_per_block + ((k-ks)*nx2+(j-js))*nx1+(i-is);
    if (is_minima(cell) == 0) return;
    const auto &block_size = meshblock_sizes(m);
    const std::uint64_t gi = static_cast<std::uint64_t>(lloc(m).lx1)*nx1+(i-is);
    const std::uint64_t gj = static_cast<std::uint64_t>(lloc(m).lx2)*nx2+(j-js);
    const std::uint64_t gk = static_cast<std::uint64_t>(lloc(m).lx3)*nx3+(k-ks);
    const int idx = Kokkos::atomic_fetch_add(&center_count(), 1);
    local_centers(idx) = {
      gi + gnx1*(gj + gnx2*gk),
      CellCenterX(i-is, nx1, block_size.x1min, block_size.x1max),
      CellCenterX(j-js, nx2, block_size.x2min, block_size.x2max),
      CellCenterX(k-ks, nx3, block_size.x3min, block_size.x3max),
      w0(m, IVX, k, j, i),
      w0(m, IVY, k, j, i),
      w0(m, IVZ, k, j, i)};
  });

  const auto local_centers_host = Kokkos::create_mirror_view_and_copy(
      Kokkos::HostSpace(), local_centers);
  int ncenters = 0;
#if MPI_PARALLEL_ENABLED
  std::vector<int> ncenters_eachrank(global_variable::nranks);
  std::vector<int> offsets(global_variable::nranks);
  MPI_Allgather(&ncenters_thisrank, 1, MPI_INT, ncenters_eachrank.data(), 1, MPI_INT, MPI_COMM_WORLD);
  for (int rank = 0; rank < global_variable::nranks; ++rank) {
    offsets[rank] = ncenters;
    ncenters += ncenters_eachrank[rank];
  }
#else
  ncenters = ncenters_thisrank;
#endif
  DvceArray1D<RadialProfileCenter> centers("potential_minima", ncenters);
  auto centers_host = Kokkos::create_mirror_view(centers);
#if MPI_PARALLEL_ENABLED
  MPI_Datatype center_type;
  MPI_Type_contiguous(sizeof(RadialProfileCenter), MPI_BYTE, &center_type);
  MPI_Type_commit(&center_type);
  MPI_Allgatherv(local_centers_host.data(), ncenters_thisrank, center_type,
                 centers_host.data(), ncenters_eachrank.data(), offsets.data(), center_type,
                 MPI_COMM_WORLD);
  MPI_Type_free(&center_type);
#else
  Kokkos::deep_copy(centers_host, local_centers_host);
#endif
  if (ncenters > 0) {
    std::sort(centers_host.data(), centers_host.data()+ncenters,
              [](const RadialProfileCenter &a, const RadialProfileCenter &b) {
                return a.id < b.id;
              });
  }
  Kokkos::deep_copy(centers, centers_host);
  return centers;
}
}  // namespace

//========================================================================================
//! \fn void MeshBlock::ProblemGenerator(ParameterInput *pin)
//  \brief
//========================================================================================
void ProblemGenerator::UserProblem(ParameterInput *pin, const bool restart) {
  rprof_center_func = FindPotentialMinima;
  if (restart) return;

  MeshBlockPack *pmbp = pmy_mesh_->pmb_pack;
  DvceArray5D<Real> u0;

  if (pmbp->phydro != nullptr) {
    // HYDRO -----------------------------------
    if (pmbp->phydro->peos->eos_data.is_ideal) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                << std::endl
                << "this problem requires isothermal eos" << std::endl;
      std::exit(EXIT_FAILURE);
    }
    if (pmbp->phydro->peos->eos_data.iso_cs != 1.0) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                << std::endl
                << "this problem takes sound speed as unit velocity."
                << " set iso_sound_speed = 1.0 in the input file" << std::endl;
      std::exit(EXIT_FAILURE);
    }
    u0 = pmbp->phydro->u0;
  } else if (pmbp->pmhd != nullptr) {
    // MHD ------------------------------------
    if (pmbp->pmhd->peos->eos_data.is_ideal) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                << std::endl
                << "this problem requires isothermal eos" << std::endl;
      std::exit(EXIT_FAILURE);
    }
    if (pmbp->pmhd->peos->eos_data.iso_cs != 1.0) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                << std::endl
                << "this problem takes sound speed as unit velocity."
                << " set iso_sound_speed = 1.0 in the input file" << std::endl;
      std::exit(EXIT_FAILURE);
    }
    u0 = pmbp->pmhd->u0;
  } else {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "this problem can only be run with Hydro and/or MHD, but no "
              << "<hydro> or <mhd> block in input file" << std::endl;
    std::exit(EXIT_FAILURE);
  }

  auto &indcs = pmy_mesh_->mb_indcs;

  // capture variables for kernel
  int &is = indcs.is; int &ie = indcs.ie;
  int &js = indcs.js; int &je = indcs.je;
  int &ks = indcs.ks; int &ke = indcs.ke;


  // Initialize Hydro variables -------------------------------
  if (pmbp->phydro != nullptr) {
    // Set initial conditions
    par_for("pgen_turb", DevExeSpace(),0,(pmbp->nmb_thispack-1),ks,ke,js,je,is,ie,
    KOKKOS_LAMBDA(int m, int k, int j, int i) {
      u0(m,IDN,k,j,i) = 1.0;
      u0(m,IM1,k,j,i) = 0.0;
      u0(m,IM2,k,j,i) = 0.0;
      u0(m,IM3,k,j,i) = 0.0;
      // TODO(SMOON) SCALARS?
    });
  }

  // Initialize MHD variables ---------------------------------
  if (pmbp->pmhd != nullptr) {
    auto &b0 = pmbp->pmhd->b0;
    Real mu_phi = pin->GetReal("problem", "mu_phi");
    Real lbox = pmy_mesh_->mesh_size.x1max - pmy_mesh_->mesh_size.x1min;
    Real B0 = M_PI*lbox / mu_phi;

    // Set initial conditions
    par_for("pgen_turb", DevExeSpace(),0,(pmbp->nmb_thispack-1),ks,ke,js,je,is,ie,
    KOKKOS_LAMBDA(int m, int k, int j, int i) {
      u0(m,IDN,k,j,i) = 1.0;
      u0(m,IM1,k,j,i) = 0.0;
      u0(m,IM2,k,j,i) = 0.0;
      u0(m,IM3,k,j,i) = 0.0;
      b0.x1f(m,k,j,i) = 0.0;
      b0.x2f(m,k,j,i) = 0.0;
      b0.x3f(m,k,j,i) = B0;
      if (i==ie) {b0.x1f(m,k,j,i+1) = 0.0;}
      if (j==je) {b0.x2f(m,k,j+1,i) = 0.0;}
      if (k==ke) {b0.x3f(m,k+1,j,i) = B0;}
    });
  }

  // Add turbulent velocity perturbations ---------------------------------
  Real mach = pin->GetReal("problem", "Mach");
  int rseed = pin->GetInteger("problem", "rseed");
  int nlow = pin->GetInteger("problem", "nlow");
  int nhigh = pin->GetInteger("problem", "nhigh");
  Real expo = pin->GetReal("problem", "expo");

  if (mach <= 0.0) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl
              << "Mach number must be positive" << std::endl;
    std::exit(EXIT_FAILURE);
  }
  if (nhigh <= nlow) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl
              << "problem/nhigh must be greater than problem/nlow" << std::endl;
    std::exit(EXIT_FAILURE);
  }
  // TODO(SMOON) Check if cs=1.0

  Real x1size = pmbp->pmesh->mesh_size.x1max - pmbp->pmesh->mesh_size.x1min;
  Real x2size = pmbp->pmesh->mesh_size.x2max - pmbp->pmesh->mesh_size.x2min;
  Real x3size = pmbp->pmesh->mesh_size.x3max - pmbp->pmesh->mesh_size.x3min;

  if (!(x1size == x2size && x1size == x3size)) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl
              << "this problem assumes cubic domain" << std::endl;
    std::exit(EXIT_FAILURE);
  }

  Real dk = 2.0*M_PI/x1size;

  std::mt19937_64 gen(rseed);
  std::normal_distribution<Real> gauss(0.0, 1.0);

  std::vector<Real> host_kx, host_ky, host_kz;
  std::vector<Real> host_ax, host_ay, host_az;
  std::vector<Real> host_bx, host_by, host_bz;

  // Here, we utilize Hermitian symmetry to only keep non-redundant modes.
  // The points symmetric with respect to the origin are complex conjugates,
  // so we only need to keep one of them. Also, we discard DC component, i.e.,
  // kx=ky=kz=0, to avoid adding a net bulk velocity.
  for (int nkx = 0; nkx <= nhigh; ++nkx) {
    for (int nky = -nhigh; nky <= nhigh; ++nky) {
      for (int nkz = -nhigh; nkz <= nhigh; ++nkz) {
        if (nkx == 0) {
          // We are on the kx=0 plane
          if (nky < 0) continue;
          if (nky == 0 && nkz <= 0) continue;
        }

        Real kx = dk*static_cast<Real>(nkx);
        Real ky = dk*static_cast<Real>(nky);
        Real kz = dk*static_cast<Real>(nkz);
        Real kmag = std::sqrt(SQR(kx) + SQR(ky) + SQR(kz));
        int nmag2 = SQR(nkx) + SQR(nky) + SQR(nkz);
        if ( (SQR(nlow) <= nmag2) && (nmag2 <= SQR(nhigh)) ) {
          host_kx.push_back(kx);
          host_ky.push_back(ky);
          host_kz.push_back(kz);
          // Keep the unprojected isotropic field, equivalent to f_shear = 0.5 after
          // renormalization to the target Mach number.
          Real pcoeff = 1.0/std::pow(kmag, (expo + 2.0)/2.0);
          host_ax.push_back(pcoeff*gauss(gen));
          host_ay.push_back(pcoeff*gauss(gen));
          host_az.push_back(pcoeff*gauss(gen));
          host_bx.push_back(pcoeff*gauss(gen));
          host_by.push_back(pcoeff*gauss(gen));
          host_bz.push_back(pcoeff*gauss(gen));
        }
      }
    }
  }
  int nmode = static_cast<int>(host_kx.size());
  DualArray1D<Real> kx("kx", nmode);
  DualArray1D<Real> ky("ky", nmode);
  DualArray1D<Real> kz("kz", nmode);
  DualArray1D<Real> vxk_re("vxk_re", nmode);
  DualArray1D<Real> vyk_re("vyk_re", nmode);
  DualArray1D<Real> vzk_re("vzk_re", nmode);
  DualArray1D<Real> vxk_im("vxk_im", nmode);
  DualArray1D<Real> vyk_im("vyk_im", nmode);
  DualArray1D<Real> vzk_im("vzk_im", nmode);
  for (int n = 0; n < nmode; ++n) {
    kx.h_view(n) = host_kx[n];
    ky.h_view(n) = host_ky[n];
    kz.h_view(n) = host_kz[n];
    vxk_re.h_view(n) = host_ax[n];
    vyk_re.h_view(n) = host_ay[n];
    vzk_re.h_view(n) = host_az[n];
    vxk_im.h_view(n) = host_bx[n];
    vyk_im.h_view(n) = host_by[n];
    vzk_im.h_view(n) = host_bz[n];
  }

  auto sync_to_device = [](auto &arr) {
    arr.template modify<HostMemSpace>();
    arr.template sync<DevExeSpace>();
  };
  sync_to_device(kx);
  sync_to_device(ky);
  sync_to_device(kz);
  sync_to_device(vxk_re);
  sync_to_device(vyk_re);
  sync_to_device(vzk_re);
  sync_to_device(vxk_im);
  sync_to_device(vyk_im);
  sync_to_device(vzk_im);

  int nmb = pmbp->nmb_thispack;
  int nx1 = indcs.nx1;
  int nx2 = indcs.nx2;
  int nx3 = indcs.nx3;
  int ng = indcs.ng;
  auto &size = pmbp->pmb->mb_size;
  DvceArray5D<Real> dv("vel_perturb", nmb, 3, nx3 + 2*ng, nx2 + 2*ng, nx1 + 2*ng);

  par_for("gmtf_build_turb", DevExeSpace(), 0, nmb - 1, ks, ke, js, je, is, ie,
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    Real x1v = CellCenterX(i - is, nx1, size.d_view(m).x1min, size.d_view(m).x1max);
    Real x2v = CellCenterX(j - js, nx2, size.d_view(m).x2min, size.d_view(m).x2max);
    Real x3v = CellCenterX(k - ks, nx3, size.d_view(m).x3min, size.d_view(m).x3max);

    // Sum over all modes to construct the velocity perturbation in real space
    Real dv_x = 0.0;
    Real dv_y = 0.0;
    Real dv_z = 0.0;
    for (int n = 0; n < nmode; ++n) {
      Real kdotx = kx.d_view(n)*x1v + ky.d_view(n)*x2v + kz.d_view(n)*x3v;
      dv_x += vxk_re.d_view(n)*std::cos(kdotx) - vxk_im.d_view(n)*std::sin(kdotx);
      dv_y += vyk_re.d_view(n)*std::cos(kdotx) - vyk_im.d_view(n)*std::sin(kdotx);
      dv_z += vzk_re.d_view(n)*std::cos(kdotx) - vzk_im.d_view(n)*std::sin(kdotx);
    }
    dv(m,0,k,j,i) = dv_x;
    dv(m,1,k,j,i) = dv_y;
    dv(m,2,k,j,i) = dv_z;
  });

  const int nmkji = nmb*nx3*nx2*nx1;
  const int nkji = nx3*nx2*nx1;
  const int nji = nx2*nx1;
  Real v2_sum = 0.0;
  Kokkos::parallel_reduce("gmtf_vrms",
  Kokkos::RangePolicy<>(DevExeSpace(), 0, nmkji),
  KOKKOS_LAMBDA(const int &idx, Real &v2_sum_local) {
    int m = idx/nkji;
    int k = (idx - m*nkji)/nji;
    int j = (idx - m*nkji - k*nji)/nx1;
    int i = (idx - m*nkji - k*nji - j*nx1) + is;
    k += ks;
    j += js;
    v2_sum_local += SQR(dv(m,0,k,j,i)) + SQR(dv(m,1,k,j,i)) + SQR(dv(m,2,k,j,i));
  }, Kokkos::Sum<Real>(v2_sum));
#if MPI_PARALLEL_ENABLED
  MPI_Allreduce(MPI_IN_PLACE, &v2_sum, 1, MPI_ATHENA_REAL, MPI_SUM, MPI_COMM_WORLD);
#endif

  // Rescale the velocity perturbations to achieve the target Mach number
  long long gnx1 = static_cast<long long>(pmbp->pmesh->mesh_indcs.nx1);
  long long gnx2 = static_cast<long long>(pmbp->pmesh->mesh_indcs.nx2);
  long long gnx3 = static_cast<long long>(pmbp->pmesh->mesh_indcs.nx3);
  Real vrms = std::sqrt(v2_sum / static_cast<Real>(gnx1*gnx2*gnx3));
  par_for("gmtf_init_turb", DevExeSpace(), 0, nmb - 1, ks, ke, js, je, is, ie,
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    u0(m,IM1,k,j,i) += mach/vrms*dv(m,0,k,j,i);
    u0(m,IM2,k,j,i) += mach/vrms*dv(m,1,k,j,i);
    u0(m,IM3,k,j,i) += mach/vrms*dv(m,2,k,j,i);
  });
}
