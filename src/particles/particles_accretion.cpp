//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file particles_accretion.cpp
//! \brief control-volume sink-particle accretion (Gong & Ostriker 2013; Moon & Ostriker
//! 2025, ApJ 10.3847/1538-4357/add477).
//!
//! Port of the reference implementation in Chang-Goo Kim's Athena++ fork
//! (athena/src/particles/sink_particles.cpp). The 27 cells centred on the sink act as
//! internal ghost zones. Once per step (last RK stage), for each sink:
//!   1(a) integrate conserved gas (rho, momenta) over the control volume -> M^{n+1}
//!   1(b) reset the control volume by outflow extrapolation from the cells just outside
//!        it (6 faces copy, 12 edges avg-of-2, 8 corners avg-of-3, centre avg of the 6
//!        reset faces)
//!   1(c) re-integrate -> M^{n+1}_ctrl
//!   1(d) dM_sink = M^{n+1} - M^{n+1}_ctrl  (exactly the flux that entered the volume)
//!   2    dump conservatively onto the sink: v <- (m v + dM)/(m + dm), m <- m + dm.
//!
//! Every overlapping finest-level MeshBlock evaluates the complete reset from the
//! immutable global survivor list and writes only its own active cells. Surviving
//! control volumes have disjoint write/read stencils under the integer merger rule.
//! Ghosts remain read-only here and are refreshed by the native gas boundary tasks.
//! The particle owner alone receives the full-volume mass and momentum difference.
//!
//! Scope: three-dimensional isothermal resets on a common finest grid, ordinary
//! periodic boundaries and conserved-variable prolongation. Every reset cell must
//! have a finest-level active owner; exterior donors may be prolonged ghost samples.
//! Native EOS floors in the subsequent conversion remain separate gas mass sources.

#include "athena.hpp"
#include "mesh/mesh.hpp"
#include "coordinates/cell_locations.hpp"
#include "driver/driver.hpp"
#include "hydro/hydro.hpp"
#include "mhd/mhd.hpp"
#include "particles.hpp"
#include "sink_convergence.hpp"

namespace particles {

//----------------------------------------------------------------------------------------
//! \fn TaskStatus Particles::AccreteMass
//! \brief accrete gas in the control volume onto each sink particle (last RK stage only).

TaskStatus Particles::AccreteMass(Driver *pdriver, int stage) {
  // operator split: accrete once per step, at the final stage, after the hydro update
  if (stage != pdriver->nexp_stages) return TaskStatus::complete;
  if (particle_type != ParticleType::sink) return TaskStatus::complete;
  if (!(pmy_pack->pmesh->three_d)) return TaskStatus::complete;
  if (pmy_pack->phydro == nullptr && pmy_pack->pmhd == nullptr) {
    return TaskStatus::complete;
  }
  const int nsinks = global_prtcl_tags.extent_int(0);
  if (nsinks == 0) return TaskStatus::complete;
  auto u0 = pmy_pack->pmhd != nullptr ? pmy_pack->pmhd->u0 : pmy_pack->phydro->u0;
  auto gr = global_prtcl_rdata;
  auto gt = global_prtcl_tags;
  auto pr = prtcl_rdata;
  auto pi = prtcl_idata;
  auto mb_lev = pmy_pack->pmb->mb_lev.d_view;
  auto *pm = pmy_pack->pmesh;
  const auto indcs = pm->mb_indcs;
  const auto msz = pm->mesh_size;
  const int max_level = pm->max_level;
  const int shift = max_level - pm->root_level;
  const int nx1_eff = pm->mesh_indcs.nx1 << shift;
  const int nx2_eff = pm->mesh_indcs.nx2 << shift;
  const int nx3_eff = pm->mesh_indcs.nx3 << shift;
  const Real dx1 = (msz.x1max-msz.x1min)/nx1_eff;
  const Real dx2 = (msz.x2max-msz.x2min)/nx2_eff;
  const Real dx3 = (msz.x3max-msz.x3min)/nx3_eff;
  const Real dV = dx1*dx2*dx3;
  const bool periodic = pm->strictly_periodic;
  const int nmb = pmy_pack->nmb_thispack;
  const int gids = pmy_pack->gids;
  const int nlocal = nprtcl_thispack;
  const int convergence = sink_convergence;

  // Logical locations are host data. Copy exact finest-grid block offsets once;
  // coarse blocks are excluded before these offsets are used in the kernel.
  DualArray2D<int> mb_offset("accretion_mb_offset", nmb, 3);
  for (int m=0; m<nmb; ++m) {
    const auto &ll = pm->lloc_eachmb[gids+m];
    mb_offset.h_view(m,0) = ll.lx1*indcs.nx1;
    mb_offset.h_view(m,1) = ll.lx2*indcs.nx2;
    mb_offset.h_view(m,2) = ll.lx3*indcs.nx3;
  }
  mb_offset.template modify<HostMemSpace>();
  mb_offset.template sync<DevExeSpace>();
  auto off = mb_offset.d_view;

  constexpr int iaccept = 4;
  const int scr_level = 0;
  const size_t scr_bytes = ScrArray2D<Real>::shmem_size(4, 27)
                         + ScrArray1D<Real>::shmem_size(iaccept+1);

  par_for_outer("sink_accrete", DevExeSpace(), scr_bytes, scr_level,
                0, nmb-1, 0, nsinks-1,
  KOKKOS_LAMBDA(TeamMember_t member, const int m, const int p) {
    if (mb_lev(m) != max_level) return;
    const int nx[3] = {nx1_eff, nx2_eff, nx3_eff};
    const int nb[3] = {indcs.nx1, indcs.nx2, indcs.nx3};
    const Real lo[3] = {msz.x1min, msz.x2min, msz.x3min};
    const Real hi[3] = {msz.x1max, msz.x2max, msz.x3max};
    const Real xp[3] = {gr(IPX,p), gr(IPY,p), gr(IPZ,p)};
    int center[3];
    for (int axis=0; axis<3; ++axis) {
      Real x = xp[axis];
      if (periodic) {
        const Real length = hi[axis]-lo[axis];
        x -= length*Kokkos::floor((x-lo[axis])/length);
      }
      center[axis] = CellCenterIndex(x,nx[axis],lo[axis],hi[axis]);
      if (periodic) {
        if (center[axis] < off(m,axis)-1) center[axis] += nx[axis];
        if (center[axis] > off(m,axis)+nb[axis]) center[axis] -= nx[axis];
      }
      if (center[axis] < off(m,axis)-1 ||
          center[axis] > off(m,axis)+nb[axis]) return;
    }
    const int ic = center[0]-off(m,0)+indcs.is;
    const int jc = center[1]-off(m,1)+indcs.js;
    const int kc = center[2]-off(m,2)+indcs.ks;
    if (!SinkConverging(u0, m, ic, jc, kc, gr(IPVX,p), gr(IPVY,p),
                        gr(IPVZ,p), convergence)) {
      if (gr(IPM,p) == 0.0) {
        Kokkos::abort("Massless sink failed convergence in AccreteMass.");
      }
      return;
    }

    ScrArray2D<Real> scr(member.team_scratch(scr_level), 4, 27);
    ScrArray1D<Real> acc(member.team_scratch(scr_level), iaccept+1);
    par_for_inner(member, 0, 26, [&](const int c) {
      const int di = c%3-1, dj = (c/3)%3-1, dk = c/9-1;
      const int nnz = (di != 0)+(dj != 0)+(dk != 0);
      if (nnz == 0) return;
      for (int v=0; v<4; ++v) {
        Real val = 0.0;
        if (di != 0) val += u0(m,v,kc+dk,jc+dj,ic+2*di);
        if (dj != 0) val += u0(m,v,kc+dk,jc+2*dj,ic+di);
        if (dk != 0) val += u0(m,v,kc+2*dk,jc+dj,ic+di);
        scr(v,c) = val/nnz;
      }
    });
    member.team_barrier();

    Kokkos::single(Kokkos::PerTeam(member), [&]() {
      Real old_mass = 0.0, reset_mass = 0.0;
      for (int v=0; v<4; ++v) {
        scr(v,13) = (scr(v,12)+scr(v,14)+scr(v,10)+scr(v,16)
                    +scr(v,4)+scr(v,22))/6.0;
        Real s1 = 0.0, s2 = 0.0;
        // Fixed cell order makes replicated decisions independent of team size.
        for (int c=0; c<27; ++c) {
          const int di = c%3-1, dj = (c/3)%3-1, dk = c/9-1;
          s1 += u0(m,v,kc+dk,jc+dj,ic+di);
          s2 += scr(v,c);
        }
        acc(v) = (s1-s2)*dV;
        if (v == 0) {old_mass = s1; reset_mass = s2;}
      }
      if (gr(IPM,p) == 0.0 && !(acc(0) > 0.0)) {
        Kokkos::abort("Massless sink has nonpositive mass gain in AccreteMass.");
      }
      acc(iaccept) = reset_mass > old_mass ? 0.0 : 1.0;
      if (acc(iaccept) == 0.0) return;

      // Migration may reorder the local arrays. Only the matching owner team
      // credits the complete CV difference; other teams only reset their gas.
      for (int q=0; q<nlocal; ++q) {
        if (pi(PTAG,q) != gt(p) || pi(PGID,q) != gids+m) continue;
        const Real mp = gr(IPM,p), mnew = mp+acc(0);
        if (mnew > 0.0) {
          const Real minv = 1.0/mnew;
          const Real vx0 = gr(IPVX,p), vy0 = gr(IPVY,p), vz0 = gr(IPVZ,p);
          pr(IPVX,q) = (mp*vx0+acc(1))*minv;
          pr(IPVY,q) = (mp*vy0+acc(2))*minv;
          pr(IPVZ,q) = (mp*vz0+acc(3))*minv;
          pr(IPM,q) = mnew;
        }
        break;
      }
    });
    member.team_barrier();
    if (acc(iaccept) == 0.0) return;

    par_for_inner(member, 0, 26, [&](const int c) {
      const int delta[3] = {c%3-1, (c/3)%3-1, c/9-1};
      int local[3];
      for (int axis=0; axis<3; ++axis) {
        int cell = center[axis]+delta[axis];
        if (periodic) cell = (cell+nx[axis])%nx[axis];
        local[axis] = cell-off(m,axis);
        if (local[axis] < 0 || local[axis] >= nb[axis]) return;
      }
      // Read images can be ghosts even when the wrapped write belongs to the
      // opposite active edge of a block spanning an entire periodic direction.
      for (int v=0; v<4; ++v) {
        u0(m,v,local[2]+indcs.ks,local[1]+indcs.js,local[0]+indcs.is) = scr(v,c);
      }
    });
  });

  return TaskStatus::complete;
}

}  // namespace particles
