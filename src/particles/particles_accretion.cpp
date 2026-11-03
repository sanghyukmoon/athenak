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

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

#include "athena.hpp"
#include "globals.hpp"
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

  const int scr_level = 0;
  const size_t scr_bytes = ScrArray2D<Real>::shmem_size(4, 27)
                         + ScrArray1D<Real>::shmem_size(5);
  const Real svmax2 = sanity_vmax > 0.0 ? sanity_vmax*sanity_vmax : -1.0;
  auto sflag = sanity_flag_;
  auto srec = sanity_rec_;

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
    ScrArray1D<Real> acc(member.team_scratch(scr_level), 5);
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
      acc(4) = reset_mass > old_mass ? 0.0 : 1.0;
      if (acc(4) == 0.0) return;

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
          if (svmax2 > 0.0) {
            const Real v2 = pr(IPVX,q)*pr(IPVX,q)+pr(IPVY,q)*pr(IPVY,q)
                          +pr(IPVZ,q)*pr(IPVZ,q);
            if (!(v2 <= svmax2) &&
                Kokkos::atomic_compare_exchange(&sflag(0), 0, 2) == 0) {
              srec(0) = static_cast<Real>(pi(PTAG,q));
              srec(1) = mp; srec(2) = acc(0);
              srec(3) = acc(1); srec(4) = acc(2); srec(5) = acc(3);
              srec(6) = vx0; srec(7) = vy0; srec(8) = vz0;
              srec(9) = pr(IPVX,q); srec(10) = pr(IPVY,q); srec(11) = pr(IPVZ,q);
              srec(12) = pr(IPX,q); srec(13) = pr(IPY,q); srec(14) = pr(IPZ,q);
            }
          }
        }
        break;
      }
    });
    member.team_barrier();
    if (acc(4) == 0.0) return;

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

  CheckSinkSanity("accrete", stage);
  return TaskStatus::complete;
}

//----------------------------------------------------------------------------------------
//! \fn void Particles::ExchangeCVReset()
//! \brief MPI transport for the control-volume reset. The accretion kernel stages every
//! reset cell of near-a-rank-boundary sinks into cvemit_ (owner_m, x, y, z, post-reset
//! rho/M1..3, pre-reset rho/M1..3). Here, host-side: for each staged cell find the
//! off-rank neighbour block(s) whose expanded bounds contain it (dest geometry from
//! Mesh::lloc_eachmb + mesh_size, so no device access to off-rank blocks is needed),
//! negotiate counts, Isend/Irecv the payload, and the receiver applies it to every local
//! coincident copy. Records carry the OWNER's refinement level, so the receiver can apply
//! the same three-way level rule as the on-rank scatter (assign at the same level, assign
//! to the covered sub-cells of a finer block, conservative partial restriction into a
//! coarser block). Mirrors the ParticlesBoundaryValues MPI pattern.

void Particles::ExchangeCVReset() {
#if MPI_PARALLEL_ENABLED
  const int my_rank = global_variable::my_rank;
  const int nranks  = global_variable::nranks;
  if (nranks == 1) return;

  // how many cells were staged this step
  auto cnt_h = Kokkos::create_mirror_view(cvemit_cnt_);
  Kokkos::deep_copy(cnt_h, cvemit_cnt_);
  const int nemit = std::min(cnt_h(0), cvemit_max_);
  if (cnt_h(0) > cvemit_max_ && my_rank == 0) {
    std::cout << "### WARNING in Particles::ExchangeCVReset: cvemit overflow ("
              << cnt_h(0) << " > " << cvemit_max_ << "); some cross-rank resets dropped."
              << std::endl;
  }
  auto emit_h = Kokkos::create_mirror_view(cvemit_);
  Kokkos::deep_copy(emit_h, cvemit_);

  Mesh *pm = pmy_pack->pmesh;
  auto &ms = pm->mesh_size;
  const Real Lx1 = ms.x1max - ms.x1min;
  const Real Lx2 = ms.x2max - ms.x2min;
  const Real Lx3 = ms.x3max - ms.x3min;
  auto &indcs = pm->mb_indcs;
  const int is = indcs.is, js = indcs.js, ks = indcs.ks;
  auto &nghbr = pmy_pack->pmb->nghbr;
  auto &mblev = pmy_pack->pmb->mb_lev;
  const int nnghbr = pmy_pack->pmb->nnghbr;
  // wire-record width: x, y, z, 4 post-reset, 4 pre-reset, owner level
  constexpr int NCVREC = 12;

  // build the send plan: a reset cell must reach EVERY rank owning a neighbour block
  // whose interior+ghost array contains it -- the receiver then writes all its local
  // coincident copies (interior AND ghosts), mirroring the on-rank scatter. Writing only
  // interiors is NOT enough: the receiving rank's ghost copies would stay pre-reset and
  // its next flux computation near the boundary would diverge from the single-rank run.
  // Payload: 12 reals (x, y, z, post-reset rho/M1..3, pre-reset rho/M1..3, owner level)
  // per record; one record per (cell, destination rank), deduplicated across that rank's
  // blocks -- the receiver rescans all of its own blocks, so one record per rank suffices
  // even when that rank holds several blocks at different levels.
  std::vector<int>  s_rank;    // destination rank per record
  std::vector<Real> s_real;    // NCVREC reals per record
  for (int e = 0; e < nemit; ++e) {
    const int om = static_cast<int>(emit_h(e, 0));
    if (om < 0 || om >= pmy_pack->nmb_thispack) continue;
    const int olev = mblev.h_view(om);
    const Real xw = emit_h(e, 1), yw = emit_h(e, 2), zw = emit_h(e, 3);
    int sent_ranks[8]; int nsent = 0;   // per-cell rank dedup (few peers in practice)
    for (int n = 0; n < nnghbr; ++n) {
      if (nghbr.h_view(om, n).gid < 0) continue;
      if (nghbr.h_view(om, n).rank == my_rank) continue;   // on-rank done on device
      const int dl = nghbr.h_view(om, n).lev - olev;
      if (dl > 1 || dl < -1) continue;                     // proper nesting violated
      const int drank = nghbr.h_view(om, n).rank;
      bool dup = false;
      for (int q = 0; q < nsent; ++q) if (sent_ranks[q] == drank) { dup = true; break; }
      if (dup) continue;
      const int dg = nghbr.h_view(om, n).gid;
      const auto &ll = pm->lloc_eachmb[dg];
      const int nmbx1 = pm->nmb_rootx1 << (ll.level - pm->root_level);
      const int nmbx2 = pm->nmb_rootx2 << (ll.level - pm->root_level);
      const int nmbx3 = pm->nmb_rootx3 << (ll.level - pm->root_level);
      const Real x1d = (ll.lx1 == 0) ? ms.x1min
                                   : LeftEdgeX(ll.lx1, nmbx1, ms.x1min, ms.x1max);
      const Real x1u = (ll.lx1 == nmbx1-1) ? ms.x1max
                                         : LeftEdgeX(ll.lx1+1, nmbx1, ms.x1min, ms.x1max);
      const Real x2d = (ll.lx2 == 0) ? ms.x2min
                                   : LeftEdgeX(ll.lx2, nmbx2, ms.x2min, ms.x2max);
      const Real x2u = (ll.lx2 == nmbx2-1) ? ms.x2max
                                         : LeftEdgeX(ll.lx2+1, nmbx2, ms.x2min, ms.x2max);
      const Real x3d = (ll.lx3 == 0) ? ms.x3min
                                   : LeftEdgeX(ll.lx3, nmbx3, ms.x3min, ms.x3max);
      const Real x3u = (ll.lx3 == nmbx3-1) ? ms.x3max
                                         : LeftEdgeX(ll.lx3+1, nmbx3, ms.x3min, ms.x3max);
      const Real dx1 = (x1u-x1d)/indcs.nx1;
      const Real dx2 = (x2u-x2d)/indcs.nx2;
      const Real dx3 = (x3u-x3d)/indcs.nx3;
      // wrap the cell centre into the neighbour's frame (periodic domains)
      Real xc = xw, yc = yw, zc = zw, ctr;
      ctr = 0.5*(x1d + x1u);
      if (xc-ctr >  0.5*Lx1) xc -= Lx1;
      if (xc-ctr < -0.5*Lx1) xc += Lx1;
      ctr = 0.5*(x2d + x2u);
      if (yc-ctr >  0.5*Lx2) yc -= Lx2;
      if (yc-ctr < -0.5*Lx2) yc += Lx2;
      ctr = 0.5*(x3d + x3u);
      if (zc-ctr >  0.5*Lx3) zc -= Lx3;
      if (zc-ctr < -0.5*Lx3) zc += Lx3;
      // containment in the EXPANDED (interior + ng ghosts) bounds
      const Real g1 = indcs.ng*dx1, g2 = indcs.ng*dx2, g3 = indcs.ng*dx3;
      if (xc < x1d-g1 || xc >= x1u+g1 || yc < x2d-g2 || yc >= x2u+g2 ||
          zc < x3d-g3 || zc >= x3u+g3) continue;
      if (nsent < 8) sent_ranks[nsent++] = drank;
      s_rank.push_back(drank);
      s_real.push_back(xw); s_real.push_back(yw); s_real.push_back(zw);
      s_real.push_back(emit_h(e,4)); s_real.push_back(emit_h(e,5));
      s_real.push_back(emit_h(e,6)); s_real.push_back(emit_h(e,7));
      s_real.push_back(emit_h(e,8)); s_real.push_back(emit_h(e,9));
      s_real.push_back(emit_h(e,10)); s_real.push_back(emit_h(e,11));
      s_real.push_back(static_cast<Real>(olev));
    }
  }

  // per-destination-rank counts of records to send
  std::vector<int> nsend(nranks, 0);
  for (int r : s_rank) nsend[r]++;
  std::vector<int> nrecv(nranks, 0);
  {
    SinkTimerScope _st(this, ST_ALLTOALL);
    MPI_Alltoall(nsend.data(), 1, MPI_INT, nrecv.data(), 1, MPI_INT, mpi_comm_cvscat_);
  }

  // Pack sends contiguously per destination rank, preserving emission order.
  std::vector<int> soff(nranks, 0), sacc(nranks, 0);
  for (int r = 1; r < nranks; ++r) soff[r] = soff[r-1] + nsend[r-1];
  const int ntot_send = soff[nranks-1] + nsend[nranks-1];
  std::vector<Real> send_r(NCVREC*ntot_send);
  for (size_t rec = 0; rec < s_rank.size(); ++rec) {
    const int r = s_rank[rec];
    const int pos = soff[r] + sacc[r]; sacc[r]++;
    for (int q = 0; q < NCVREC; ++q) send_r[NCVREC*pos+q] = s_real[NCVREC*rec+q];
  }

  // receive layout
  std::vector<int> roff(nranks, 0);
  for (int r = 1; r < nranks; ++r) roff[r] = roff[r-1] + nrecv[r-1];
  const int ntot_recv = roff[nranks-1] + nrecv[nranks-1];
  std::vector<Real> recv_r(NCVREC*ntot_recv);

  // post non-blocking recvs then sends, per peer rank
  std::vector<MPI_Request> reqs;
  for (int r = 0; r < nranks; ++r) {
    if (r == my_rank || nrecv[r] == 0) continue;
    reqs.emplace_back();
    MPI_Irecv(&recv_r[NCVREC*roff[r]], NCVREC*nrecv[r], MPI_ATHENA_REAL, r, 0,
              mpi_comm_cvscat_, &reqs.back());
  }
  for (int r = 0; r < nranks; ++r) {
    if (r == my_rank || nsend[r] == 0) continue;
    reqs.emplace_back();
    MPI_Isend(&send_r[NCVREC*soff[r]], NCVREC*nsend[r], MPI_ATHENA_REAL, r, 0,
              mpi_comm_cvscat_, &reqs.back());
  }
  // records destined for this rank never go through MPI (neighbour rank != my_rank), so
  // no self-copy is needed.
  if (!reqs.empty()) MPI_Waitall(reqs.size(), reqs.data(), MPI_STATUSES_IGNORE);

  // opt-in trace (same switch as the cross-rank particle migration): without it there is
  // no way to tell a run where no control volume crossed a rank boundary from one where
  // the crossing silently did nothing.
  static bool xdbg = (std::getenv("PART_XRANK_DBG") != nullptr);
  if (xdbg && (ntot_send > 0 || ntot_recv > 0)) {
    std::cout << "  [cvscat] rank " << my_rank << " sent " << ntot_send
              << " recvd " << ntot_recv << " control-volume reset cells" << std::endl;
  }

  if (ntot_recv == 0) return;

  // apply on the device with the SAME geometric scatter as the on-rank path: for each
  // received cell, write every local coincident copy -- any local block whose
  // interior+ghost array contains the (wrapped) position, interior or ghost alike.
  // Parallel over blocks, serial over records within a block: preserves record order
  // without write races, since different blocks write disjoint arrays.
  const bool is_mhd = (pmy_pack->pmhd != nullptr);
  auto u0 = is_mhd ? pmy_pack->pmhd->u0 : pmy_pack->phydro->u0;
  auto w0 = is_mhd ? pmy_pack->pmhd->w0 : pmy_pack->phydro->w0;
  const int nmb_ = pmy_pack->nmb_thispack;
  auto &mbsize = pmy_pack->pmb->mb_size;
  const int ncells1 = indcs.nx1 + 2*indcs.ng;
  const int ncells2 = indcs.nx2 + 2*indcs.ng;
  const int ncells3 = indcs.nx3 + 2*indcs.ng;

  DualArray2D<Real> rreal("cvrecv_r", ntot_recv, NCVREC);
  for (int t = 0; t < ntot_recv; ++t) {
    for (int q = 0; q < NCVREC; ++q) rreal.h_view(t,q) = recv_r[NCVREC*t+q];
  }
  rreal.template modify<HostMemSpace>();  rreal.template sync<DevExeSpace>();
  auto rr = rreal.d_view;
  const int is_ = is, js_ = js, ks_ = ks;
  auto &dmblev = pmy_pack->pmb->mb_lev;

  Kokkos::parallel_for("cvreset_apply", Kokkos::RangePolicy<>(DevExeSpace(), 0, nmb_),
  KOKKOS_LAMBDA(const int m) {
    const Real x1d = mbsize.d_view(m).x1min, x1u = mbsize.d_view(m).x1max;
    const Real x2d = mbsize.d_view(m).x2min, x2u = mbsize.d_view(m).x2max;
    const Real x3d = mbsize.d_view(m).x3min, x3u = mbsize.d_view(m).x3max;
    const Real dx1 = mbsize.d_view(m).dx1;
    const Real dx2 = mbsize.d_view(m).dx2;
    const Real dx3 = mbsize.d_view(m).dx3;
    const int mylev = dmblev.d_view(m);
    for (int t = 0; t < ntot_recv; ++t) {
      // wrap the cell centre into this block's frame (periodic domains)
      Real xc = rr(t,0), yc = rr(t,1), zc = rr(t,2), ctr;
      ctr = 0.5*(x1d + x1u);
      if (xc - ctr >  0.5*Lx1) xc -= Lx1;
      if (xc - ctr < -0.5*Lx1) xc += Lx1;
      ctr = 0.5*(x2d + x2u);
      if (yc - ctr >  0.5*Lx2) yc -= Lx2;
      if (yc - ctr < -0.5*Lx2) yc += Lx2;
      ctr = 0.5*(x3d + x3u);
      if (zc - ctr >  0.5*Lx3) zc -= Lx3;
      if (zc - ctr < -0.5*Lx3) zc += Lx3;
      const Real rho = rr(t,3), m1 = rr(t,4), m2 = rr(t,5), m3 = rr(t,6);
      // dlev < 0: this block is COARSER than the control volume the record came from.
      // dlev == 0 / > 0: same level / finer. Same three-way rule as the on-rank scatter
      // (see the file comment). Records are applied serially within a block, so the
      // coarser accumulation needs no atomics and w0 stays consistent with the final u0.
      const int dlev = mylev - static_cast<int>(rr(t,11));
      if (dlev == 0) {
        const int i = static_cast<int>(Kokkos::floor((xc - x1d)/dx1)) + is_;
        const int j = static_cast<int>(Kokkos::floor((yc - x2d)/dx2)) + js_;
        const int k = static_cast<int>(Kokkos::floor((zc - x3d)/dx3)) + ks_;
        if (i < 0 || i >= ncells1 || j < 0 || j >= ncells2 ||
            k < 0 || k >= ncells3) continue;
        u0(m, IDN, k, j, i) = rho;
        u0(m, IM1, k, j, i) = m1;
        u0(m, IM2, k, j, i) = m2;
        u0(m, IM3, k, j, i) = m3;
        if (rho > 0.0) {
          w0(m, IDN, k, j, i) = rho;
          w0(m, IVX, k, j, i) = m1/rho;
          w0(m, IVY, k, j, i) = m2/rho;
          w0(m, IVZ, k, j, i) = m3/rho;
        }
      } else if (dlev > 0) {
        // finer: uniform split over the sub-cells covering the reset cell. The record's
        // cell width is this block's width scaled by the level difference.
        const int r = 1 << dlev;
        const Real rinv = 1.0/static_cast<Real>(r);
        const Real cw1 = dx1*r, cw2 = dx2*r, cw3 = dx3*r;   // reset cell width
        for (int sk = 0; sk < r; ++sk) {
        for (int sj = 0; sj < r; ++sj) {
        for (int si = 0; si < r; ++si) {
          const Real xs = xc + ((si + 0.5)*rinv - 0.5)*cw1;
          const Real ys = yc + ((sj + 0.5)*rinv - 0.5)*cw2;
          const Real zs = zc + ((sk + 0.5)*rinv - 0.5)*cw3;
          const int i = static_cast<int>(Kokkos::floor((xs - x1d)/dx1)) + is_;
          const int j = static_cast<int>(Kokkos::floor((ys - x2d)/dx2)) + js_;
          const int k = static_cast<int>(Kokkos::floor((zs - x3d)/dx3)) + ks_;
          if (i < 0 || i >= ncells1 || j < 0 || j >= ncells2 ||
              k < 0 || k >= ncells3) continue;
          u0(m, IDN, k, j, i) = rho;
          u0(m, IM1, k, j, i) = m1;
          u0(m, IM2, k, j, i) = m2;
          u0(m, IM3, k, j, i) = m3;
          if (rho > 0.0) {
            w0(m, IDN, k, j, i) = rho;
            w0(m, IVX, k, j, i) = m1/rho;
            w0(m, IVY, k, j, i) = m2/rho;
            w0(m, IVZ, k, j, i) = m3/rho;
          }
        }
        }
        }
      } else {
        // coarser: conservative partial restriction of this one reset cell
        const int r = 1 << (-dlev);
        const Real fac = 1.0/static_cast<Real>(r*r*r);
        const int i = static_cast<int>(Kokkos::floor((xc - x1d)/dx1)) + is_;
        const int j = static_cast<int>(Kokkos::floor((yc - x2d)/dx2)) + js_;
        const int k = static_cast<int>(Kokkos::floor((zc - x3d)/dx3)) + ks_;
        if (i < 0 || i >= ncells1 || j < 0 || j >= ncells2 ||
            k < 0 || k >= ncells3) continue;
        u0(m, IDN, k, j, i) += fac*(rho - rr(t,7));
        u0(m, IM1, k, j, i) += fac*(m1  - rr(t,8));
        u0(m, IM2, k, j, i) += fac*(m2  - rr(t,9));
        u0(m, IM3, k, j, i) += fac*(m3  - rr(t,10));
        const Real wdc = u0(m, IDN, k, j, i);
        if (wdc > 0.0) {
          w0(m, IDN, k, j, i) = wdc;
          w0(m, IVX, k, j, i) = u0(m, IM1, k, j, i)/wdc;
          w0(m, IVY, k, j, i) = u0(m, IM2, k, j, i)/wdc;
          w0(m, IVZ, k, j, i) = u0(m, IM3, k, j, i)/wdc;
        }
      }
    }
  });
#endif
}

}  // namespace particles
