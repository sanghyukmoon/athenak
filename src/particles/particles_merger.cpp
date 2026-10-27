//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file particles_merger.cpp
//! \brief sink-particle merging on overlapping control volumes.
//!
//! Cell-centred CV overlap and iterative pair reduction.
//! New candidates carry zero physical mass and separate temporary merger weights.
//! Every rank resolves the same tag-sorted global list, then updates its local arrays.
//! The task chain migrates moved survivors before accretion.
//! Shared-face contact merges; edge/corner-only contact does not.

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <map>
#include <set>
#include <utility>
#include <vector>

#include "athena.hpp"
#include "globals.hpp"
#include "mesh/mesh.hpp"
#include "driver/driver.hpp"
#include "particles.hpp"
#include "sink_merge.hpp"
#include "coordinates/cell_locations.hpp"

#if MPI_PARALLEL_ENABLED
#include <mpi.h>
#endif

namespace particles {

namespace {
constexpr int REC = 11;   // per-sink record: x,y,z,vx,vy,vz,m,dx1,dx2,dx3,tag
}  // namespace

//----------------------------------------------------------------------------------------
//! \fn TaskStatus Particles::MergeSinks
//! \brief merge sinks whose control volumes overlap (last RK stage only).

TaskStatus Particles::MergeSinks(Driver *pdriver, int stage) {
  if (stage != pdriver->nexp_stages) return TaskStatus::complete;
  if (particle_type != ParticleType::sink) return TaskStatus::complete;
  SinkTimerScope _st(this, ST_MERGE);

  const int nranks = global_variable::nranks;
  const int my_rank = global_variable::my_rank;

  auto &mbsize = pmy_pack->pmb->mb_size;
  auto msz_h = mbsize.h_view;
  const int gids = pmy_pack->gids;
  const int nmb = pmy_pack->nmb_thispack;
  const bool per = pmy_pack->pmesh->strictly_periodic;
  auto &msz = pmy_pack->pmesh->mesh_size;
  const Real L[3] = {msz.x1max-msz.x1min, msz.x2max-msz.x2min, msz.x3max-msz.x3min};
  const Real dlo[3] = {msz.x1min, msz.x2min, msz.x3min};

  auto refresh_return = [&]() {
    RefreshMeshParticleCounts();
    CheckSinkSanity("merge", stage);
    return TaskStatus::complete;
  };

  // ---- host mirror of this rank's particles + local records ----
  const int nloc = nprtcl_thispack;
  auto hr = Kokkos::create_mirror(prtcl_rdata);
  auto hi = Kokkos::create_mirror(prtcl_idata);
  Kokkos::deep_copy(hr, prtcl_rdata);
  Kokkos::deep_copy(hi, prtcl_idata);
  std::vector<Real> loc(static_cast<size_t>(nloc)*REC);
  for (int p = 0; p < nloc; ++p) {
    int m = hi(PGID, p) - gids;
    Real dx1 = msz_h(0).dx1, dx2 = msz_h(0).dx2, dx3 = msz_h(0).dx3;
    if (m >= 0 && m < nmb) { dx1 = msz_h(m).dx1; dx2 = msz_h(m).dx2; dx3 = msz_h(m).dx3; }
    Real *rec = &loc[static_cast<size_t>(p)*REC];
    rec[0]=hr(IPX,p);  rec[1]=hr(IPY,p);  rec[2]=hr(IPZ,p);
    rec[3]=hr(IPVX,p); rec[4]=hr(IPVY,p); rec[5]=hr(IPVZ,p);
    rec[6]=hr(IPM,p);  rec[7]=dx1; rec[8]=dx2; rec[9]=dx3;
    rec[10]=static_cast<Real>(hi(PTAG,p));
  }

  // ---- gather every rank's sinks (cheap; sinks are few) ----
  std::vector<Real> G;
  int Ntot = nloc;
#if MPI_PARALLEL_ENABLED
  if (nranks > 1) {
    std::vector<int> cnt(nranks), cntR(nranks), dspR(nranks);
    MPI_Allgather(&nloc, 1, MPI_INT, cnt.data(), 1, MPI_INT, MPI_COMM_WORLD);
    int off = 0; Ntot = 0;
    for (int r = 0; r < nranks; ++r) {
      cntR[r] = cnt[r]*REC; dspR[r] = off; off += cntR[r]; Ntot += cnt[r];
    }
    G.resize(static_cast<size_t>(Ntot)*REC);
    MPI_Allgatherv(loc.data(), nloc*REC, MPI_ATHENA_REAL, G.data(),
                   cntR.data(), dspR.data(), MPI_ATHENA_REAL, MPI_COMM_WORLD);
  } else {
    G = std::move(loc);
  }
#else
  G = std::move(loc);
#endif
  if (Ntot < 2) return refresh_return();

  // ---- PTAGs must be globally unique --------------------------------------------------
  // Everything below keys survivors and absorbed sinks by tag, so two distinct sinks
  // sharing a tag are silently taken for one object: they get "merged" however far apart
  // they are, their masses are summed, and every rank writes the sum onto its own copy --
  // a mass that doubles every cycle. Creation numbers tags from a global base to prevent
  // this, but check it here too, where the assumption is actually relied on, and where a
  // violation can be reported against the offending tag instead of being inferred later
  // from a runaway mass. Cheap: a sort over the (few) sinks, no communication, and G is
  // identical on every rank so the verdict is too.
  {
    std::vector<int> tags(Ntot);
    for (int i = 0; i < Ntot; ++i) {
      tags[i] = static_cast<int>(G[static_cast<size_t>(i)*REC + 10]);
    }
    std::vector<int> sorted_tags = tags;
    std::sort(sorted_tags.begin(), sorted_tags.end());
    auto dup = std::adjacent_find(sorted_tags.begin(), sorted_tags.end());
    if (dup != sorted_tags.end()) {
      if (my_rank == 0) {
        std::cout << "### FATAL ERROR in Particles::MergeSinks: PTAG " << *dup
                  << " is used by more than one sink (of " << Ntot << " globally). Sink "
                  << "tags must be unique across ALL ranks -- merging keys on them, so "
                  << "duplicates fuse unrelated sinks and their mass diverges."
                  << std::endl;
      }
      std::exit(EXIT_FAILURE);
    }
  }

  struct Surv { Real x, y, z, vx, vy, vz, M; };
  std::map<int, Surv> survmap;
  std::set<int> absorbed;
  std::vector<SinkMergeRecord<Real>> sinks;
  for (int i=0; i<Ntot; ++i) {
    const Real *r=&G[static_cast<size_t>(i)*REC];
    sinks.push_back({{r[0],r[1],r[2]}, {r[3],r[4],r[5]}, {r[7],r[8],r[9]},
                     r[6], 1.0, static_cast<int>(r[10])});
  }
  auto locate = [&](SinkMergeRecord<Real> &s) {
    auto *pm=pmy_pack->pmesh;
    for (int g=0; g<pm->nmb_total; ++g) {
      const auto &ll=pm->lloc_eachmb[g];
      const int shift=ll.level-pm->root_level;
      const int nb[3]={pm->nmb_rootx1<<shift,pm->nmb_rootx2<<shift,
                       pm->nmb_rootx3<<shift};
      const auto ix=std::array<std::int64_t,3>{ll.lx1,ll.lx2,ll.lx3};
      bool inside=true;
      for (int c=0; c<3; ++c) {
        const Real lo=LeftEdgeX(ix[c],nb[c],dlo[c],dlo[c]+L[c]);
        const Real hi=LeftEdgeX(ix[c]+1,nb[c],dlo[c],dlo[c]+L[c]);
        inside=inside && s.x[c]>=lo && s.x[c]<hi;
      }
      if (inside) {
        s.dx={L[0]/(nb[0]*pm->mb_indcs.nx1), L[1]/(nb[1]*pm->mb_indcs.nx2),
              L[2]/(nb[2]*pm->mb_indcs.nx3)};
        return;
      }
    }
    std::cerr << "Merged sink has no owning MeshBlock" << std::endl;
    std::exit(EXIT_FAILURE);
  };
  ResolveSinkMergers(sinks, std::array<Real,3>{dlo[0],dlo[1],dlo[2]},
                    std::array<Real,3>{L[0],L[1],L[2]}, per, locate);
  for (const auto &s : sinks) {
    if (!s.alive) {
      absorbed.insert(s.tag);
    } else if (s.changed) {
      survmap[s.tag]={s.x[0],s.x[1],s.x[2],s.v[0],s.v[1],s.v[2],s.mass};
    }
  }
  if (absorbed.empty()) return refresh_return();
  if (my_rank==0) std::printf("MergeSinks: iterative removed %zu sinks cycle=%d\n",
                              absorbed.size(),pmy_pack->pmesh->ncycle);

  // ---- pre-merge totals, measured from the ACTUAL particle arrays ----------------
  // NOT from the gathered list G. Summing G before and after the group reduction only
  // proves the reduction arithmetic is self-consistent: if G itself is wrong -- e.g. it
  // lists one physical sink twice because two ranks assigned the same PTAG -- then both
  // sums double-count it identically and the check reports dM=0 while the sink's mass
  // doubles every cycle. That is exactly what happened before tags were made globally
  // unique: 230 cycles of "dM=0.000e+00" while the mass ran to 1e72. So measure the real
  // state instead, and compare it after the arrays have been mutated and compacted.
  // Safe under MPI: every rank derives survmap from the same G, so all ranks take this
  // branch together and the reductions below cannot deadlock.
  Real m_pre_loc = 0.0, p_pre_loc[3] = {0.0, 0.0, 0.0};
  const bool check_cons = !survmap.empty();
  if (check_cons) {
    for (int p = 0; p < nloc; ++p) {
      const Real m = hr(IPM, p);
      m_pre_loc += m;
      p_pre_loc[0] += m*hr(IPVX, p);
      p_pre_loc[1] += m*hr(IPVY, p);
      p_pre_loc[2] += m*hr(IPVZ, p);
    }
  }

  // ---- mutate this rank's local particles, keyed by tag ----
  std::vector<char> dead(nloc, 0);
  for (int p = 0; p < nloc; ++p) {
    const int tag = hi(PTAG, p);
    if (absorbed.count(tag)) { dead[p] = 1; continue; }
    auto it = survmap.find(tag);
    if (it == survmap.end()) continue;          // not merged -> untouched
    const Surv &s = it->second;
    hr(IPX,p)=s.x;   hr(IPY,p)=s.y;   hr(IPZ,p)=s.z;
    hr(IPVX,p)=s.vx; hr(IPVY,p)=s.vy; hr(IPVZ,p)=s.vz;
    hr(IPM,p)=s.M;
    // treat the merge as NOT a cell crossing so the following AccreteMass does not
    // re-accrete a spurious old control volume
    hr(IPX0,p)=s.x;  hr(IPY0,p)=s.y;  hr(IPZ0,p)=s.z;
    // re-derive the owning block from the COM (lower-inclusive). If the COM left this
    // rank, keep the old PGID; the migration chain relocates the sink next cycle.
    for (int mm = 0; mm < nmb; ++mm) {
      if (s.x >= msz_h(mm).x1min && s.x < msz_h(mm).x1max &&
          s.y >= msz_h(mm).x2min && s.y < msz_h(mm).x2max &&
          s.z >= msz_h(mm).x3min && s.z < msz_h(mm).x3max) {
        hi(PGID,p) = gids + mm; break;
      }
    }
  }

  // ---- compact out absorbed slots (order-preserving) ----
  std::vector<int> keep;
  keep.reserve(nloc);
  for (int p = 0; p < nloc; ++p) {
    if (!dead[p]) keep.push_back(p);
  }
  const int nnew = static_cast<int>(keep.size());
  Kokkos::resize(prtcl_rdata, nrdata, nnew);
  Kokkos::resize(prtcl_idata, nidata, nnew);
  auto hr2 = Kokkos::create_mirror(prtcl_rdata);
  auto hi2 = Kokkos::create_mirror(prtcl_idata);
  for (int n = 0; n < nnew; ++n) {
    const int src = keep[n];
    for (int v = 0; v < nrdata; ++v) hr2(v, n) = hr(v, src);
    for (int v = 0; v < nidata; ++v) hi2(v, n) = hi(v, src);
  }
  Kokkos::deep_copy(prtcl_rdata, hr2);
  Kokkos::deep_copy(prtcl_idata, hi2);
  nprtcl_thispack = nnew;

  // ---- post-merge totals from the ACTUAL arrays, and the verdict -------------------
  // A merge moves mass and momentum between particles, so both totals must be unchanged.
  // Reported as a relative error and escalated to a WARNING past a tolerance far above
  // round-off, so a real violation is impossible to read as noise.
  if (check_cons) {
    Real m_post_loc = 0.0, p_post_loc[3] = {0.0, 0.0, 0.0};
    for (int n = 0; n < nnew; ++n) {
      const Real m = hr2(IPM, n);
      m_post_loc += m;
      p_post_loc[0] += m*hr2(IPVX, n);
      p_post_loc[1] += m*hr2(IPVY, n);
      p_post_loc[2] += m*hr2(IPVZ, n);
    }
    Real loc[8] = {m_pre_loc, p_pre_loc[0], p_pre_loc[1], p_pre_loc[2],
                   m_post_loc, p_post_loc[0], p_post_loc[1], p_post_loc[2]};
    Real glb[8];
    for (int c = 0; c < 8; ++c) glb[c] = loc[c];
#if MPI_PARALLEL_ENABLED
    if (global_variable::nranks > 1) {
      MPI_Allreduce(loc, glb, 8, MPI_ATHENA_REAL, MPI_SUM, MPI_COMM_WORLD);
    }
#endif
    const Real dM = glb[4] - glb[0];
    const Real dP[3] = {glb[5]-glb[1], glb[6]-glb[2], glb[7]-glb[3]};
    const Real mscale = std::max(std::abs(glb[0]), 1.0e-300);
    const Real pscale = std::max({std::abs(glb[1]), std::abs(glb[2]),
                                  std::abs(glb[3]), 1.0e-300});
    const Real relM = std::abs(dM)/mscale;
    const Real relP = std::max({std::abs(dP[0]), std::abs(dP[1]),
                                std::abs(dP[2])})/pscale;
    if (my_rank == 0) {
      std::printf("  merge conservation (measured): dM/M=% .3e  dP/P=% .3e"
                  "  (M %.6e -> %.6e)\n", relM, relP, glb[0], glb[4]);
      if (relM > 1.0e-10 || relP > 1.0e-8) {
        std::cout << "### WARNING in Particles::MergeSinks: the merge did NOT conserve "
                  << "sink mass/momentum (dM/M=" << relM << ", dP/P=" << relP
                  << "). Check for duplicate PTAGs across ranks." << std::endl;
      }
    }
  }
  return refresh_return();
}

}  // namespace particles
