#ifndef PARTICLES_SINK_PARTICLES_HPP_
#define PARTICLES_SINK_PARTICLES_HPP_
//========================================================================================
// AthenaK astrophysical fluid dynamics code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file sink_particles.hpp
//! \brief helpers for sink convergence, merging and refinement criteria

#include <algorithm>
#include <array>
#include <cmath>
#include <vector>

#include "athena.hpp"
#include "globals.hpp"
#include "coordinates/cell_locations.hpp"
#include "mesh/mesh.hpp"
#include "particles/particles.hpp"

#if MPI_PARALLEL_ENABLED
#include <mpi.h>
#endif

namespace particles {

//----------------------------------------------------------------------------------------
// Net mass inflow on each pair of CV faces. Uniform face areas cancel within an axis.
// Use conserved variables so the check sees the same state as the proposed reset.
template<class View>
KOKKOS_INLINE_FUNCTION bool ConvergingFlow(const View &u, int m, int i, int j, int k,
                                          Real vx, Real vy, Real vz) {
  const Real vref[3] = {vx, vy, vz};
  for (int axis=0; axis<3; ++axis) {
    Real inflow = 0.0;
    for (int a=-1; a<=1; ++a) for (int b=-1; b<=1; ++b) {
      int l[3]={i,j,k}, r[3]={i,j,k};
      l[axis]--; r[axis]++;
      l[(axis+1)%3]+=a; r[(axis+1)%3]+=a;
      l[(axis+2)%3]+=b; r[(axis+2)%3]+=b;
      const Real dl=u(m,IDN,l[2],l[1],l[0]), dr=u(m,IDN,r[2],r[1],r[0]);
      if (!(dl > 0.0 && dr > 0.0)) return false;
      const Real ml=u(m,IM1+axis,l[2],l[1],l[0]);
      const Real mr=u(m,IM1+axis,r[2],r[1],r[0]);
      inflow += (ml-dl*vref[axis])-(mr-dr*vref[axis]);
    }
    if (!(inflow > 0.0)) return false;
  }
  return true;
}

//----------------------------------------------------------------------------------------
// Host-only, deterministic global pair reduction. Candidate weights never become mass.
template<class T> struct SinkMergeRecord {
  std::array<T,3> x, v;
  T mass, weight;
  int tag;
  bool alive = true, changed = false;
};
template<class T> T SinkMinimumImage(T d, T length, bool periodic) {
  return periodic ? d-length*std::floor(d/length+T(0.5)) : d;
}
template<class T> bool SinkContact(const SinkMergeRecord<T> &a,
    const SinkMergeRecord<T> &b, const std::array<T,3> &origin,
    const std::array<T,3> &upper, const std::array<int,3> &nx_eff, bool periodic) {
  // All sinks use the same finest-level cell spacing.
  int touches=0;
  for (int c=0; c<3; ++c) {
    const T length=upper[c]-origin[c];
    T x=a.x[c], y=b.x[c];
    if (periodic) {
      x-=length*std::floor((x-origin[c])/length);
      y-=length*std::floor((y-origin[c])/length);
    }
    const int i=CellCenterIndex(x,nx_eff[c],origin[c],upper[c]);
    const int j=CellCenterIndex(y,nx_eff[c],origin[c],upper[c]);
    int distance=std::abs(i-j);
    if (periodic) {
      distance=std::min(distance,nx_eff[c]-distance);
    }
    if (distance > 3) return false;
    if (distance == 3) {
      if (++touches > 1) return false;
    }
  }
  return true;
}
template<class T> void ResolveSinkMergers(
    std::vector<SinkMergeRecord<T>> &s, const std::array<T,3> &origin,
    const std::array<T,3> &upper, const std::array<int,3> &nx_eff,
    bool periodic) {
  const std::array<T,3> length={upper[0]-origin[0],upper[1]-origin[1],
                                upper[2]-origin[2]};
  std::sort(s.begin(),s.end(),[](const auto &a,const auto &b){return a.tag<b.tag;});
  for (;;) {
    bool merged=false;
    for (size_t i=0; i<s.size() && !merged; ++i) {
      if (!s[i].alive) continue;
      for (size_t j=i+1; j<s.size(); ++j) {
        if (!s[j].alive ||
            !SinkContact(s[i],s[j],origin,upper,nx_eff,periodic)) continue;
        auto &a=s[i]; auto &b=s[j];
        const bool anew=a.mass==T(0), bnew=b.mass==T(0);
        if (anew != bnew) {
          (anew ? a : b).alive=false;
          (anew ? b : a).changed=true;
        } else {
          // Most massive survives; equal masses/weights use the lowest stable tag.
          size_t keep=i, drop=j;
          if (b.mass>a.mass) {keep=j; drop=i;}
          auto &u=s[keep]; auto &v=s[drop];
          T wu=anew?u.weight:u.mass, wv=anew?v.weight:v.mass, total=wu+wv;
          for (int c=0; c<3; ++c) {
            u.x[c]+=wv/total*SinkMinimumImage(v.x[c]-u.x[c],length[c],periodic);
            if(periodic) u.x[c]-=length[c]*std::floor((u.x[c]-origin[c])/length[c]);
            u.v[c]=anew?T(0):(wu*u.v[c]+wv*v.v[c])/total;
          }
          u.mass+=v.mass; u.weight+=v.weight; u.changed=true; v.alive=false;
        }
        merged=true;
        break; // Restart global scan: the new COM may contact an earlier particle.
      }
    }
    if (!merged) break;
  }
}

}  // namespace particles

//----------------------------------------------------------------------------------------
//! \fn DualArray1D<Real> GatherAllSinkPositions(MeshBlockPack*, int&)
//! \brief GatherAllSinkPositions(): every sink's (x,y,z) on every rank, for use by AMR
//! refinement criteria in problem generators.
//!
//! WHY THIS EXISTS. A user refinement function fills `refine_flag` for the MeshBlocks of
//! the rank that runs it (CheckForRefinement then Allgathers the flags across ranks). So
//! a criterion of the form "refine blocks near a sink" must be evaluated against ALL
//! sinks, not just `ppart->nprtcl_thispack` (the ones this rank owns). Using only local
//! particles makes the refinement -- and therefore the whole mesh -- depend on the rank
//! count: blocks that neighbour an off-rank sink are never flagged, and because these
//! criteria default to "derefine" they are actively coarsened. The sink then sits against
//! an asymmetric coarse-fine boundary, the particle-mesh deposit/gather pair stops
//! cancelling, and the sink picks up a large spurious force. Measured on part_orbit_amr
//! with 4 ranks before this was fixed: 78 MeshBlocks instead of 120, and an out-of-plane
//! force g_z ~ 0.9 where symmetry requires 0.
//!
//! Returns a device-resident flat array [x0,y0,z0, x1,y1,z1, ...] of length 3*nsink_all.
//! MUST be called by every rank (it contains MPI collectives). Cheap: sinks number in the
//! tens, and the gather is O(nranks).

inline DualArray1D<Real> GatherAllSinkPositions(MeshBlockPack *pmbp, int &nsink_all) {
  const int npart = (pmbp->ppart == nullptr) ? 0 : pmbp->ppart->nprtcl_thispack;
  nsink_all = npart;

  // pull this rank's positions to the host
  std::vector<Real> pos(3*std::max(npart, 0));
  if (npart > 0) {
    auto pr = pmbp->ppart->prtcl_rdata;
    auto pr_h = Kokkos::create_mirror_view(pr);
    Kokkos::deep_copy(pr_h, pr);
    for (int p = 0; p < npart; ++p) {
      pos[3*p    ] = pr_h(IPX, p);
      pos[3*p + 1] = pr_h(IPY, p);
      pos[3*p + 2] = pr_h(IPZ, p);
    }
  }

#if MPI_PARALLEL_ENABLED
  const int nranks = global_variable::nranks;
  if (nranks > 1) {
    std::vector<int> ncnt(nranks, 0);
    MPI_Allgather(&npart, 1, MPI_INT, ncnt.data(), 1, MPI_INT, MPI_COMM_WORLD);
    std::vector<int> rcnt(nranks), displ(nranks, 0);
    nsink_all = 0;
    for (int r = 0; r < nranks; ++r) { rcnt[r] = 3*ncnt[r]; nsink_all += ncnt[r]; }
    for (int r = 1; r < nranks; ++r) { displ[r] = displ[r-1] + rcnt[r-1]; }
    std::vector<Real> all(3*std::max(nsink_all, 0));
    MPI_Allgatherv(pos.data(), 3*npart, MPI_ATHENA_REAL, all.data(), rcnt.data(),
                   displ.data(), MPI_ATHENA_REAL, MPI_COMM_WORLD);
    pos.swap(all);
  }
#endif

  // upload to the device for use inside the refinement kernel
  DualArray1D<Real> spos("sink_pos_all", 3*std::max(nsink_all, 1));
  for (int i = 0; i < 3*nsink_all; ++i) { spos.h_view(i) = pos[i]; }
  spos.template modify<HostMemSpace>();
  spos.template sync<DevExeSpace>();
  return spos;
}

#endif  // PARTICLES_SINK_PARTICLES_HPP_
