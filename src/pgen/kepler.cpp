//========================================================================================
// Athena++ astrophysical MHD code, Kokkos version
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file kepler.cpp
//! \brief Problem generator that initializes particle positions and velocities.

#include <iostream>

#include "parameter_input.hpp"
#include "athena.hpp"
#include "mesh/mesh.hpp"
#include "particles/particles.hpp"

//----------------------------------------------------------------------------------------
//! \fn ProblemGenerator::UserProblem_()
//! \brief Problem Generator for random particle positions/velocities

void ProblemGenerator::UserProblem(ParameterInput *pin, const bool restart) {
  if (restart) return;

  MeshBlockPack *pmbp = pmy_mesh_->pmb_pack;
  if (pmbp->ppart == nullptr) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "Kepler test requires <particles> block in input file"
              << std::endl;
    exit(EXIT_FAILURE);
  }

  Real px = 1.0;
  Real py = 0.0;
  Real pz = 0.0;
  Real vx = 0.0;
  Real vy = 1.0;
  Real vz = 0.0;

  // capture variables for the kernel
  auto &mbsize = pmbp->pmb->mb_size;
  auto &pr = pmbp->ppart->prtcl_rdata;
  auto &pi = pmbp->ppart->prtcl_idata;
  auto &npart = pmbp->ppart->nprtcl_thispack;
  auto gids = pmbp->gids;
  auto gide = pmbp->gide;

  bool found = false;
  int gid_containing_par = -1;
  for (int m=0; m<pmbp->nmb_thispack; ++m) {
    Real x1min = mbsize.h_view(m).x1min;
    Real x1max = mbsize.h_view(m).x1max;
    Real x2min = mbsize.h_view(m).x2min;
    Real x2max = mbsize.h_view(m).x2max;
    Real x3min = mbsize.h_view(m).x3min;
    Real x3max = mbsize.h_view(m).x3max;

    bool in_x1 = (px >= x1min && px < x1max);
    bool in_x2 = (py >= x2min && py < x2max);
    bool in_x3 = (pz >= x3min && pz < x3max);

    // Respect dimensionality of the mesh: if mesh is effectively 1D/2D, accept by ignoring
    // the unused directions (multi_d and three_d flags live on Mesh).
    bool multi_d = pmbp->pmesh->multi_d;
    bool three_d = pmbp->pmesh->three_d;
    if (!three_d) in_x3 = true;
    if (!multi_d) in_x2 = true;

    if (in_x1 && in_x2 && in_x3) {
      gid_containing_par = gids + m;
      found = true;
      break;
    }
  }

  if (found) {
    // initialize particles
    par_for("part_update",DevExeSpace(),0,(npart-1),
    KOKKOS_LAMBDA(const int p) {
      pi(PGID,p) = gid_containing_par;
      pr(IPX,p) = px;
      pr(IPY,p) = py;
      pr(IPZ,p) = pz;
      pr(IPVX,p) = vx;
      pr(IPVY,p) = vy;
      pr(IPVZ,p) = vz;
    });
  }


  // set timestep (which will remain constant for entire run
  // Assumes uniform mesh (no SMR or AMR)
  // Assumes velocities normalized to one, so dt=min(dx)
  Real &dtnew_ = pmbp->ppart->dtnew;
  dtnew_ = 2*M_PI/1000;

  return;
}
