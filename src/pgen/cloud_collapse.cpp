//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file cloud_collapse.cpp
//! \brief Problem generator for gravitational collapse of turbulent cloud

// C headers

// C++ headers
#include <cmath>
#include <iostream> // cout

#include "athena.hpp"
#include "parameter_input.hpp"
#include "coordinates/cell_locations.hpp"
#include "mesh/mesh.hpp"
#include "hydro/hydro.hpp"
#include "mhd/mhd.hpp"
#include "pgen.hpp"

//========================================================================================
//! \fn void MeshBlock::ProblemGenerator(ParameterInput *pin)
//  \brief
//========================================================================================
void ProblemGenerator::UserProblem(ParameterInput *pin, const bool restart) {
  if (restart) return;

  MeshBlockPack *pmbp = pmy_mesh_->pmb_pack;
  auto &indcs = pmy_mesh_->mb_indcs;

  if (pmbp->phydro == nullptr && pmbp->pmhd == nullptr) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
       << "Turbulence problem generator can only be run with Hydro and/or MHD, but no "
       << "<hydro> or <mhd> block in input file" << std::endl;
    exit(EXIT_FAILURE);
  }

  if (pmbp->pcoord->is_special_relativistic ||
      pmbp->pcoord->is_general_relativistic) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl
              << "cloud_collapse requires non-relativistic Hydro/MHD" << std::endl;
    std::exit(EXIT_FAILURE);
  }

  if (pmbp->phydro != nullptr && pmbp->phydro->peos->eos_data.is_ideal) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl
              << "cloud_collapse requires isothermal Hydro" << std::endl;
    std::exit(EXIT_FAILURE);
  }

  if (pmbp->pmhd != nullptr && pmbp->pmhd->peos->eos_data.is_ideal) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl
              << "cloud_collapse requires isothermal MHD" << std::endl;
    std::exit(EXIT_FAILURE);
  }

  // capture variables for kernel
  int &is = indcs.is; int &ie = indcs.ie;
  int &js = indcs.js; int &je = indcs.je;
  int &ks = indcs.ks; int &ke = indcs.ke;

  // Initialize Hydro variables -------------------------------
  if (pmbp->phydro != nullptr) {
    auto &u0 = pmbp->phydro->u0;

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
    auto &u0 = pmbp->pmhd->u0;
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

  return;
}
