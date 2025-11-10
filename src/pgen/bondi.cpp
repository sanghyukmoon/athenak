//========================================================================================
// Athena++ astrophysical MHD code, Kokkos version
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file bondi.cpp
//! \brief Isothermal Bondi accretion problem with sink particle

#include <algorithm>
#include <iostream>
#include <vector>

#include "parameter_input.hpp"
#include "athena.hpp"
#include "mesh/mesh.hpp"
#include "coordinates/cell_locations.hpp"
#include "eos/eos.hpp"
#include "hydro/hydro.hpp"
#include "particles/particles.hpp"

namespace {

std::vector<Real> r_table, v_table, rho_table;
bool reset_ic = false;    // reset initial conditions after run

// ODE right-hand side for dv/dr (v positive inward)
KOKKOS_INLINE_FUNCTION
Real dvdr(const Real r, const Real v) {
  if (r == 0.5) {
    return 2.0; // avoid singularity at sonic point
  } else {
    return (2.0/r - 1.0/(r*r)) / (v - 1.0/v);
  }
}

//----------------------------------------------------------------------------------------
// Simple rk4 integrator
// Inputs:
//   x, y, h: current independent variable, dependent variable, and step size
// Outputs:
//   new value of dependent variable after step h

KOKKOS_INLINE_FUNCTION
Real rk4(const Real x, const Real y, const Real h) {
  Real k1 = h*dvdr(x, y);
  Real k2 = h*dvdr(x + 0.5*h, y + 0.5*k1);
  Real k3 = h*dvdr(x + 0.5*h, y + 0.5*k2);
  Real k4 = h*dvdr(x + h, y + k3);
  return y + (1.0/6.0)*(k1 + 2.0*k2 + 2.0*k3 + k4);
}

//----------------------------------------------------------------------------------------
// Function for returning corresponding Bondi accretion solution
// Inputs:
//   r_in: radial coordinate
// Outputs:
//   v_out, rho_out: variables pointed to set to radial velocity and density

KOKKOS_INLINE_FUNCTION
void InterpolateFromTable(const Real r_in, Real &v_out, Real &rho_out) {
  if (r_in <= r_table.front()) {
    v_out = v_table.front();
    rho_out = rho_table.front();
    return;
  } else if (r_in >= r_table.back()) {
    v_out = v_table.back();
    rho_out = rho_table.back();
    return;
  }
  // binary search
  size_t lo = 0, hi = r_table.size()-1;
  while (hi - lo > 1) {
    size_t mid = (lo + hi) >> 1;
    if (r_table[mid] <= r_in) {
      lo = mid;
    } else {
      hi = mid;
    }
  }
  Real r0 = r_table[lo], r1 = r_table[hi];
  Real w = (r_in - r0) / (r1 - r0);
  v_out = v_table[lo] * (1.0-w) + v_table[hi] * w;
  rho_out = rho_table[lo] * (1.0-w) + rho_table[hi] * w;
}

} // namespace

// prototypes for user-defined BCs and error functions
void FixedBondiInflow(Mesh *pm);
void BondiErrors(ParameterInput *pin, Mesh *pm);

//----------------------------------------------------------------------------------------
//! \fn void ProblemGenerator::UserProblem_()
//! \brief set initial conditions for isothermal Bondi accretion test
//  Compile with '-D PROBLEM=bondi' to enroll as user-specific problem generator

void ProblemGenerator::UserProblem(ParameterInput *pin, const bool restart) {
  MeshBlockPack *pmbp = pmy_mesh_->pmb_pack;

  if (pmbp->pcoord->is_general_relativistic) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl
              << "Isothermal bondi problem cannot be run with GR defined in <coord> block"
              << std::endl;
    exit(EXIT_FAILURE);
  }

  // set user-defined BCs and error function pointers
  // TODO(SMOON): implement these two functions
//  user_bcs_func = FixedBondiInflow;
//  pgen_final_func = BondiErrors;
  if (restart) return;

  // Read problem-specific parameters from input file
  // global parameters

  auto &mesh_size = pmy_mesh_->mesh_size;
  auto &mesh_indcs = pmy_mesh_->mesh_indcs;
  int hnx1 = mesh_indcs.nx1 >> 1;
  int hnx2 = mesh_indcs.nx2 >> 1;
  int hnx3 = mesh_indcs.nx3 >> 1;
  Real x1min = mesh_size.x1min;
  Real x1max = mesh_size.x1max;
  Real x2min = mesh_size.x2min;
  Real x2max = mesh_size.x2max;
  Real x3min = mesh_size.x3min;
  Real x3max = mesh_size.x3max;
  Real xc = x1min + (0.5 + hnx1)*mesh_size.dx1;
  Real yc = x2min + (0.5 + hnx2)*mesh_size.dx2;
  Real zc = x3min + (0.5 + hnx3)*mesh_size.dx3;
  Real rmax = std::sqrt(3)*std::max({std::abs(x1max - xc), std::abs(x1min - xc),
                                     std::abs(x2max - yc), std::abs(x2min - yc),
                                     std::abs(x3max - zc), std::abs(x3min - zc)});
  Real rmin = 0.5*std::min({mesh_size.dx1, mesh_size.dx2, mesh_size.dx3});

  Real n_r = 1e6; // number of radial points in Bondi table

  // Build the 1D Bondi table
  r_table.clear(); v_table.clear(); rho_table.clear();
  const Real r_s = 0.5; // sonic radius in unit of r_B
  Real h = (rmax - rmin)/n_r; // step size

  // outward integration (r increasing)
  // Start from the sonic radius
  Real r = r_s;
  Real v = -1.0;
  r_table.push_back(r);
  v_table.push_back(v);
  while (r <= rmax) {
    v = rk4(r, v, h);
    r += h;
    r_table.push_back(r);
    v_table.push_back(v);
  }
  // inward integration (r decreasing)
  r = r_s;
  v = -1.0;
  while (r >= rmin) {
    v = rk4(r, v, -h);
    r -= h;
    r_table.push_back(r);
    v_table.push_back(v);
  }

  // Sort table by increasing r
  std::vector<size_t> idx(r_table.size());
  for (size_t i=0; i<idx.size(); ++i) idx[i]=i;
  std::sort(idx.begin(), idx.end(),
            [&](size_t a, size_t b){return r_table[a] < r_table[b];});
  std::vector<Real> r_sorted, v_sorted;
  r_sorted.reserve(r_table.size());
  v_sorted.reserve(v_table.size());
  for (size_t k=0; k<idx.size(); ++k) {
    r_sorted.push_back(r_table[idx[k]]);
    v_sorted.push_back(v_table[idx[k]]);
  }
  r_table.swap(r_sorted);
  v_table.swap(v_sorted);

  // compute rho(r) from Bernoulli
  rho_table.resize(r_table.size());
  for (size_t i=0; i<r_table.size(); ++i) {
    Real r = r_table[i];
    Real v = v_table[i];
    rho_table[i] = std::exp(1.0/r - 2.0 - 0.5*(v*v - 1.0));
  }

  // capture variables for the kernel
  auto &indcs = pmy_mesh_->mb_indcs;
  auto &size = pmbp->pmb->mb_size;
  auto &coord = pmbp->pcoord->coord_data;
  int &ng = indcs.ng;
  int n1 = indcs.nx1 + 2*ng;
  int n2 = (indcs.nx2 > 1)? (indcs.nx2 + 2*ng) : 1;
  int n3 = (indcs.nx3 > 1)? (indcs.nx3 + 2*ng) : 1;
  int is = indcs.is;
  int js = indcs.js;
  int ks = indcs.ks;
  int nmb = pmbp->nmb_thispack;
  DvceArray5D<Real> w0_ = pmbp->phydro->w0;

  // Initialize primitive values (HYDRO ONLY)
  par_for("pgen_bondi", DevExeSpace(), 0,(nmb-1),0,(n3-1),0,(n2-1),0,(n1-1),
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    Real vr, rho;
    Real &x1min = size.d_view(m).x1min;
    Real &x1max = size.d_view(m).x1max;
    Real x1v = CellCenterX(i-is, indcs.nx1, x1min, x1max);

    Real &x2min = size.d_view(m).x2min;
    Real &x2max = size.d_view(m).x2max;
    Real x2v = CellCenterX(j-js, indcs.nx2, x2min, x2max);

    Real &x3min = size.d_view(m).x3min;
    Real &x3max = size.d_view(m).x3max;
    Real x3v = CellCenterX(k-ks, indcs.nx3, x3min, x3max);

    Real rad = std::sqrt(SQR(x1v-xc) + SQR(x2v-yc) + SQR(x3v-zc));

    InterpolateFromTable(rad, vr, rho);

    // decompose to Cartesian
    Real vx=0.0, vy=0.0, vz=0.0;
    if (rad > 0.0) {
      vx = vr * ((x1v - xc) / rad);
      vy = vr * ((x2v - yc) / rad);
      vz = vr * ((x3v - zc) / rad);
    }
    w0_(m,IDN,k,j,i) = rho;
    w0_(m,IVX,k,j,i) = vx;
    w0_(m,IVY,k,j,i) = vy;
    w0_(m,IVZ,k,j,i) = vz;
  });

  // Convert primitives to conserved
  auto &u0_ = pmbp->phydro->u0;
  auto &u1_ = pmbp->phydro->u1;
  if (reset_ic) {
    pmbp->phydro->peos->PrimToCons(w0_, u1_, 0, (n1-1), 0, (n2-1), 0, (n3-1));
  } else {
    pmbp->phydro->peos->PrimToCons(w0_, u0_, 0, (n1-1), 0, (n2-1), 0, (n3-1));
  }

  return;
}
