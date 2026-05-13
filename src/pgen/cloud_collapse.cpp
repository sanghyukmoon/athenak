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
#include <ctime>
#include <sstream>
#include <stdexcept>

// Athena++ headers
#include "../athena.hpp"
#include "../athena_arrays.hpp"
#include "../coordinates/coordinates.hpp"
#include "../eos/eos.hpp"
#include "../fft/athena_fft.hpp"
#include "../fft/perturbation.hpp"
#include "../field/field.hpp"
#include "../globals.hpp"
#include "../gravity/gravity.hpp"
#include "../hydro/hydro.hpp"
#include "../mesh/mesh.hpp"
#include "../parameter_input.hpp"
#include "../scalars/scalars.hpp"
#include "../utils/utils.hpp"

#ifdef OPENMP_PARALLEL
#include <omp.h>
#endif

// user function
void CollectCounters(Mesh *pm);
int JeansCondition(MeshBlock *pmb);
int MassCondition(MeshBlock *pmb);
Real Mach, njeans, dm_max, sfe_term;

//========================================================================================
//! \fn void Mesh::InitUserMeshData(ParameterInput *pin)
//  \brief
//========================================================================================
void Mesh::InitUserMeshData(ParameterInput *pin) {
  if (NON_BAROTROPIC_EOS) {
    std::stringstream msg;
    msg << "This problem generator does not support adiabatic EOS." << std::endl;
    ATHENA_ERROR(msg);
    return;
  }

  if (SELF_GRAVITY_ENABLED) {
    SetGravitationalConstant(PI);
  }

  Mach = pin->GetReal("problem", "Mach");
  sfe_term = pin->GetReal("problem", "sfe_term");

  if (adaptive) {
    njeans = pin->GetReal("mesh", "njeans");
    dm_max = pin->GetReal("mesh", "dm_max");
//    EnrollUserRefinementCondition(JeansCondition);
    EnrollUserRefinementCondition(MassCondition);
  }
  return;
}

//========================================================================================
//! \fn void MeshBlock::InitUserMeshBlockData(ParameterInput *pin)
//! \brief Function to initialize problem-specific data in MeshBlock class.  Can also be
//! used to initialize variables which are global to other functions in this file.
//! Called in MeshBlock constructor before ProblemGenerator.
//========================================================================================

void MeshBlock::InitUserMeshBlockData(ParameterInput *pin) {
  for (Particles* ppar : ppars)
    ppar->ToggleParHstOutFlag();
  return;
}

//========================================================================================
//! \fn void MeshBlock::ProblemGenerator(ParameterInput *pin)
//  \brief
//========================================================================================
void MeshBlock::ProblemGenerator(ParameterInput *pin) {
  for (int k=ks; k<=ke; k++) {
    for (int j=js; j<=je; j++) {
      for (int i=is; i<=ie; i++) {
        phydro->u(IDN,k,j,i) = 1.0;
        phydro->u(IM1,k,j,i) = 0.0;
        phydro->u(IM2,k,j,i) = 0.0;
        phydro->u(IM3,k,j,i) = 0.0;
        if (NSCALARS > 0) {
          for (int n=0; n<NSCALARS; ++n) {
            pscalars->s(n,k,j,i) = 0.0;
          }
        }
      }
    }
  }

  if (MAGNETIC_FIELDS_ENABLED) {
    Real mu_phi = pin->GetReal("problem", "mu_phi");
    Real lbox = pmy_mesh->mesh_size.x3len;
    for (int k=ks; k<=ke; k++) {
      for (int j=js; j<=je; j++) {
        for (int i=is; i<=ie+1; i++) {
          pfield->b.x1f(k,j,i) = 0.0;
        }
      }
    }
    for (int k=ks; k<=ke; k++) {
      for (int j=js; j<=je+1; j++) {
        for (int i=is; i<=ie; i++) {
          pfield->b.x2f(k,j,i) = 0.0;
        }
      }
    }
    for (int k=ks; k<=ke+1; k++) {
      for (int j=js; j<=je; j++) {
        for (int i=is; i<=ie; i++) {
          pfield->b.x3f(k,j,i) = PI*lbox / mu_phi;
        }
      }
    }
  }
}


//========================================================================================
//! \fn void Mesh::PostInitialize(ParameterInput *pin)
//  \brief
//========================================================================================
void Mesh::PostInitialize(int res_flag, ParameterInput *pin) {
  if (res_flag != 0)
    return;
  TurbulenceDriver trbd = TurbulenceDriver(this, pin);
  trbd.dedt = 0.5*mesh_size.x1len*mesh_size.x2len*mesh_size.x3len*SQR(Mach);
  if (trbd.turb_flag > 1) {
    std::stringstream msg;
    msg << "This problem generator requires decaying turbulence. Set turb_flag=1."
        << std::endl;
    ATHENA_ERROR(msg);
  } else {
    trbd.Driving();
  }
}

//========================================================================================
//! \fn void Mesh::UserWorkInLoop()
//! \brief Function called once every time step for user-defined work.
//========================================================================================
void Mesh::UserWorkInLoop() {
  // check number of bad cells
  CollectCounters(this);

  // Terminate the simulation when reaching the target SFE
  Real m0 = GetTotalCells()*my_blocks(0)->pcoord->GetCellVolume(0, 0, 0);
  Real msp = 0;
  for (int b = 0; b < nblocal; ++b) {
    MeshBlock *pmb(my_blocks(b));
    for (Particles *ppar : pmb->ppars) {
      for (int idx=0; idx<ppar->GetNumPar(); ++idx) {
        msp += ppar->mass(idx);
      }
    }
  }
  Real sfe = msp / m0;

  if (Globals::my_rank == 0) {
    MPI_Reduce(MPI_IN_PLACE, &sfe, 1, MPI_ATHENA_REAL, MPI_SUM, 0, MPI_COMM_WORLD);
    if (sfe > sfe_term) {
      std::raise(SIGTERM);
    }
  } else {
    MPI_Reduce(&sfe, &sfe, 1, MPI_ATHENA_REAL, MPI_SUM, 0, MPI_COMM_WORLD);
  }

}

//========================================================================================
//! \fn void MeshBlock::UserWorkInLoop()
//! \brief Function called once every time step for user-defined work.
//========================================================================================

void MeshBlock::UserWorkInLoop() {
//  Real phi_x = 1.12;
//  // TODO(SMOON) forbid x1len != x2len != x3len
//  Real lmb_sonic = pmy_mesh->mesh_size.x1len / SQR(Mach);
//  Real critical_density_KM05 = SQR(phi_x/lmb_sonic);
//  if (NSCALARS > 0) {
//    for (int n=0; n<NSCALARS; ++n) {
//      for (int k=ks; k<=ke; k++) {
//        for (int j=js; j<=je; j++) {
//          for (int i=is; i<=ie; i++) {
//            if (phydro->u(IDN,k,j,i) > (1 << n)*critical_density_KM05) {
//              pscalars->s(n,k,j,i) = 1.0;
//            }
//          }
//        }
//      }
//    }
//  }
  return;
}

//========================================================================================
//! \fn void Mesh::UserWorkAfterLoop(ParameterInput *pin)
//  \brief
//========================================================================================
void Mesh::UserWorkAfterLoop(ParameterInput *pin) {
  // do nothing
}

//========================================================================================
//! \fn void CollectCounters(Mesh *pm)
//! \brief collect counters
//========================================================================================
void CollectCounters(Mesh *pm) {
  int nbad_d=0, nbad_p=0;

  // summing up over meshblocks within the rank
  for (int b=0; b<pm->nblocal; ++b) {
    MeshBlock *pmb = pm->my_blocks(b);
    nbad_d += pmb->nbad_d;
    nbad_p += pmb->nbad_p;
  }

#ifdef MPI_PARALLEL
  MPI_Allreduce(MPI_IN_PLACE, &nbad_d, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, &nbad_p, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
#endif

  if (Globals::my_rank == 0) {
    if (nbad_p > 0) std::cerr << nbad_p << " cells had negative pressure" << std::endl;
    if (nbad_d > 0) std::cerr << nbad_d << " cells had negative density" << std::endl;
  }
}

// Jeans Condition
int JeansCondition(MeshBlock *pmb) {
  Real njmin = HUGE_NUMBER;
  const Real idx = 1.0 / pmb->pcoord->dx1f(0); // assuming uniform cubic cells
  for (int k = pmb->ks; k<=pmb->ke; ++k) {
    for (int j = pmb->js; j<=pmb->je; ++j) {
      for (int i = pmb->is; i<=pmb->ie; ++i) {
        Real nj = idx / std::sqrt(pmb->phydro->w(IDN,k,j,i));
        njmin = std::min(njmin, nj);
      }
    }
  }
  if (njmin < njeans)
    return 1;
  if (njmin > njeans * 2.5)
    return -1;
  return 0;
}

// Mass Condition
int MassCondition(MeshBlock *pmb) {
  Real dm = TINY_NUMBER;

  const Real dx = pmb->pcoord->dx1f(0); // assuming uniform cubic cells
  for (int k = pmb->ks; k<=pmb->ke; ++k) {
    for (int j = pmb->js; j<=pmb->je; ++j) {
      for (int i = pmb->is; i<=pmb->ie; ++i) {
        // maximum cell mass in a meshblock
        dm = std::max(dm, pmb->phydro->w(IDN,k,j,i)*dx*dx*dx);
      }
    }
  }
// refinement criterion:
// cell mass cannot exceed dm_max
  if (dm > dm_max)
    return 1;
// derefinement criterion
// If a MeshBlock is refined, dm decreases by a factor of 8. Therefore, the coefficient
// multiplied to dm_max must be smaller than 0.125: otherwise, the MeshBlock will be
// flagged for derefinement whenever it is refined.
  if (dm < 0.1*dm_max)
    return -1;
  return 0;
}
