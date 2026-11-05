#ifndef PARTICLES_PARTICLES_HPP_
#define PARTICLES_PARTICLES_HPP_
//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file particles.hpp
//  \brief definitions for Particles class

#include <cstdint>
#include <map>
#include <memory>
#include <string>

#include "athena.hpp"
#include "parameter_input.hpp"
#include "tasklist/task_list.hpp"
#include "bvals/bvals.hpp"

// forward declarations

// constants that enumerate ParticlesPusher options
enum class ParticlesPusher {drift, leapfrog, rk3, lagrangian_tracer, lagrangian_mc};

// constants that enumerate ParticleTypes
enum class ParticleType {cosmic_ray, sink};

//----------------------------------------------------------------------------------------
//! \struct ParticlesTaskIDs
//  \brief container to hold TaskIDs of all particles tasks

struct ParticlesTaskIDs {
  TaskID setgid;
  TaskID deposit;
  TaskID flush;
  // forward ghost exchange of the particle-mesh deposit (the "inbox" half; see
  // ParticleMesh::RefreshGhosts*)
  TaskID dm_irecv;
  TaskID dm_rest;
  TaskID dm_send;
  TaskID dm_recv;
  TaskID dm_prol;
  TaskID dm_csend;
  TaskID dm_crecv;
  TaskID push;
  TaskID merge;
  TaskID accrete;
  TaskID create;
  TaskID newdt;
  TaskID newgid;
  TaskID count;
  TaskID irecv;
  TaskID sendp;
  TaskID recvp;
  TaskID crecv;
  TaskID csend;
  // Post-accretion gas refresh.
  TaskID gas_restu;
  TaskID gas_restb;
  TaskID gas_irecv;
  TaskID gas_sendu;
  TaskID gas_recvu;
  TaskID gas_sendb;
  TaskID gas_recvb;
  TaskID gas_prol;
  TaskID gas_bcs;
  TaskID gas_c2p;
  TaskID gas_newdt;
  TaskID gas_csend;
  TaskID gas_crecv;
  // gravitational-potential halo exchange, split into tasks like every other exchange
  TaskID xphi_irecv;
  TaskID xphi_rest;
  TaskID xphi_send;
  TaskID xphi_recv;
  TaskID xphi_bcs;
  TaskID xphi_prol;
  TaskID xphi_csend;
  TaskID xphi_crecv;
};

namespace particles {

// forward declaration
class ParticleMesh;

//----------------------------------------------------------------------------------------
//! \class Particles

class Particles {
  friend class ParticlesBoundaryValues;
 public:
  Particles(MeshBlockPack *ppack, ParameterInput *pin);
  ~Particles();

  // data
  ParticleType particle_type;
  int nprtcl_thispack;             // number of particles this MeshBlockPack
  int nrdata, nidata;
  DvceArray2D<Real> prtcl_rdata;   // real number properties each particle (x,v,etc.)
  DvceArray2D<int>  prtcl_idata;   // integer properties each particle (gid, tag, etc.)
  // Immutable post-merger snapshot for this step's accretion; not restart state.
  DvceArray2D<Real> global_prtcl_rdata;  // rows IPX through IPM
  DvceArray1D<int> global_prtcl_tags;
  // Particle timestep constraint, consumed by Mesh::NewTimeStep as dt = min(dt, dtnew).
  // Initialised to "no constraint" in the constructor. It MUST have a defined value from
  // the moment the module exists: with <particles>/ppc = 0 (creation-driven runs) there
  // are no particles to derive it from, NewTimeStep returns early, and an uninitialised
  // dtnew silently drives the global dt to zero -- the run then spins at t = 0 forever,
  // never advancing far enough for a sink to form and set it. Every sink pgen used to
  // seed this by hand, which worked only because they all remembered to.
  Real dtnew;

  ParticlesPusher pusher;
  Real point_mass_gm;  // temporary treatment of source term on particle

  // Particle CFL number (<particles>/cfl_par, default 0.5): dtnew = cfl_par*min(dx/|v|)
  // over particles. Mesh::NewTimeStep applies NO cfl factor to ppart->dtnew and caps
  // per-cycle dt growth at 2x, so 0.5 guarantees sinks cross at most one cell per step
  // (required by neighbor-based particle migration).
  Real cfl_par;

  bool accretion;
  bool creation;
  bool creation_finest_only;
  bool converging_flow;
  int created_total_ = 0;   // running count, for unique tags

  // Particle-mesh coupling layer (allocated for sink type, nullptr for tracers).
  ParticleMesh *ppm = nullptr;

  // Dedicated boundary-values object for the potential's halo exchange. Its own object,
  // NOT the particle-mesh deposit's (ppm->pmbval): one object per exchanged field is the
  // convention here, each carrying its own buffers, MPI requests and communicators, and
  // sharing one between two different fields couples their exchange state.
  MeshBoundaryValuesCC *pbval_phi = nullptr;

  // Boundary communication buffers and functions for particles
  ParticlesBoundaryValues *pbval_part;

  // container to hold names of TaskIDs
  ParticlesTaskIDs id;

  // functions...
  // ---- host / MPI phase timers (env SINK_TIMERS) --------------------------------------
  // Wall-clock accounting for the sink module's HOST and MPI regions -- precisely what a
  // Kokkos profiler cannot see. kokkos-tools reports device kernels by label, but the
  // per-step collectives and host<->device particle mirrors are invisible to it.
  //
  //   SINK_TIMERS=1  accumulate and report. Low perturbation, but Kokkos kernels are
  //                  asynchronous, so a region can absorb the completion of work launched
  //                  before it, charged at whatever call first synchronizes.
  //   SINK_TIMERS=2  additionally Kokkos::fence() + MPI_Barrier before each region and
  //                  account that separately as "wait". The region's own cost is then
  //                  isolated, and "wait" is the load imbalance that a collective merely
  //                  EXPOSES rather than causes. Perturbing by construction: an
  //                  attribution run, never a timing run.
  //
  // ST_REFRESH is nested inside ST_CREATE and ST_MERGE, so totals are inclusive.
  enum SinkTimerId {ST_CREATE=0, ST_MERGE, ST_REFRESH,
                    ST_NUM};
  int st_mode_ = 0;
  double st_time_[ST_NUM] = {};
  double st_wait_[ST_NUM] = {};
  std::int64_t st_calls_[ST_NUM] = {};
  Kokkos::Timer st_clock_;  // SINK_TIMERS clock
  double StBegin(int id);
  void   StEnd(int id, double t0);
  void   ReportSinkTimers(double wall_seconds);
  // scope guard, so a region with several early returns still stops its timer
  struct SinkTimerScope {
    Particles *p; int id; double t0;
    SinkTimerScope(Particles *pp, int i) : p(pp), id(i), t0(pp->StBegin(i)) {}
    ~SinkTimerScope() { p->StEnd(id, t0); }
  };

  void CreateParticleTags(ParameterInput *pin);
  void RefreshMeshParticleCounts();  // after pgen-side array resize
  // Resize the particle arrays to npart_new and, with them, every buffer that was sized
  // from the ppc-derived count in the constructor, then refresh the Mesh's bookkeeping.
  // Use this when restoring or seeding particles rather than a bare Kokkos::resize,
  // so a population larger than ppc implied also gets sufficient reset staging storage.
  void ResizeParticles(int npart_new);
  void AssembleTasks(std::map<std::string, std::shared_ptr<TaskList>> tl);
  TaskStatus SetGIDFromPosition(Driver *pdriver, int stage);
  void ReconcileOwnership(Driver *pdriver, int stage);  // setgid + cross-rank migration
  TaskStatus Deposit(Driver *pdriver, int stage);
  TaskStatus FlushDeposit(Driver *pdriver, int stage);
  // Forward ghost exchange of dmesh, completing the pair the flush starts: the flush
  // ADDS each block's ghost spill into the neighbour interiors that own it (the reverse
  // half), and these copy the now-complete interiors back OUT into the ghosts (the
  // forward half, which is the only half the gas needs). Thin wrappers so the steps can
  // be task-list entries; the work and the rationale live on ParticleMesh.
  TaskStatus DMeshInitRecv(Driver *pdriver, int stage);
  TaskStatus DMeshRestrict(Driver *pdriver, int stage);
  TaskStatus DMeshSend(Driver *pdriver, int stage);
  TaskStatus DMeshRecv(Driver *pdriver, int stage);
  TaskStatus DMeshProlongate(Driver *pdriver, int stage);
  TaskStatus DMeshClearSend(Driver *pdriver, int stage);
  TaskStatus DMeshClearRecv(Driver *pdriver, int stage);
  // Gravitational-potential halo exchange. GatherGravity reads phi two cells deep near
  // MeshBlock boundaries but the multigrid leaves only mg_nghost layers valid, so phi
  // needs a halo swap after each solve. Split into the same task sequence every other
  // field exchange uses -- receives posted in "before_stagen", restrict/send/recv/BC/
  // prolongate in "stagen" before the gather, buffers cleared in "after_stagen" -- rather
  // than one call that posts, sends and then SPINS until the receives drain. The spin
  // blocked the whole task list mid-stage on every rank, which is both a serialization
  // point and a structure no other exchange in the code uses.
  TaskStatus XPhiInitRecv(Driver *pdriver, int stage);
  TaskStatus XPhiRestrict(Driver *pdriver, int stage);
  TaskStatus XPhiSend(Driver *pdriver, int stage);
  TaskStatus XPhiRecv(Driver *pdriver, int stage);
  TaskStatus XPhiBCs(Driver *pdriver, int stage);
  TaskStatus XPhiProlongate(Driver *pdriver, int stage);
  TaskStatus XPhiClearSend(Driver *pdriver, int stage);
  TaskStatus XPhiClearRecv(Driver *pdriver, int stage);
  TaskStatus Push(Driver *pdriver, int stage);
  TaskStatus MergeSinks(Driver *pdriver, int stage);
  TaskStatus AccreteMass(Driver *pdriver, int stage);
  TaskStatus CreateSinks(Driver *pdriver, int stage);
  TaskStatus NewTimeStep(Driver *pdriver, int stage);
  TaskStatus NewGID(Driver *pdriver, int stage);
  TaskStatus SendCnt(Driver *pdriver, int stage);
  TaskStatus InitRecv(Driver *pdriver, int stage);
  TaskStatus SendP(Driver *pdriver, int stage);
  TaskStatus RecvP(Driver *pdriver, int stage);
  TaskStatus ClearSend(Driver *pdriver, int stage);
  TaskStatus ClearRecv(Driver *pdriver, int stage);

 private:
  MeshBlockPack* pmy_pack;  // ptr to MeshBlockPack containing this Particles
};

} // namespace particles
#endif // PARTICLES_PARTICLES_HPP_
