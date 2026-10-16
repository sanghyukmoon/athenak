//========================================================================================
// AthenaK astrophysical fluid dynamics code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file particle_history.cpp
//! \brief Individual sink histories, including their massless formation state.

#include <sys/stat.h>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <limits>

#include "athena.hpp"
#include "mesh/mesh.hpp"
#include "particles/particles.hpp"
#include "outputs.hpp"

ParticleHistoryOutput::ParticleHistoryOutput(ParameterInput *pin, Mesh *pm,
                                             OutputParameters op) :
    BaseTypeOutput(pin, pm, op) {
  auto *pp = pm->pmb_pack->ppart;
  if (pp == nullptr || pp->particle_type != ParticleType::sink) {
    std::cout << "### FATAL ERROR: parhst output requires sink particles" << std::endl;
    std::exit(EXIT_FAILURE);
  }
  mkdir("parhst", 0775);
  pp->phistory = this;
}

void ParticleHistoryOutput::WriteParticles(Mesh *pm, int first, int count,
                                           Real time, int cycle) {
  if (count == 0) return;
  auto *pp = pm->pmb_pack->ppart;
  auto range = std::make_pair(first, first + count);
  auto ids = Kokkos::create_mirror_view_and_copy(HostMemSpace(),
      Kokkos::subview(pp->prtcl_idata, static_cast<int>(PTAG), range));
  // Copy before selecting particles: a partial real-state slice is noncontiguous.
  auto state = Kokkos::create_mirror_view_and_copy(HostMemSpace(), pp->prtcl_rdata);
  const int fields[7] = {IPM, IPX, IPY, IPZ, IPVX, IPVY, IPVZ};
  const int precision = std::numeric_limits<Real>::max_digits10;
  for (int p=0; p<count; ++p) {
    std::string fname = "parhst/" + out_params.file_basename + ".par" +
                        std::to_string(ids(p)) + ".csv";
    FILE *file = std::fopen(fname.c_str(), "a");
    if (file == nullptr) {
      std::cout << "### FATAL ERROR: could not open " << fname << std::endl;
      std::exit(EXIT_FAILURE);
    }
    std::fseek(file, 0, SEEK_END);
    if (std::ftell(file) == 0) {
      std::fprintf(file, "time,cycle,pid,mass,x1,x2,x3,v1,v2,v3\n");
    }
    std::fprintf(file, "%.*g,%d,%d", precision, static_cast<double>(time), cycle, ids(p));
    for (int v=0; v<7; ++v) {
      std::fprintf(file, ",%.*g", precision, static_cast<double>(state(fields[v], first + p)));
    }
    std::fprintf(file, "\n");
    std::fclose(file);
  }
}

void ParticleHistoryOutput::WriteOutputFile(Mesh *pm, ParameterInput *pin) {
  WriteParticles(pm, 0, pm->pmb_pack->ppart->nprtcl_thispack, pm->time, pm->ncycle);
  out_params.file_number++;
  out_params.last_time = out_params.last_time < 0.0 ? pm->time :
                        out_params.last_time + out_params.dt;
  pin->SetInteger(out_params.block_name, "file_number", out_params.file_number);
  pin->SetReal(out_params.block_name, "last_time", out_params.last_time);
}
