//========================================================================================
// AthenaK astrophysical fluid dynamics code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file particle_binary.cpp
//! \brief Native sink snapshots with integer IDs and full-precision physical state.

#include <sys/stat.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <vector>

#include "athena.hpp"
#include "globals.hpp"
#include "mesh/mesh.hpp"
#include "particles/particles.hpp"
#include "outputs.hpp"

#if MPI_PARALLEL_ENABLED
#include <mpi.h>
#endif

ParticleBinaryOutput::ParticleBinaryOutput(ParameterInput *pin, Mesh *pm,
                                           OutputParameters op) :
    BaseTypeOutput(pin, pm, op) {
  auto *pp = pm->pmb_pack->ppart;
  if (pp == nullptr || pp->particle_type != ParticleType::sink) {
    std::cout << "### FATAL ERROR: parbin output requires sink particles" << std::endl;
    std::exit(EXIT_FAILURE);
  }
  mkdir("parbin", 0775);
}

void ParticleBinaryOutput::WriteOutputFile(Mesh *pm, ParameterInput *pin) {
  auto *pp = pm->pmb_pack->ppart;
  int count = pp->nprtcl_thispack;
  auto ids = Kokkos::create_mirror_view_and_copy(HostMemSpace(),
      Kokkos::subview(pp->prtcl_idata, static_cast<int>(PTAG), std::make_pair(0, count)));
  auto state = Kokkos::create_mirror_view_and_copy(HostMemSpace(),
      Kokkos::subview(pp->prtcl_rdata, Kokkos::ALL, std::make_pair(0, count)));
  const int fields[7] = {IPM, IPX, IPY, IPZ, IPVX, IPVY, IPVZ};
  std::vector<std::int32_t> local_ids(count), all_ids;
  std::vector<Real> local_state(7*count), all_state;
  for (int p=0; p<count; ++p) {
    local_ids[p] = ids(p);
    for (int v=0; v<7; ++v) local_state[7*p+v] = state(fields[v], p);
  }
  int total = count;
#if MPI_PARALLEL_ENABLED
  std::vector<int> counts(global_variable::nranks), offsets(global_variable::nranks);
  MPI_Allgather(&count, 1, MPI_INT, counts.data(), 1, MPI_INT, MPI_COMM_WORLD);
  total = 0;
  for (int rank=0; rank<global_variable::nranks; ++rank) {
    offsets[rank] = total;
    total += counts[rank];
  }
  if (global_variable::my_rank == 0) all_ids.resize(total);
  MPI_Gatherv(local_ids.data(), count, MPI_INT32_T, all_ids.data(), counts.data(),
              offsets.data(), MPI_INT32_T, 0, MPI_COMM_WORLD);
  for (int rank=0; rank<global_variable::nranks; ++rank) {
    counts[rank] *= 7;
    offsets[rank] *= 7;
  }
  if (global_variable::my_rank == 0) all_state.resize(7*total);
  MPI_Gatherv(local_state.data(), 7*count, MPI_ATHENA_REAL, all_state.data(),
              counts.data(), offsets.data(), MPI_ATHENA_REAL, 0, MPI_COMM_WORLD);
#else
  all_ids = std::move(local_ids);
  all_state = std::move(local_state);
#endif
  if (global_variable::my_rank == 0) {
    char number[16];
    std::snprintf(number, sizeof(number), ".%05d", out_params.file_number);
    std::string fname = "parbin/" + out_params.file_basename + number + ".parbin";
    FILE *file = std::fopen(fname.c_str(), "wb");
    if (file == nullptr) {
      std::cout << "### FATAL ERROR: could not open " << fname << std::endl;
      std::exit(EXIT_FAILURE);
    }
    std::fprintf(file, "Athena particle output version=1.0\n"
                 "time=%.*g\ncycle=%d\noutput number=%d\nnumber of particles=%d\n"
                 "size of real=%zu\nbyte order=%s\ndata:\n",
                 std::numeric_limits<Real>::max_digits10, static_cast<double>(pm->time),
                 pm->ncycle, out_params.file_number, total, sizeof(Real),
                 IsBigEndian() ? "big" : "little");
    if (std::fwrite(all_ids.data(), sizeof(std::int32_t), total, file) != total ||
        std::fwrite(all_state.data(), sizeof(Real), 7*total, file) != 7*total) {
      std::cout << "### FATAL ERROR: particle data not written to " << fname << std::endl;
      std::exit(EXIT_FAILURE);
    }
    std::fclose(file);
  }
  out_params.file_number++;
  out_params.last_time = out_params.last_time < 0.0 ? pm->time :
                        out_params.last_time + out_params.dt;
  pin->SetInteger(out_params.block_name, "file_number", out_params.file_number);
  pin->SetReal(out_params.block_name, "last_time", out_params.last_time);
}
