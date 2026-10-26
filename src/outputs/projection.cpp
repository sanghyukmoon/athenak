//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file projection.cpp
//! \brief Full-domain surface density and mass-weighted Cartesian vector projections.

#include <sys/stat.h>
#include <algorithm>
#include <climits>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>

#include "athena.hpp"
#include "globals.hpp"
#include "coordinates/coordinates.hpp"
#include "hydro/hydro.hpp"
#include "mhd/mhd.hpp"
#include "mesh/mesh.hpp"
#include "outputs.hpp"

#if MPI_PARALLEL_ENABLED
#include <mpi.h>
#endif

ProjectionOutput::ProjectionOutput(ParameterInput *pin, Mesh *pm, OutputParameters op) :
    BaseTypeOutput(pin, pm, op) {
  auto *pack = pm->pmb_pack;
  auto *coord = pack->pcoord;
  if (!pm->three_d || pm->multilevel ||
      coord->is_special_relativistic || coord->is_general_relativistic ||
      coord->is_dynamical_relativistic || pack->pionn != nullptr ||
      ((pack->phydro == nullptr) == (pack->pmhd == nullptr))) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl << "Projection output requires uniform 3D Newtonian hydro "
              << "or MHD with a single fluid" << std::endl;
    std::exit(EXIT_FAILURE);
  }
  if (op.include_gzs || op.gid != -1 || op.slice1 || op.slice2 || op.slice3) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl << "Projection output requires the full domain without "
              << "ghost zones, gid selection, or slices" << std::endl;
    std::exit(EXIT_FAILURE);
  }
  if (global_variable::my_rank == 0) mkdir("proj", 0775);
}

// Sum each block's line-of-sight cells before atomically adding to the global image.
void ProjectionOutput::CalculateProjection(Mesh *pm, int axis) {
  auto *pack = pm->pmb_pack;
  bool mhd = pack->pmhd != nullptr;
  int nfield = mhd ? 7 : 4;
  int fast_axis = axis == 0 ? 1 : 0;
  int slow_axis = axis == 2 ? 1 : 2;
  int counts[3] = {pm->mesh_indcs.nx1, pm->mesh_indcs.nx2, pm->mesh_indcs.nx3};
  int block_counts[3] = {pm->mb_indcs.nx1, pm->mb_indcs.nx2, pm->mb_indcs.nx3};
  int starts[3] = {pm->mb_indcs.is, pm->mb_indcs.js, pm->mb_indcs.ks};
  Real lengths[3] = {pm->mesh_size.x1max - pm->mesh_size.x1min,
                       pm->mesh_size.x2max - pm->mesh_size.x2min,
                       pm->mesh_size.x3max - pm->mesh_size.x3min};
  int nfast = counts[fast_axis], nslow = counts[slow_axis];
  int block_fast = block_counts[fast_axis], block_slow = block_counts[slow_axis];
  int nline = block_counts[axis];
  int is = starts[0], js = starts[1], ks = starts[2];
  Real dl = lengths[axis]/counts[axis];

  // Release the previous axis before allocation to avoid keeping two image buffers.
  if (image.extent(1) != nslow || image.extent(2) != nfast) {
    image = DvceArray3D<Real>();
    image = DvceArray3D<Real>("projection", nfield, nslow, nfast);
  }
  Kokkos::deep_copy(image, 0.0);
  auto sums = image;
  auto u = mhd ? pack->pmhd->u0 : pack->phydro->u0;
  DvceArray5D<Real> bcc;
  if (mhd) bcc = pack->pmhd->bcc0;

  DualArray2D<int> offsets("projection_offsets", pack->nmb_thispack, 2);
  for (int m=0; m<pack->nmb_thispack; ++m) {
    auto loc = pm->lloc_eachmb[pack->pmb->mb_gid.h_view(m)];
    int logical[3] = {loc.lx1, loc.lx2, loc.lx3};
    offsets.h_view(m,0) = logical[fast_axis]*block_fast;
    offsets.h_view(m,1) = logical[slow_axis]*block_slow;
  }
  offsets.modify_host();
  offsets.sync_device();
  auto pixel_offset = offsets.d_view;
  par_for("projection_sum", DevExeSpace(), 0, pack->nmb_thispack-1,
          0, block_slow-1, 0, block_fast-1,
    KOKKOS_LAMBDA(int m, int slow, int fast) {
      Real column[7] = {};
      for (int line=0; line<nline; ++line) {
        int i = is + (axis == 0 ? line : fast);
        int j = js + (axis == 0 ? fast : (axis == 1 ? line : slow));
        int k = ks + (axis == 2 ? line : slow);
        Real rho = u(m,IDN,k,j,i);
        column[0] += rho*dl;
        for (int c=0; c<3; ++c) {
          column[1+c] += u(m,IM1+c,k,j,i)*dl;
          if (mhd) column[4+c] += rho*bcc(m,c,k,j,i)*dl;
        }
      }
      int global_fast = pixel_offset(m,0) + fast;
      int global_slow = pixel_offset(m,1) + slow;
      for (int f=0; f<nfield; ++f) {
        Kokkos::atomic_add(&sums(f,global_slow,global_fast), column[f]);
      }
    });
  Kokkos::fence();
#if MPI_PARALLEL_ENABLED
  for (std::size_t offset=0; offset<image.size();) {
    int count = static_cast<int>(std::min(image.size()-offset,
                                        static_cast<std::size_t>(INT_MAX)));
    MPI_Allreduce(MPI_IN_PLACE, image.data()+offset, count, MPI_ATHENA_REAL, MPI_SUM,
                  MPI_COMM_WORLD);
    offset += count;
  }
#endif
  if (global_variable::my_rank == 0) {
    par_for("projection_normalize", DevExeSpace(), 1, nfield-1, 0, nslow-1, 0, nfast-1,
      KOKKOS_LAMBDA(int f, int slow, int fast) {
        sums(f,slow,fast) /= sums(0,slow,fast);
      });
    Kokkos::fence();
  }
}

void ProjectionOutput::WriteOutputFile(Mesh *pm, ParameterInput *pin) {
  IOWrapper projfile;
  int nfield = pm->pmb_pack->pmhd == nullptr ? 4 : 7;
  if (global_variable::my_rank == 0) {
    std::ostringstream filename;
    filename << "proj/" << out_params.file_basename << '.'
             << std::setw(5) << std::setfill('0') << out_params.file_number << ".proj";
    projfile.Open(filename.str().c_str(), IOWrapper::FileMode::write, true);
    std::uint16_t endian = 1;
    const char *byte_order = *reinterpret_cast<unsigned char*>(&endian) ? "little" : "big";
    std::ostringstream parameters;
    pin->ParameterDump(parameters);
    std::string parameter_dump = parameters.str();
    std::ostringstream header;
    header << "Athena projection output version=1.0\n"
           << "  size of preheader=11\n" << std::setprecision(17)
           << "  time=" << pm->time << '\n'
           << "  cycle=" << pm->ncycle << '\n'
           << "  output number=" << out_params.file_number << '\n'
           << "  size of variable=" << sizeof(Real) << '\n'
           << "  byte order=" << byte_order << '\n'
           << "  number of axes=3\n"
           << "  root grid size=" << pm->mesh_indcs.nx1 << ' ' << pm->mesh_indcs.nx2
           << ' ' << pm->mesh_indcs.nx3 << '\n'
           << "  domain left edge=" << pm->mesh_size.x1min << ' ' << pm->mesh_size.x2min
           << ' ' << pm->mesh_size.x3min << '\n'
           << "  domain right edge=" << pm->mesh_size.x1max << ' ' << pm->mesh_size.x2max
           << ' ' << pm->mesh_size.x3max << '\n'
           << "  feature flags=" << (nfield == 7 ? 1 : 0) << '\n'
           << "  number of variables=" << nfield << '\n'
           << "  variables: surface_density velocity_mass_weighted_x "
           << "velocity_mass_weighted_y velocity_mass_weighted_z";
    if (nfield == 7) {
      header << " bfield_mass_weighted_x bfield_mass_weighted_y bfield_mass_weighted_z";
    }
    header << "\n  header offset=" << parameter_dump.size() << '\n';
    std::string prefix = header.str();
    projfile.Write_any_type(prefix.data(), prefix.size(), "byte", true);
    projfile.Write_any_type(parameter_dump.data(), parameter_dump.size(), "byte", true);
  }
  for (int axis=0; axis<3; ++axis) {
    CalculateProjection(pm, axis);
    if (global_variable::my_rank == 0) {
      auto host_image = Kokkos::create_mirror_view_and_copy(HostMemSpace(), image);
      std::size_t pixels = image.extent(1)*image.extent(2);
      for (int f=0; f<nfield; ++f) {
        projfile.Write_any_type(host_image.data()+f*pixels, pixels, "Real", true);
      }
    }
  }
  if (global_variable::my_rank == 0) projfile.Close(true);
  out_params.file_number++;
  if (out_params.last_time < 0.0) {
    out_params.last_time = pm->time;
  } else {
    out_params.last_time += out_params.dt;
  }
  pin->SetInteger(out_params.block_name, "file_number", out_params.file_number);
  pin->SetReal(out_params.block_name, "last_time", out_params.last_time);
}
