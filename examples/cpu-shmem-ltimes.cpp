//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-18, Lawrence Livermore National Security, LLC.
// 
// Produced at the Lawrence Livermore National Laboratory
// 
// LLNL-CODE-689114
// 
// All rights reserved.
// 
// This file is part of RAJA.
// 
// For details about use and distribution, please read RAJA/LICENSE.
//
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//


#include <stddef.h>
#include <stdio.h>



#include "RAJA/RAJA.hpp"
#include "RAJA/util/Timer.hpp"


#include <cstdio>
#include <cmath>


using namespace RAJA;


RAJA_INDEX_VALUE(IMoment, "IMoment");
RAJA_INDEX_VALUE(IDirection, "IDirection");
RAJA_INDEX_VALUE(IGroup, "IGroup");
RAJA_INDEX_VALUE(IZone, "IZone");


void runLTimesBare(bool ,
                          Index_type num_moments,
                          Index_type num_directions,
                          Index_type num_groups,
                          Index_type num_zones)
{


  // allocate data
  // phi is initialized to all zeros, the others are randomized
  std::vector<double> ell_data(num_moments * num_directions);
  std::vector<double> psi_data(num_directions * num_groups * num_zones);
  std::vector<double> phi_data(num_moments * num_groups * num_zones);


  // randomize data
  for (size_t i = 0; i < ell_data.size(); ++i) {
    ell_data[i] = i; //drand48();
  }

  for (size_t i = 0; i < psi_data.size(); ++i) {
    psi_data[i] = 2*i; //drand48();
  }

  for (size_t i = 0; i < phi_data.size(); ++i) {
    phi_data[i] = 0; //drand48();
  }


  double * RAJA_RESTRICT d_ell = &ell_data[0];
  double * RAJA_RESTRICT d_phi = &phi_data[0];
  double * RAJA_RESTRICT d_psi = &psi_data[0];



  RAJA::Timer timer;
  timer.start();

  for (int m(0); m < num_moments; ++m) {
    for (int g(0); g < num_groups; ++g) {
      for (int d(0); d < num_directions; ++d) {
        for (int z(0); z < num_zones; ++z) {
          d_phi[m*num_groups*num_zones + g*num_zones + z] +=
              d_ell[m*num_directions + d] *
              d_psi[d*num_groups*num_zones + g*num_zones + z];
        }

      }
    }
  }



  timer.stop();
  printf("LTimes took %lf seconds using bare loops and pointers\n",
      timer.elapsed());

}

void runLTimesBareView(bool debug,
                          Index_type num_moments,
                          Index_type num_directions,
                          Index_type num_groups,
                          Index_type num_zones)
{

  // psi[direction, group, zone]
  using PsiView = RAJA::TypedView<double, Layout<3, Index_type, 2>, IDirection, IGroup, IZone>;

  // phi[moment, group, zone]
  using PhiView = RAJA::TypedView<double, Layout<3, Index_type, 2>, IMoment, IGroup, IZone>;

  // ell[moment, direction]
  using EllView = RAJA::TypedView<double, Layout<2, Index_type, 1>, IMoment, IDirection>;


  // allocate data
  // phi is initialized to all zeros, the others are randomized
  std::vector<double> ell_data(num_moments * num_directions);
  std::vector<double> psi_data(num_directions * num_groups * num_zones);
  std::vector<double> phi_data(num_moments * num_groups * num_zones);


  // randomize data
  for (size_t i = 0; i < ell_data.size(); ++i) {
    ell_data[i] = i; //drand48();
  }

  for (size_t i = 0; i < psi_data.size(); ++i) {
    psi_data[i] = 2*i; //drand48();
  }

  for (size_t i = 0; i < phi_data.size(); ++i) {
    phi_data[i] = 0; //drand48();
  }


  double *d_ell = &ell_data[0];
  double *d_phi = &phi_data[0];
  double *d_psi = &psi_data[0];



  // create views on data
  std::array<RAJA::idx_t, 2> ell_perm {{0, 1}};
  EllView ell(
      d_ell,
      make_permuted_layout({{num_moments, num_directions}}, ell_perm));

  std::array<RAJA::idx_t, 3> psi_perm {{0, 1, 2}};
  PsiView psi(
      d_psi,
      make_permuted_layout({{num_directions, num_groups, num_zones}}, psi_perm));

  std::array<RAJA::idx_t, 3> phi_perm {{0, 1, 2}};
  PhiView phi(
      d_phi,
      make_permuted_layout({{num_moments, num_groups, num_zones}}, phi_perm));



  RAJA::Timer timer;
  timer.start();

  for (IMoment m(0); m < num_moments; ++m) {
    for (IGroup g(0); g < num_groups; ++g) {
      for (IDirection d(0); d < num_directions; ++d) {
        for (IZone z(0); z < num_zones; ++z) {

          phi(m, g, z) += ell(m, d) * psi(d, g, z);

        }

      }
    }
  }



  timer.stop();
  printf("LTimes took %lf seconds using bare loops and RAJA::Views\n",
      timer.elapsed());



  // Check correctness
  if(debug){

    size_t errors = 0;
    double total_error = 0.;
    for (IZone z(0); z < num_zones; ++z) {
      for (IGroup g(0); g < num_groups; ++g) {
        for (IMoment m(0); m < num_moments; ++m) {
          double total = 0.0;
          for (IDirection d(0); d < num_directions; ++d) {
            double val = ell(m, d) * psi(d, g, z);
            total += val;
          }
          if(std::abs(total-phi(m,g,z)) > 1e-9){
            ++ errors;
          }
          total_error += std::abs(total-phi(m,g,z));
        }
      }
    }
    if(errors == 0){
      printf("  -- no errors (%e)\n", total_error);
    }
    else{
      printf("  -- failed : %ld errors\n", (long)errors);
    }
  }

}




void runLTimesRajaKernel(bool debug,
                          Index_type num_moments,
                          Index_type num_directions,
                          Index_type num_groups,
                          Index_type num_zones)
{

	using namespace RAJA::statement;

  // psi[direction, group, zone]
  using PsiView = RAJA::TypedView<double, Layout<3, Index_type, 2>, IDirection, IGroup, IZone>;

  // phi[moment, group, zone]
  using PhiView = RAJA::TypedView<double, Layout<3, Index_type, 2>, IMoment, IGroup, IZone>;

  // ell[moment, direction]
  using EllView = RAJA::TypedView<double, Layout<2, Index_type, 1>, IMoment, IDirection>;


  // allocate data
  // phi is initialized to all zeros, the others are randomized
  std::vector<double> ell_data(num_moments * num_directions);
  std::vector<double> psi_data(num_directions * num_groups * num_zones);
  std::vector<double> phi_data(num_moments * num_groups * num_zones);


  // randomize data
  for (size_t i = 0; i < ell_data.size(); ++i) {
    ell_data[i] = i; //drand48();
  }

  for (size_t i = 0; i < psi_data.size(); ++i) {
    psi_data[i] = 2*i; //drand48();
  }

  for (size_t i = 0; i < phi_data.size(); ++i) {
    phi_data[i] = 0; //drand48();
  }


  double *d_ell = &ell_data[0];
  double *d_phi = &phi_data[0];
  double *d_psi = &psi_data[0];



  // create views on data
  std::array<RAJA::idx_t, 2> ell_perm {{0, 1}};
  EllView ell(
      d_ell,
      make_permuted_layout({{num_moments, num_directions}}, ell_perm));

  std::array<RAJA::idx_t, 3> psi_perm {{0, 1, 2}};
  PsiView psi(
      d_psi,
      make_permuted_layout({{num_directions, num_groups, num_zones}}, psi_perm));

  std::array<RAJA::idx_t, 3> phi_perm {{0, 1, 2}};
  PhiView phi(
      d_phi,
      make_permuted_layout({{num_moments, num_groups, num_zones}}, phi_perm));



  using Pol = RAJA::KernelPolicy<
    For<0, loop_exec,
      For<1, loop_exec,
        For<2, loop_exec,
          For<3, simd_exec, Lambda<0>>
        >
      >
    >>;


  RAJA::Timer timer;
  timer.start();

  auto segments =  RAJA::make_tuple(TypedRangeSegment<IMoment>(0, num_moments),
      TypedRangeSegment<IDirection>(0, num_directions),
      TypedRangeSegment<IGroup>(0, num_groups),
      TypedRangeSegment<IZone>(0, num_zones));


  kernel<Pol>(

      segments,

      // Lambda_CalcPhi
      [=] (IMoment m, IDirection d, IGroup g, IZone z) {
        phi(m, g, z) += ell(m, d) * psi(d, g, z);
      });



  timer.stop();
  printf("LTimes took %lf seconds using RAJA::kernel\n",
      timer.elapsed());


  // Check correctness
  if(debug){

    size_t errors = 0;
    double total_error = 0.;
    for (IZone z(0); z < num_zones; ++z) {
      for (IGroup g(0); g < num_groups; ++g) {
        for (IMoment m(0); m < num_moments; ++m) {
          double total = 0.0;
          for (IDirection d(0); d < num_directions; ++d) {
            double val = ell(m, d) * psi(d, g, z);
            total += val;
          }
          if(std::abs(total-phi(m,g,z)) > 1e-9){
            ++ errors;
          }
          total_error += std::abs(total-phi(m,g,z));
        }
      }
    }
    if(errors == 0){
      printf("  -- no errors (%e)\n", total_error);
    }
    else{
      printf("  -- failed : %ld errors\n", (long)errors);
    }
  }

}


void runLTimesRajaKernelShmem(bool debug,
                          Index_type num_moments,
                          Index_type num_directions,
                          Index_type num_groups,
                          Index_type num_zones)
{

	using namespace RAJA::statement;
  
	// psi[direction, group, zone]
  using PsiView = RAJA::TypedView<double, Layout<3, Index_type, 2>, IDirection, IGroup, IZone>;

  // phi[moment, group, zone]
  using PhiView = RAJA::TypedView<double, Layout<3, Index_type, 2>, IMoment, IGroup, IZone>;

  // ell[moment, direction]
  using EllView = RAJA::TypedView<double, Layout<2, Index_type, 1>, IMoment, IDirection>;



  // allocate data
  // phi is initialized to all zeros, the others are randomized
  std::vector<double> ell_data(num_moments * num_directions);
  std::vector<double> psi_data(num_directions * num_groups * num_zones);
  std::vector<double> phi_data(num_moments * num_groups * num_zones, 0.0);


  // randomize data
  for (size_t i = 0; i < ell_data.size(); ++i) {
    ell_data[i] = i; //drand48();
  }

  for (size_t i = 0; i < psi_data.size(); ++i) {
    psi_data[i] = 2*i; //drand48();
  }

  for (size_t i = 0; i < phi_data.size(); ++i) {
    phi_data[i] = 0; //drand48();
  }


  double *d_ell = &ell_data[0];
  double *d_phi = &phi_data[0];
  double *d_psi = &psi_data[0];



  // create views on data
  std::array<RAJA::idx_t, 2> ell_perm {{0, 1}};
  EllView ell(
      d_ell,
      make_permuted_layout({{num_moments, num_directions}}, ell_perm));

  std::array<RAJA::idx_t, 3> psi_perm {{0, 1, 2}};
  PsiView psi(
      d_psi,
      make_permuted_layout({{num_directions, num_groups, num_zones}}, psi_perm));

  std::array<RAJA::idx_t, 3> phi_perm {{0, 1, 2}};
  PhiView phi(
      d_phi,
      make_permuted_layout({{num_moments, num_groups, num_zones}}, phi_perm));


  constexpr size_t tile_moments = 25;
  constexpr size_t tile_directions = 80;
  constexpr size_t tile_zones = 256;
  constexpr size_t tile_groups = 0;

  using Lambda_LoadEll = Lambda<0>;
  using Lambda_LoadPsi = Lambda<1>;
  using Lambda_LoadPhi = Lambda<2>;
  using Lambda_CalcPhi = Lambda<3>;
  using Lambda_SavePhi = Lambda<4>;

  using Pol = RAJA::KernelPolicy<
    statement::Tile<0, statement::tile_fixed<tile_moments>, loop_exec,
      statement::Tile<1, statement::tile_fixed<tile_directions>, loop_exec,
        SetShmemWindow<

          // Load shmem L
          For<0, simd_exec, For<1, simd_exec, Lambda_LoadEll>>,

          For<2, loop_exec,
            statement::Tile<3, statement::tile_fixed<tile_zones>, loop_exec,
              SetShmemWindow<
                // Load Psi into shmem
                For<1, simd_exec, For<3, simd_exec, Lambda_LoadPsi >>,

                For<0, loop_exec, //m
                  For<3, simd_exec, Lambda_LoadPhi>, //z

                  For<1, loop_exec, // d
                    For<3, simd_exec, Lambda_CalcPhi>
                  >,

                  For<3, simd_exec, Lambda_SavePhi>
                >  // m
              > // Shmem Window
            > // Tile zones
          > // for g
        > // Shmem Window (mom, dir)
      > // Tile directions
    > // Tile moments
  >; // Pol



  RAJA::Timer timer;
  timer.start();

  auto segments =  RAJA::make_tuple(TypedRangeSegment<IMoment>(0, num_moments),
      TypedRangeSegment<IDirection>(0, num_directions),
      TypedRangeSegment<IGroup>(0, num_groups),
      TypedRangeSegment<IZone>(0, num_zones));


  using shmem_ell_t = ShmemTile<seq_shmem, double, ArgList<0,1>, SizeList<tile_moments, tile_directions>, decltype(segments)>;
  using shmem_psi_t = ShmemTile<seq_shmem, double, ArgList<1,2,3>, SizeList<tile_directions, tile_groups, tile_zones>, decltype(segments)>;
  using shmem_phi_t = ShmemTile<seq_shmem, double, ArgList<0,2,3>, SizeList<tile_moments, tile_groups, tile_zones>, decltype(segments)>;

  shmem_ell_t shmem_ell;
  shmem_psi_t shmem_psi;
  shmem_phi_t shmem_phi;

  kernel_param<Pol>(

      segments,

      RAJA::make_tuple(shmem_ell, shmem_psi, shmem_phi),

      // Lambda_LoadEll
      [=] (IMoment m, IDirection d, IGroup, IZone, shmem_ell_t&sh_ell, shmem_psi_t&, shmem_phi_t&) {
        sh_ell(m, d) = ell(m, d);
      },

      // Lambda_LoadPsi
      [=] (IMoment, IDirection d, IGroup g, IZone z, shmem_ell_t&, shmem_psi_t&sh_psi, shmem_phi_t&) {
        sh_psi(d, g, z) = psi(d, g, z);
      },

      // Lambda_LoadPhi
      [=] (IMoment m, IDirection, IGroup g, IZone z, shmem_ell_t&, shmem_psi_t&, shmem_phi_t&sh_phi) {
        sh_phi(m, g, z) = phi(m,g,z);
      },

      // Lambda_CalcPhi
      [=] (IMoment m, IDirection d, IGroup g, IZone z, shmem_ell_t&sh_ell, shmem_psi_t&sh_psi, shmem_phi_t&sh_phi) {
        sh_phi(m, g, z) += sh_ell(m, d) * sh_psi(d, g, z);
      },

      // Lambda_SavePhi
      [=] (IMoment m, IDirection, IGroup g, IZone z, shmem_ell_t&, shmem_psi_t&, shmem_phi_t&sh_phi) {
        phi(m,g,z) = sh_phi(m, g, z);
      });



  timer.stop();
  printf("LTimes took %lf seconds using RAJA::kernel and shmem\n",
      timer.elapsed());


  // Check correctness
  if(debug){

    size_t errors = 0;
    double total_error = 0.;
    for (IZone z(0); z < num_zones; ++z) {
      for (IGroup g(0); g < num_groups; ++g) {
        for (IMoment m(0); m < num_moments; ++m) {
          double total = 0.0;
          for (IDirection d(0); d < num_directions; ++d) {
            double val = ell(m, d) * psi(d, g, z);
            total += val;
          }
          if(std::abs(total-phi(m,g,z)) > 1e-9){
            ++ errors;
          }
          total_error += std::abs(total-phi(m,g,z));
        }
      }
    }
    if(errors == 0){
      printf("  -- no errors (%e)\n", total_error);
    }
    else{
      printf("  -- failed : %ld errors\n", (long)errors);
    }
  }

}



int main(){

  bool debug = true;

  int m = 25;
  int d = 80;
  int g = 32;
  int z = 32*1024;

  printf("m=%d, d=%d, g=%d, z=%d\n", m, d, g, z);

  runLTimesBare(debug, m, d, g, z);
  runLTimesBareView(debug, m, d, g, z);
  runLTimesRajaKernel(debug, m, d, g, z);
  runLTimesRajaKernelShmem(debug, m, d, g, z);



  return 0;
}


