#include "RAJA/RAJA.hpp"
#include "RAJA/util/Timer.hpp"

#ifndef RAJA_ENABLE_CUDA
#error This example program requires CUDA
#endif

#include <cstdio>

#include <cuda_runtime.h>



using namespace RAJA;


RAJA_INDEX_VALUE(IMoment, "IMoment");
RAJA_INDEX_VALUE(IDirection, "IDirection");
RAJA_INDEX_VALUE(IGroup, "IGroup");
RAJA_INDEX_VALUE(IZone, "IZone");



void runLTimesRajaCudaNested(bool debug,
    Index_type num_moments,
                          Index_type num_directions,
                          Index_type num_groups,
                          Index_type num_zones)
{

  using namespace RAJA::nested;

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
    ell_data[i] = drand48();
  }

  for (size_t i = 0; i < psi_data.size(); ++i) {
    psi_data[i] = drand48();
  }


  // create device memory
  double *d_ell, *d_phi, *d_psi;
  cudaErrchk(cudaMalloc(&d_ell, sizeof(double) * ell_data.size()));
  cudaErrchk(cudaMalloc(&d_phi, sizeof(double) * phi_data.size()));
  cudaErrchk(cudaMalloc(&d_psi, sizeof(double) * psi_data.size()));

  // Copy to device
  cudaMemcpy(d_ell,
             &ell_data[0],
             sizeof(double) * ell_data.size(),
             cudaMemcpyHostToDevice);
  cudaMemcpy(d_phi,
             &phi_data[0],
             sizeof(double) * phi_data.size(),
             cudaMemcpyHostToDevice);
  cudaMemcpy(d_psi,
             &psi_data[0],
             sizeof(double) * psi_data.size(),
             cudaMemcpyHostToDevice);



  // create views on data
  std::array<camp::idx_t, 2> ell_perm {{0, 1}};
  EllView ell(
      d_ell,
      make_permuted_layout({num_moments, num_directions}, ell_perm));

  std::array<camp::idx_t, 3> psi_perm {{0, 1, 2}};
  PsiView psi(
      d_psi,
      make_permuted_layout({num_directions, num_groups, num_zones}, psi_perm));

  std::array<camp::idx_t, 3> phi_perm {{0, 1, 2}};
  PhiView phi(
      d_phi,
      make_permuted_layout({num_moments, num_groups, num_zones}, phi_perm));

  // get execution policy
  cudaDeviceSynchronize();
  RAJA::Timer timer;
  timer.start();





  using Pol = RAJA::nested::Policy<
      CudaKernel<
        Collapse<RAJA::cuda_block_thread_exec, ArgList<0,2,3>,
          For<1, RAJA::seq_exec, Lambda<0>>
        >
      >>;


  nested::forall(
      Pol{},

      camp::make_tuple(TypedRangeSegment<IMoment>(0, num_moments),
          TypedRangeSegment<IDirection>(0, num_directions),
          TypedRangeSegment<IGroup>(0, num_groups),
          TypedRangeSegment<IZone>(0, num_zones)),

      [=] __device__ (IMoment m, IDirection d, IGroup g, IZone z) {
          phi(m, g, z) += ell(m, d) * psi(d, g, z);
      });




  cudaDeviceSynchronize();
  timer.stop();
  printf("LTimes took %lf seconds using RAJA::nested::forall\n",
      timer.elapsed());


  // Check correctness
  if(debug){

    // Copy to host the result
    cudaMemcpy(&phi_data[0],
               d_phi,
               sizeof(double) * phi_data.size(),
               cudaMemcpyDeviceToHost);

    ell.set_data(&ell_data[0]);
    phi.set_data(&phi_data[0]);
    psi.set_data(&psi_data[0]);
    size_t errors = 0;
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
        }
      }
    }
    if(errors == 0){
      printf("  -- no errors\n");
    }
    else{
      printf("  -- failed : %ld errors\n", (long)errors);
    }
  }


  // Free CUDA memory
  cudaFree(d_ell);
  cudaFree(d_phi);
  cudaFree(d_psi);
}



void runLTimesRajaCudaShmem(bool debug,
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
  std::vector<double> phi_data(num_moments * num_groups * num_zones, 0.0);


  // randomize data
  for (size_t i = 0; i < ell_data.size(); ++i) {
    ell_data[i] = drand48();
  }

  for (size_t i = 0; i < psi_data.size(); ++i) {
    psi_data[i] = drand48();
  }


  // create device memory
  double *d_ell, *d_phi, *d_psi;
  cudaErrchk(cudaMalloc(&d_ell, sizeof(double) * ell_data.size()));
  cudaErrchk(cudaMalloc(&d_phi, sizeof(double) * phi_data.size()));
  cudaErrchk(cudaMalloc(&d_psi, sizeof(double) * psi_data.size()));

  // Copy to device
  cudaMemcpy(d_ell,
             &ell_data[0],
             sizeof(double) * ell_data.size(),
             cudaMemcpyHostToDevice);
  cudaMemcpy(d_phi,
             &phi_data[0],
             sizeof(double) * phi_data.size(),
             cudaMemcpyHostToDevice);
  cudaMemcpy(d_psi,
             &psi_data[0],
             sizeof(double) * psi_data.size(),
             cudaMemcpyHostToDevice);



  // create views on data
  std::array<camp::idx_t, 2> ell_perm {{0, 1}};
  EllView ell(
      d_ell,
      make_permuted_layout({num_moments, num_directions}, ell_perm));

  std::array<camp::idx_t, 3> psi_perm {{0, 1, 2}};
  PsiView psi(
      d_psi,
      make_permuted_layout({num_directions, num_groups, num_zones}, psi_perm));

  std::array<camp::idx_t, 3> phi_perm {{0, 1, 2}};
  PhiView phi(
      d_phi,
      make_permuted_layout({num_moments, num_groups, num_zones}, phi_perm));





  // get execution policy
  cudaDeviceSynchronize();
  RAJA::Timer timer;
  timer.start();



  // A possible implementation:
  using namespace RAJA::nested;
  using Pol = nested::Policy<
        CudaKernel<
          SetShmemWindow<
            // First, load Ell into shared memory in each block
            Collapse<cuda_thread_exec, ArgList<0, 1>, Lambda<0>>,
            CudaThreadSync,


            // Distribute groups and zones across blocks
            Collapse<cuda_block_seq_exec, ArgList<2, 3>,

              // Load Psi for this g,z
              For<1, cuda_thread_exec, Lambda<1>>,
              CudaThreadSync,


              // Compute phi for all m's and this g,z
              For<0, cuda_thread_exec, Lambda<2>>,
              CudaThreadSync
            >
          >
        >
      >;


  auto segments = camp::make_tuple(TypedRangeSegment<IMoment>(0,num_moments),
                                    TypedRangeSegment<IDirection>(0,num_directions),
                                    TypedRangeSegment<IGroup>(0,num_groups),
                                    TypedRangeSegment<IZone>(0,num_zones));


  using shmem_ell_t = SharedMemory<cuda_shmem, double, 25*80>;
  ShmemWindowView<shmem_ell_t, ArgList<1,0>, SizeList<80, 25>, decltype(segments)> shmem_ell;

  using shmem_psi_t = SharedMemory<cuda_shmem, double, 80>;
  ShmemWindowView<shmem_psi_t, ArgList<1>, SizeList<80>, decltype(segments)> shmem_psi;


  nested::forall(
      Pol{},

      segments,

     // Lambda<0>
     // load L matrix into shmem
     [=] RAJA_DEVICE (IMoment m, IDirection d, IGroup g, IZone z){
        shmem_ell(d, m) = ell(m, d);
     },

     // Lambda<1>
     // load slice of psi into shared
     [=] RAJA_DEVICE (IMoment m, IDirection d, IGroup g, IZone z){
        shmem_psi(d) = psi(d,g,z);
     },

     // Lambda<2>
     // Compute phi_m_g_z
     [=] RAJA_DEVICE (IMoment m, IDirection d, IGroup g, IZone z){

       double phi_m_g_z = phi(m, g, z);
       for(IDirection d(0);d < num_directions; ++ d){
         phi_m_g_z += shmem_ell(d, m) * shmem_psi(d);
       }
       phi(m, g, z) = phi_m_g_z;

     }

  );



  cudaDeviceSynchronize();
  timer.stop();


  printf("LTimes took %lf seconds using RAJA w/ shmem\n", timer.elapsed());


  // Check correctness
  if(debug){

    // Copy to host the result
    cudaMemcpy(&phi_data[0],
               d_phi,
               sizeof(double) * phi_data.size(),
               cudaMemcpyDeviceToHost);

    ell.set_data(&ell_data[0]);
    phi.set_data(&phi_data[0]);
    psi.set_data(&psi_data[0]);
    size_t errors = 0;
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
        }
      }
    }
    if(errors == 0){
      printf("  -- no errors\n");
    }
    else{
      printf("  -- failed : %ld errors\n", (long)errors);
    }
  }


  // Free CUDA memory
  cudaFree(d_ell);
  cudaFree(d_phi);
  cudaFree(d_psi);
}




int main(){

  bool debug = true;
#if 1
  int m = 25;
  int d = 80;
  int g = 32;
  int z = 2*1024;
#else
  int m = 4;
  int d = 4;
  int g = 4;
  int z = 4;
#endif

  runLTimesRajaCudaNested(debug, m, d, g, z); // warm up

  runLTimesRajaCudaShmem(debug, m, d, g, z);
  runLTimesRajaCudaNested(debug, m, d, g, z);


  return 0;
}


