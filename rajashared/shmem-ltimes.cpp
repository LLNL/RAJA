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
  using PsiView = RAJA::TypedView<double, Layout<3>, IDirection, IGroup, IZone>;

  // phi[moment, group, zone]
  using PhiView = RAJA::TypedView<double, Layout<3>, IMoment, IGroup, IZone>;

  // ell[moment, direction]
  using EllView = RAJA::TypedView<double, Layout<2>, IMoment, IDirection>;



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
      CudaKernel<1024,
        Collapse<RAJA::cuda_block_seq_exec, ArgList<0,2>,
          For<3, RAJA::cuda_block_thread_exec,
            For<1, RAJA::seq_exec, Lambda<0>>
          >
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



void runLTimesCudaShmem(bool debug,
    Index_type num_moments,
                          Index_type num_directions,
                          Index_type num_groups,
                          Index_type num_zones)
{



  // psi[direction, group, zone]
  using PsiView = RAJA::TypedView<double, Layout<3>, IDirection, IGroup, IZone>;

  // phi[moment, group, zone]
  using PhiView = RAJA::TypedView<double, Layout<3>, IMoment, IGroup, IZone>;

  // ell[moment, direction]
  using EllView = RAJA::TypedView<double, Layout<2>, IMoment, IDirection>;



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



  // launch a kernel, with 1 block per SM, and 1024 threads per block

  int cur_device = -1;
  cudaGetDevice(&cur_device);
//  printf("Using CUDA device %d\n", cur_device);

  cudaDeviceProp dev_props;
  cudaGetDeviceProperties(&dev_props, cur_device);
  int num_threads = dev_props.maxThreadsPerBlock;
  int num_sm = dev_props.multiProcessorCount;
//  printf("  threads/block:  %d\n", num_threads);
//  printf("  num SM:         %d\n", num_sm);
//  printf("  shmem/SM:       %ld bytes\n", (long)dev_props.sharedMemPerMultiprocessor);


  num_sm *= 2;

  int num_gz = num_groups * num_zones;

  if(num_gz < num_sm){
    num_sm = num_gz;
  }

  int gz_per_sm = num_gz / num_sm;
  if(gz_per_sm*num_sm < num_gz){
    gz_per_sm ++;
  }

//  printf("Blocks =          %d\n", num_sm);
//  printf("Groups*Zones    = %d\n", num_gz);
//  printf("Groups*Zones/SM = %d\n", gz_per_sm);


//  int GZ_chunk = 12;
  num_threads = 768;
  int GZ_chunk = 8;


  using namespace RAJA::nested;


  // get execution policy
  cudaDeviceSynchronize();
  RAJA::Timer timer;
  timer.start();

  RAJA::SharedMemoryView<RAJA::SharedMemory<RAJA::cuda_shmem, double>,
  RAJA::Layout<2>, RAJA::ident_shmem, RAJA::ident_shmem> ell_shared(num_directions, num_moments);

  RAJA::SharedMemoryView<RAJA::SharedMemory<RAJA::cuda_shmem, double>,
  RAJA::Layout<2>, RAJA::ident_shmem, RAJA::ident_shmem> psi_shared(num_directions, GZ_chunk);

  using LaunchPolicy =
      nested::Policy<
        CudaKernelBase<cuda_explicit_launch<false, 112, 1024>,
          For<0, cuda_block_seq_exec,
            For<1, cuda_thread_exec, Lambda<0>>
          >
        >>;

  nested::forall(
      LaunchPolicy{},

      camp::make_tuple(RangeSegment(0,num_sm), RangeSegment(0,num_threads)),

      [=] __device__ (int block, int thread) {
        // Compute our block's subset of groups and zones


        // Load L into shared memory
        {
          int num_tiles = num_moments*num_directions/num_threads;
          if(num_tiles*num_threads < num_moments*num_directions){
            num_tiles ++;
          }

          for(int t = 0;t < num_tiles;++ t){
            int i = t*num_threads + thread;

            if(i < num_moments*num_directions){
              // compute nm and d
              int d = i / num_moments;
              int m = i % num_moments;
              if(d < num_directions && m < num_moments){
                ell_shared(d, m) = ell(IMoment(m), IDirection(d));
              }
            }
          }
        }
        __syncthreads();



        {
          int gz_start = block * gz_per_sm;
          int gz_end = gz_start + gz_per_sm;

          // Split threads over chunks
          int thread0 = thread / GZ_chunk;
          int chunk0 = thread % GZ_chunk;

          // Loop over each group and zone
          for(int gz = gz_start;gz < gz_end;gz += GZ_chunk){



            int gz0 = gz + chunk0;


            IGroup g(gz0 / num_zones);
            IZone z(gz0 % num_zones);

            if(g < num_groups){

              // use threads to load all directions for psi(g,z) into shmem
              {
                IDirection d(thread0);
                if(d < num_directions){
                  psi_shared(*d, chunk0) = psi(d,g,z);
                }
              }

              __syncthreads();

              // use threads to compute each nm
              {
                IMoment m(thread0);
                if(m < num_moments){
                  double phi_m_g_z = 0.0;

                  for(IDirection d(0);d < num_directions;d++){
                    phi_m_g_z += ell_shared(*d, *m) * psi_shared(*d, chunk0);
                  }

                  // write phi out
                  phi(m,g,z) = phi_m_g_z;
                }
              }

              __syncthreads();

            } // gz bounds test
          } // gz loop

        }

      });


  cudaDeviceSynchronize();
  timer.stop();


  printf("LTimes took %lf seconds using CUDA w/ shmem\n", timer.elapsed());


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
  using PsiView = RAJA::TypedView<double, Layout<3>, IDirection, IGroup, IZone>;

  // phi[moment, group, zone]
  using PhiView = RAJA::TypedView<double, Layout<3>, IMoment, IGroup, IZone>;

  // ell[moment, direction]
  using EllView = RAJA::TypedView<double, Layout<2>, IMoment, IDirection>;



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
        CudaKernel<1024,
          // First, load Ell into shared memory in each block
          Collapse<cuda_thread_exec, ArgList<0, 1>, Lambda<0>>,

          // Distribute groups and zones across blocks
          Collapse<cuda_block_seq_exec, ArgList<2, 3>,

            // Load Psi for this g,z
            For<1, cuda_thread_exec, Lambda<1>>,
            CudaThreadSync,

            // Compute phi for all m's and this g,z
            For<0, cuda_thread_exec, Lambda<2>>

          >
        >
      >;

  RAJA::SharedMemoryView<RAJA::SharedMemory<RAJA::cuda_shmem, double>,
  RAJA::Layout<2>, RAJA::ident_shmem, RAJA::ident_shmem> ell_shared(num_directions, num_moments);

  RAJA::SharedMemoryView<RAJA::SharedMemory<RAJA::cuda_shmem, double>,
  RAJA::Layout<2>, RAJA::ident_shmem, RAJA::ident_shmem> psi_shared(num_directions, 1);

  nested::forall(
      Pol{},

    camp::make_tuple(TypedRangeSegment<IMoment>(0,num_moments),
        TypedRangeSegment<IDirection>(0,num_directions),
        TypedRangeSegment<IGroup>(0,num_groups),
        TypedRangeSegment<IZone>(0,num_zones)),

     // Lambda<0>
     // load L matrix into shmem
     [=] RAJA_DEVICE (IMoment m, IDirection d, IGroup g, IZone z){
       ell_shared(*d, *m) = ell(m, d);
     },

     // Lambda<1>
     // load slice of psi into shared
     [=] RAJA_DEVICE (IMoment m, IDirection d, IGroup g, IZone z){
       psi_shared(*d, 0) = psi(d,g,z);
     },

     // Lambda<2>
     // Compute phi_m_g_z
     [=] RAJA_DEVICE (IMoment m, IDirection d, IGroup g, IZone z){
       double phi_m_g_z = 0.0;
       for(IDirection d(0);d < num_directions; ++ d){
         phi_m_g_z += ell_shared(*d, *m) * psi_shared(*d, 0);
       }
       phi(m, g, z) = phi_m_g_z;
     }


//     // Lambda<3> (must have same thread mapping as Lambda<2>)
//     // Write phi_m_g_z to global
//     [=] RAJA_DEVICE (IMoment m, IDirection d, IGroup g, IZone z){
//       phi(m,g,z)
//     }
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

  bool debug = false;

  int m = 25;
  int d = 80;
  int g = 32;
  int z = 128*1024;

  runLTimesRajaCudaNested(debug, m, d, g, z); // warm up
  runLTimesRajaCudaShmem(debug, m, d, g, z);
  runLTimesRajaCudaNested(debug, m, d, g, z);
  runLTimesCudaShmem(debug, m, d, g, z);


  return 0;
}


