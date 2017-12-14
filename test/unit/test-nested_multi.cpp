#include "RAJA/RAJA.hpp"
#include "RAJA_gtest.hpp"

#include "RAJA/util/Timer.hpp"

#include <cstdio>

#if defined(RAJA_ENABLE_CUDA)
#include <cuda_runtime.h>
#endif


using namespace RAJA;


#if defined(RAJA_ENABLE_OPENMP)
TEST(NestedMulti, SharedMemoryTest_OpenMP)
{

  using polI =
      RAJA::nested::Policy<
        RAJA::nested::For<0, RAJA::omp_for_nowait_exec>
      >;

  using polIJ =
      RAJA::nested::Policy<
        RAJA::nested::For<0, RAJA::omp_for_nowait_exec>,
        RAJA::nested::For<1, RAJA::loop_exec>
      >;

  RAJA::SharedMemory<RAJA::omp_shmem, double> s(4);
  RAJA::SharedMemory<RAJA::omp_shmem, double> t(16);

  double *output = new double[4];

  RAJA::nested::forall_multi<omp_multi_exec<true>>(

      // Zero out s[]
      RAJA::nested::makeLoop(
        polI{},
        camp::make_tuple(RAJA::RangeSegment(0,4)),
        [=] (int i){
          s[i] = 0;
        }),

      // Initialize t[]
      RAJA::nested::makeLoop(
        polIJ{},
        camp::make_tuple(RAJA::RangeSegment(0,4),
                         RAJA::RangeSegment(0,4)),
        [=] (int i, int j){
          t[i + 4*j] = i*j;
        }),

      // Compute s[] from t[]
      RAJA::nested::makeLoop(
        polI{},
        camp::make_tuple(RAJA::RangeSegment(0,4)),
        [=] (int i){
          for(int k = 0;k < 4;++ k){
            s[i] += t[i + 4*k];
          }
        }),

      // save output
      RAJA::nested::makeLoop(
        polI{},
        camp::make_tuple(RAJA::RangeSegment(0,4)),
        [=] (int i){
          output[i] = s[i];
        })

   );

  ASSERT_EQ(output[0], 0);
  ASSERT_EQ(output[1], 6);
  ASSERT_EQ(output[2], 12);
  ASSERT_EQ(output[3], 18);


  delete[] output;
}
#endif // RAJA_ENABLE_OPENMP


#if defined(RAJA_ENABLE_CUDA)
CUDA_TEST(NestedMulti, SharedMemoryTest_CUDA)
{

  using polI = RAJA::nested::Policy<
      RAJA::nested::CudaCollapse<
        RAJA::nested::For<0, RAJA::cuda_thread_x_exec>
      > >;

  using polIJ = RAJA::nested::Policy<
      RAJA::nested::CudaCollapse<
        RAJA::nested::For<0, RAJA::cuda_thread_x_exec>,
        RAJA::nested::For<1, RAJA::cuda_thread_y_exec>
      > >;

  RAJA::SharedMemory<RAJA::cuda_shmem, double> s(4);
  RAJA::SharedMemory<RAJA::cuda_shmem, double> t(16);


  double *output = nullptr;
  cudaErrchk(cudaMallocManaged(&output, sizeof(double) * 4) );

  cudaDeviceSynchronize();

  RAJA::nested::forall_multi<RAJA::nested::cuda_multi_exec<false>>(

      // Zero out s[]
      RAJA::nested::makeLoop(
        polI{},
        camp::make_tuple(RAJA::RangeSegment(0,4)),
        [=] __device__ (int i){
          s[i] = 0;
        }),

      // Initialize t[]
      RAJA::nested::makeLoop(
        polIJ{},
        camp::make_tuple(RAJA::RangeSegment(0,4),
                         RAJA::RangeSegment(0,4)),
        [=] __device__ (int i, int j){
          t[i + 4*j] = i*j;
        }),

      // Compute s[] from t[]
      RAJA::nested::makeLoop(
        polI{},
        camp::make_tuple(RAJA::RangeSegment(0,4)),
        [=] __device__ (int i){
          for(int k = 0;k < 4;++ k){
            s[i] += t[i + 4*k];
          }
        }),

      // save output
      RAJA::nested::makeLoop(
        polI{},
        camp::make_tuple(RAJA::RangeSegment(0,4)),
        [=] __device__ (int i){
          output[i] = s[i];
        })

  );


  cudaDeviceSynchronize();

  ASSERT_EQ(output[0], 0);
  ASSERT_EQ(output[1], 6);
  ASSERT_EQ(output[2], 12);
  ASSERT_EQ(output[3], 18);


  cudaFree(output);
}
#endif // RAJA_ENABLE_CUDA


#if defined(RAJA_ENABLE_CUDA)
CUDA_TEST(NestedMulti, SharedMemoryViewTest_CUDA)
{


  using polI = RAJA::nested::Policy<
      RAJA::nested::CudaCollapse<
        RAJA::nested::For<0, RAJA::cuda_threadblock_x_exec<4>>
      > >;

  using polIJ = RAJA::nested::Policy<
      RAJA::nested::CudaCollapse<
        RAJA::nested::For<0, RAJA::cuda_threadblock_x_exec<4>>,
        RAJA::nested::For<1, RAJA::cuda_thread_y_exec>
      > >;


  size_t N = 128;


  RAJA::CudaSharedMemoryBlocked<double> s(4);


  RAJA::SharedMemoryView<RAJA::SharedMemory<RAJA::cuda_shmem, double>,
    RAJA::Layout<2>, RAJA::block_map_x_shmem, RAJA::ident_shmem> t(4, 4);



  double *output = nullptr;
  cudaErrchk(cudaMallocManaged(&output, sizeof(double) * N) );

  cudaDeviceSynchronize();

  RAJA::nested::forall_multi<RAJA::nested::cuda_multi_exec<false>>(

      // Zero out s[]
      RAJA::nested::makeLoop(
        polI{},
        camp::make_tuple(RAJA::RangeSegment(0,N)),
        [=] __device__ (int i){
          s(i) = 0;
        }),

      // Initialize t[]
      RAJA::nested::makeLoop(
        polIJ{},
        camp::make_tuple(RAJA::RangeSegment(0,N),
                         RAJA::RangeSegment(0,4)),
        [=] __device__ (int i, int j){
          t(i,j) = i*j;
        }),

      // Compute s[] from t[]
      RAJA::nested::makeLoop(
        polI{},
        camp::make_tuple(RAJA::RangeSegment(0,N)),
        [=] __device__ (int i){
          for(int k = 0;k < 4;++ k){
            s(i) += t(i,k);
          }
        }),

      // save output
      RAJA::nested::makeLoop(
        polI{},
        camp::make_tuple(RAJA::RangeSegment(0,N)),
        [=] __device__ (int i){
          output[i] = s(i);
        })

  );


  cudaErrchk(cudaDeviceSynchronize());

	for(size_t i = 0;i < N;++ i){
		ASSERT_EQ(output[i], (double)i*6);
	}


  cudaFree(output);
}
#endif // RAJA_ENABLE_CUDA





#if defined(RAJA_ENABLE_CUDA)

RAJA_INDEX_VALUE(IMoment, "IMoment");
RAJA_INDEX_VALUE(IDirection, "IDirection");
RAJA_INDEX_VALUE(IGroup, "IGroup");
RAJA_INDEX_VALUE(IZone, "IZone");


static void runLTimesCudaShmem(Index_type num_moments,
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

  // do calculation using RAJA

  /*
   * Blocks in Z for each Moment
   * Blocks in Y for each Group
   * Threads in X for each direction
   *
   * Shmem stores Ell for all directions and one moment (defined by the block)
   */
  constexpr size_t ZoneThreads = 75;
  using Pol_LoadEll = RAJA::nested::Policy<
        RAJA::nested::CudaCollapse<
          RAJA::nested::TypedFor<0, RAJA::cuda_block_z_exec, IGroup>,
          RAJA::nested::TypedFor<1, RAJA::cuda_threadblock_x_exec<ZoneThreads>, IZone> >
        >;

  using Pol_Calc = RAJA::nested::Policy<
      RAJA::nested::CudaCollapse<
        RAJA::nested::TypedFor<0, RAJA::cuda_block_z_exec, IGroup>,
        RAJA::nested::TypedFor<1, RAJA::cuda_threadblock_x_exec<ZoneThreads>, IZone> >
      >;

  using Pol_SavePhi = RAJA::nested::Policy<
      RAJA::nested::CudaCollapse<
        RAJA::nested::TypedFor<0, RAJA::cuda_block_z_exec, IMoment>,
        RAJA::nested::TypedFor<1, RAJA::cuda_block_y_exec, IGroup>,
        RAJA::nested::TypedFor<2, RAJA::cuda_threadblock_x_exec<ZoneThreads>, IZone> >
      >;

  //SharedMemory<cuda_shmem, double> ell_shared(num_directions);

  RAJA::SharedMemoryView<RAJA::SharedMemory<RAJA::cuda_shmem, double>,
  RAJA::Layout<2>, RAJA::ident_shmem, RAJA::ident_shmem> ell_shared(num_directions, num_moments);

  RAJA::SharedMemoryView<RAJA::SharedMemory<RAJA::cuda_shmem, double>,
  RAJA::Layout<2>, RAJA::ident_shmem, RAJA::block_map_x_shmem> phi_shared(num_moments, ZoneThreads);


  nested::forall_multi<nested::cuda_multi_exec<true>>(

      nested::makeLoop(
        Pol_LoadEll{},

        camp::make_tuple(RangeSegment(0, num_groups),
                         RangeSegment(0, num_zones)),

        [=] __device__ (IGroup g, IZone z){
          IMoment m(threadIdx.x);
          if(m < num_moments){
            for(IDirection d(0);d < num_directions;++ d){
              ell_shared(*d, *m) = ell(m, d);
            }
          }
        }),

      nested::makeLoop(
        Pol_Calc{},

        camp::make_tuple(RangeSegment(0, num_groups),
                         RangeSegment(0, num_zones)),

        [=] __device__ (IGroup g, IZone z){
          for(IMoment m(0);m < num_moments;++ m){
            phi_shared(*m, *z) = 0.0;
          }

          for(IDirection d(0);d < num_directions;++ d){
            double psi_g_d_z = psi(d,g,z);
            for(IMoment m(0);m < num_moments;++ m){
              phi_shared(*m, *z) += psi_g_d_z * ell_shared(*d, *m);
            }
          }
          for(IMoment m(0);m < num_moments;++ m){
            phi(m, g, z) = phi_shared(*m, *z);
          }
        })


  ); // forall_multi





  cudaDeviceSynchronize();
  timer.stop();
  printf("LTimes took %lf seconds\n", timer.elapsed());
  // Copy to host the result
  cudaMemcpy(&phi_data[0],
             d_phi,
             sizeof(double) * phi_data.size(),
             cudaMemcpyDeviceToHost);

  // Free CUDA memory
  cudaFree(d_ell);
  cudaFree(d_phi);
  cudaFree(d_psi);

  // swap to host pointers
  ell.set_data(&ell_data[0]);
  phi.set_data(&phi_data[0]);
  psi.set_data(&psi_data[0]);
  for (IZone z(0); z < num_zones; ++z) {
    for (IGroup g(0); g < num_groups; ++g) {
      for (IMoment m(0); m < num_moments; ++m) {
        double total = 0.0;
        for (IDirection d(0); d < num_directions; ++d) {
          double val = ell(m, d) * psi(d, g, z);
          total += val;
        }
        ASSERT_FLOAT_EQ(total, phi(m, g, z));
      }
    }
  }

}




CUDA_TEST(TestNestedMulti, LTimesCudaShmem){
  runLTimesCudaShmem(25, 160, 64, 2*1024);
}



#endif // RAJA_ENABLE_CUDA


