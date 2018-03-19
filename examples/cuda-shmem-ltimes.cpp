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
  std::array<RAJA::idx_t, 2> ell_perm {{0, 1}};
  EllView ell(
      d_ell,
      make_permuted_layout({num_moments, num_directions}, ell_perm));

  std::array<RAJA::idx_t, 3> psi_perm {{0, 1, 2}};
  PsiView psi(
      d_psi,
      make_permuted_layout({num_directions, num_groups, num_zones}, psi_perm));

  std::array<RAJA::idx_t, 3> phi_perm {{0, 1, 2}};
  PhiView phi(
      d_phi,
      make_permuted_layout({num_moments, num_groups, num_zones}, phi_perm));

  // get execution policy
  cudaDeviceSynchronize();
  RAJA::Timer timer;
  timer.start();





  using Pol = RAJA::KernelPolicy<
      CudaKernel<
        For<0, cuda_block_exec,
				  //For<2, cuda_block_exec,
				  For<2, cuda_threadblock_exec<7>,
						For<3, cuda_thread_exec,
							For<1, RAJA::seq_exec, Lambda<0>>
        		>
					>
				>
      >>;


  kernel<Pol>(

      RAJA::make_tuple(TypedRangeSegment<IMoment>(0, num_moments),
          TypedRangeSegment<IDirection>(0, num_directions),
          TypedRangeSegment<IGroup>(0, num_groups),
          TypedRangeSegment<IZone>(0, num_zones)),

      [=] __device__ (IMoment m, IDirection d, IGroup g, IZone z) {
          phi(m, g, z) += ell(m, d) * psi(d, g, z);
					//phi(m,g,z) = 0;
					//printf("mdgz = %d,%d,%d,%d\n", (int)*m, (int)*d, (int)*g, (int)*z);
      });




  cudaDeviceSynchronize();
  timer.stop();
  printf("LTimes took %lf seconds using RAJA::kernel\n",
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
        //for (IMoment m(0); m < num_moments; ++m) {
        for (IMoment m(0); m < 1; ++m) {
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
  using PsiView = RAJA::TypedView<double, Layout<3, Index_type, 0>, IDirection, IGroup, IZone>;

  // phi[moment, group, zone]
  using PhiView = RAJA::TypedView<double, Layout<3, Index_type, 0>, IMoment, IGroup, IZone>;

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

  //printf("ell=%p, phi=%p, psi=%p\n", d_ell, d_phi, d_psi);

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
  std::array<RAJA::idx_t, 2> ell_perm {{0, 1}};
  EllView ell(
      d_ell,
      make_permuted_layout({num_moments, num_directions}, ell_perm));

  std::array<RAJA::idx_t, 3> psi_perm {{2, 1, 0}};
  PsiView psi(
      d_psi,
      make_permuted_layout({num_directions, num_groups, num_zones}, psi_perm));
      //make_permuted_layout({num_zones, num_groups, num_directions}, psi_perm));

  std::array<RAJA::idx_t, 3> phi_perm {{2, 1, 0}};
  PhiView phi(
      d_phi,
      make_permuted_layout({num_moments, num_groups, num_zones}, phi_perm));
      //make_permuted_layout({num_zones, num_groups, num_moments}, phi_perm));




  // get execution policy
  cudaDeviceSynchronize();
  RAJA::Timer timer;
  timer.start();



  static const int tile_mom  = 25;
  static const int tile_dir  = 80;
  static const int tile_zone = 12;
  using namespace RAJA::statement;

    using Pol = RAJA::KernelPolicy<
               CudaKernelAsync<

                RAJA::statement::Tile<1, RAJA::statement::tile_fixed<tile_mom>, seq_exec,
                  RAJA::statement::Tile<2, RAJA::statement::tile_fixed<tile_dir>, seq_exec,

                      // First, load up L matrix for each block
                      SetShmemWindow<
												For<1, cuda_thread_exec, // m
													For<2, cuda_thread_exec, Lambda<2>> //d
												>
                      >,

                      CudaSyncThreads,

                      // Distribute groups and zones across blocks
                      For<0, cuda_block_exec, // g

                      For<3, cuda_threadblock_exec<tile_zone>,  // z

                        SetShmemWindow<

                            // Load Psi for this g,z
                            For<2, cuda_thread_exec, Lambda<3>>, // d

                            CudaSyncThreads,

                            // Compute phi for all m's and this g,z
                            For<1, cuda_thread_exec, // m

                              // Compute phi(m,g,z) using thread-local storage
                              // and shared memory
                              Thread<
                                // Load phi
                                Lambda<4>,

                                // Compute phi
                                For<2, seq_exec, Lambda<5>> ,  // d

                                // Store phi
                                Lambda<6>
                              >
                            >, // m
                            CudaSyncThreads
                          > // shmem
                        >  // z
                      > //g
                    > // tile d
                  > // tile m
                > // kernel
              >; // policy



  auto segments = RAJA::make_tuple(
      TypedRangeSegment<IGroup>(0,num_groups),
      TypedRangeSegment<IMoment>(0,num_moments),
      TypedRangeSegment<IDirection>(0,num_directions),
      TypedRangeSegment<IZone>(0,num_zones));




  using shmem_ell_t = ShmemTile<cuda_shmem, double, ArgList<2,1>, SizeList<tile_dir, tile_mom>, decltype(segments)>;
  shmem_ell_t shmem_ell;

  using shmem_psi_t = ShmemTile<cuda_shmem, double, ArgList<3,2>, SizeList<tile_zone, tile_dir>, decltype(segments)>;
  shmem_psi_t shmem_psi;



  kernel_param<Pol>(

    segments,

		RAJA::make_tuple(
		    shmem_ell,
		    shmem_psi,
		    0.0), // thread private temporary storage (last arg in lambdas)

    // Lambda<0>
     // Zero out phi
    [=] RAJA_HOST_DEVICE  (IGroup g, IMoment nm, IDirection d, IZone z, shmem_ell_t &, shmem_psi_t &, double &){
      phi(nm, g, z) = 0.0;
    },

    // Lambda<1>
    // Original single lambda implementation
    [=] RAJA_HOST_DEVICE  (IGroup g, IMoment nm, IDirection d, IZone z, shmem_ell_t &, shmem_psi_t &, double &){
      phi(nm, g, z) += ell(nm, d) * psi(d,g,z);
    },

    // Lambda<2>
    // load L matrix into shmem
    [=] RAJA_DEVICE  (IGroup g, IMoment nm, IDirection d, IZone z, shmem_ell_t &sh_ell, shmem_psi_t &, double &){
      sh_ell(d, nm) = ell(nm, d);
    },

    // Lambda<3>
    // load slice of psi into shared
    [=] RAJA_DEVICE  (IGroup g, IMoment nm, IDirection d, IZone z, shmem_ell_t &, shmem_psi_t &sh_psi, double &){
      sh_psi(z,d) = psi(d,g,z);
    },

    // Lambda<4>
    // Load phi_m_g_z
    [=] RAJA_DEVICE  (IGroup g, IMoment nm, IDirection d, IZone z, shmem_ell_t &, shmem_psi_t &, double &phi_local){
      phi_local = phi(nm, g, z);
    },

    // Lambda<5>
    // Compute phi_m_g_z
    [=] RAJA_DEVICE  (IGroup g, IMoment nm, IDirection d, IZone z, shmem_ell_t &sh_ell, shmem_psi_t &sh_psi, double &phi_local){
      phi_local += sh_ell(d, nm) * sh_psi(z,d);
    },

    // Lambda<6>
    // Store phi_m_g_z
    [=] RAJA_DEVICE  (IGroup g, IMoment nm, IDirection d, IZone z, shmem_ell_t &, shmem_psi_t &, double &phi_local){
      phi(nm, g, z) = phi_local;
    }

  );


  cudaDeviceSynchronize();
  timer.stop();

  printf("LTimes took %lf seconds using RAJA w/ shmem\n", timer.elapsed());

  // Check correctness
  if(debug){
		printf("Copying back to host\n");
    // Copy to host the result
    cudaMemcpy(&phi_data[0],
               d_phi,
               sizeof(double) * phi_data.size(),
               cudaMemcpyDeviceToHost);
    cudaMemcpy(&psi_data[0],
               d_psi,
               sizeof(double) * psi_data.size(),
               cudaMemcpyDeviceToHost);

    ell.set_data(&ell_data[0]);
    phi.set_data(&phi_data[0]);
    psi.set_data(&psi_data[0]);
    size_t errors = 0;
		printf("Checking result\n");


		for (IMoment m(0); m < num_moments; ++m) {
			for (IGroup g(0); g < num_groups; ++g) {
    		for (IZone z(0); z < num_zones; ++z) {
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
			printf("m=%d (err=%ld)\n", (int)*m, (long)errors);
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

  //bool debug = false;
  bool debug = true;
#if 1
  int m = 25;
  int d = 80;
  int g = 48;
  //int z = 64*1024; //27*50*50;
  //int z = 3150;
	//int z = 17;
  //int z = 31250;
  //int z = 182250;
  int z = 131074;
#else
  int m = 1; 
  int d = 1;
  int g = 1;
  int z = 1; //64*1024+1;
#endif

  printf("Param: m=%d, d=%d, g=%d, z=%d\n", m, d, g, z);

  runLTimesRajaCudaNested(false, m, d, g, z);

  runLTimesRajaCudaShmem(false, m, d, g, z);
  runLTimesRajaCudaShmem(debug, m, d, g, z);
  //runLTimesRajaCudaNested(debug, m, d, g, z);


  return 0;
}


