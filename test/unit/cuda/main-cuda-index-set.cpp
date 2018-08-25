/*
 * Copyright (c) 2016, Lawrence Livermore National Security, LLC.
 *
 * Produced at the Lawrence Livermore National Laboratory.
 *
 * All rights reserved.
 *
 * For release details and restrictions, please see raja/README-license.txt
 */

//
// Main program illustrating simple RAJA dependence graph creation
// and execution and methods.
//

#include <time.h>
#include <cstdlib>

#include <iostream>
#include <string>
#include <vector>

#include "RAJA/RAJA.hpp"
#include "RAJA/internal/RAJAVec.hpp"
#include "RAJA/internal/MemUtils_GPU.hpp"

using namespace RAJA;
using namespace std;

#if defined(RAJA_ENABLE_CUDA)
static RAJA::Real_ptr ref_array, test_array;
#endif

///////////////////////////////////////////////////////////////////////////
//
// Main Program.
//
///////////////////////////////////////////////////////////////////////////

int main(int RAJA_UNUSED_ARG(argc), char** RAJA_UNUSED_ARG(argv[]))
{

  std::cout<<"Test for IndexSet on GPU"<<std::endl;
  RAJA::IndexSet is1;

  const int num_segments = 32;
  for (int i=0; i<num_segments; i++) {
    is1.push_back(RAJA::RangeSegment(i*2,i*2+2));
  }

  std::cout<<"pushed all segments onto the index set"<<std::endl;

#if defined(RAJA_ENABLE_CUDA)
  int max_size = num_segments * 2;
  RAJA::Real_ptr test_array = ::test_array;
  RAJA::Real_ptr ref_array = ::ref_array;
  cudaMallocManaged((void **)&test_array,
                    sizeof(Real_type) * max_size,
                    cudaMemAttachGlobal);

  cudaMallocManaged((void **)&ref_array,
                    sizeof(Real_type) * max_size,
                    cudaMemAttachGlobal);

  cudaMemset(test_array, 0, sizeof(RAJA::Real_type) * num_segments*2);
  cudaMemset(ref_array, 0, sizeof(RAJA::Real_type) * num_segments*2);

  //int myvar2 = test_array[0];   //CRASHES!!!!
  //std::cout<<"myvar2="<<myvar2<<std::endl;

  RAJA::RAJAVec<int, managed_allocator<int> > test_array2;
  for (int i=0; i<num_segments*2; i++) {
    test_array2.push_back(0);
  }
  cudaDeviceSynchronize();

  int myvar3 = test_array2[0];
  std::cout<<"myvar3="<<myvar3<<std::endl;

  std::cout<<"Starting forall using CUDA"<<std::endl;

  using cuda_pol = RAJA::ExecPolicy<RAJA::seq_segit, RAJA::seq_segit>;
  //using cuda_pol = RAJA::ExecPolicy<RAJA::seq_segit, RAJA::cuda_exec<num_segments>>;
  //using cuda_pol = RAJA::ExecPolicy<RAJA::cuda_exec<num_segments>, RAJA::seq_segit>;



  RAJA::forall<cuda_pol>(
    is1, [=] __device__ (RAJA::Index_type idx) {
      test_array[idx] = idx;
      //int i=0;
      //i++;
    });




  cudaDeviceSynchronize();

  std::cout<<"after forall on the device"<<std::endl;

/*
  std::cout<<"answer array: ";
  for (int i=0; i<max_size; i++) {
    std::cout<<test_array[i]<<" ";
  } std::cout<<std::endl;
*/

  std::cout<<"finished forall with RAJA::ExecPolicy<RAJA::seq_segit, RAJA::seq_segit>;"<<std::endl;
  //std::cout<<"finished forall with RAJA::ExecPolicy<RAJA::seq_segit, RAJA::cuda_exec<num_segments>>;"<<std::endl;
  //  std::cout<<"finished forall with RAJA::ExecPolicy<RAJA::seq_segit, RAJA::seq_exec>;"<<std::endl;

  cudaFree(::test_array);
  cudaFree(::ref_array);
  cudaDeviceSynchronize();

#endif

  cout << "\n DONE!!! " << endl;

  return 0;
}
