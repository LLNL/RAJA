//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016, Lawrence Livermore National Security, LLC.
//
// Produced at the Lawrence Livermore National Laboratory
//
// LLNL-CODE-689114
//
// All rights reserved.
//
// This file is part of RAJA.
//
// For additional details, please also read RAJA/LICENSE.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// * Redistributions of source code must retain the above copyright notice,
//   this list of conditions and the disclaimer below.
//
// * Redistributions in binary form must reproduce the above copyright notice,
//   this list of conditions and the disclaimer (as noted below) in the
//   documentation and/or other materials provided with the distribution.
//
// * Neither the name of the LLNS/LLNL nor the names of its contributors may
//   be used to endorse or promote products derived from this software without
//   specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL LAWRENCE LIVERMORE NATIONAL SECURITY,
// LLC, THE U.S. DEPARTMENT OF ENERGY OR CONTRIBUTORS BE LIABLE FOR ANY
// DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
// DAMAGES  (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS
// OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
// HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
// STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING
// IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
//
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include <cstdlib>
#include <iostream>

#include "RAJA/RAJA.hpp"
#include "RAJA/util/defines.hpp"

#define CUDA_BLOCK_SIZE 256

void checkSolution(int *C, int in_N);


//RAJA does not do memory management. 
int * allocate(RAJA::Index_type size) {
  int * ptr;
#if defined(RAJA_ENABLE_CUDA)
  cudaMallocManaged((void**)&ptr, sizeof(int)*size,cudaMemAttachGlobal);
#else
  ptr = new int[size];
#endif
  return ptr;
}

void deallocate(int* &ptr) {
  if (ptr) {
#if defined(RAJA_ENABLE_CUDA)
    cudaFree(ptr);
#else
    delete[] ptr;
#endif
    ptr = nullptr;
  }
}

/*Example 1: Adding Two Vectors

  -----[New Concepts]---------------
  1. Introduces the forall for loop and basic RAJA policies. 

  ----[RAJA forall loop]---------------
  RAJA::forall<RAJA::exec_policy>(RAJA::range_policy,[=](index i)) {
  //body
  });
  
  ----[Arguments]--------------
  [=] Corresponds to pass by copy
  [&] Corresponds to pass by reference
  RAJA::exec_policy  - specifies where and how the traversal occurs
  RAJA::range_policy - provides a list in which the index may iterate on
 */
int main(int RAJA_UNUSED_ARG(argc), char** RAJA_UNUSED_ARG(argv[]))
{

  std::cout<<"Example 1: Adding Two Vectors"<<std::endl;

  const int N = 1000;
  int *A = allocate(N);
  int *B = allocate(N);
  int *C = allocate(N);
  
  //Populate vectors
  for(int i=0; i<N; ++i) {
    A[i] = i;
    B[i] = i;
  }


  //----[Standard C++ loop]---------------
  std::cout<<"Standard C++ loop"<<std::endl;
  for(int i=0; i<N; ++i) {
    C[i] = A[i] + B[i];
  }
  checkSolution(C,N);
  std::cout<<"\n"<<std::endl;
  //======================================
     
  
  /* 
  //----[RAJA: Sequential Policy]---------
  RAJA::seq_exec -  Executes the loop in serial     

  RAJA::RangeSegment(start,stop) - generates a list of numbers, 
  which is may be used to iterate over a for loop. 
  start - starting number of the sequence
  stop  - generated numbers up, but not including N
  */
  std::cout<<"RAJA: Sequential Policy"<<std::endl;
  RAJA::forall<RAJA::seq_exec>(RAJA::RangeSegment(0,N),[=](int i) {
      C[i] = A[i] + B[i];     
    });
  checkSolution(C,N);
  std::cout<<"\n"<<std::endl;
  //======================================


#if defined(RAJA_ENABLE_OPENMP)
  //----[RAJA: OpenMP Policy]---------
  /*
    RAJA::omp_parallel_for_exec - executes the for loop using the
    #pragma omp parallel for directive
   */  

  std::cout<<"RAJA: OpenMP Policy"<<std::endl;
  RAJA::forall<RAJA::omp_parallel_for_exec>(RAJA::RangeSegment(0,N),[=](int i) {
      C[i] = A[i] + B[i];     
    });
  checkSolution(C,N);
  std::cout<<"\n"<<std::endl;
  //======================================
#endif



#if defined(RAJA_ENABLE_CUDA)
  //----[RAJA: CUDA Policy]---------
  /*
    RAJA::cuda_exec<CUDA_BLOCK_SIZE> - excecutes loop using the CUDA API. 
    Each thread is assigned to an iteration of the loop

    CUDA_BLOCK_SIZE - specifies the number of threads per block
    on a one dimension compute grid
   */
  std::cout<<"RAJA: CUDA Policy"<<std::endl;  
  RAJA::forall<RAJA::cuda_exec<CUDA_BLOCK_SIZE>>(RAJA::RangeSegment(0,N),[=] __device__(int i) {
      C[i] = A[i] + B[i];
    });

  checkSolution(C,N);
  std::cout<<"\n"<<std::endl;
  //======================================
#endif

  deallocate(A);
  deallocate(B);
  deallocate(C);

  return 0;
}

void checkSolution(int *C, int in_N){

  bool eFlag = true;
  for(int i=0; i<in_N; ++i){
    if( (C[i] - (i+i)) > 1e-9){
      std::cout<<"Error in result!"<<std::endl;
      eFlag = false; break;
    }
  }

  if(eFlag){
    std::cout<<"Result is correct"<<std::endl;
  }
}
