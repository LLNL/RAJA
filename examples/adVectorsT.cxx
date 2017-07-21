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

constexpr int CUDA_BLOCK_SIZE = 256;

void checkSolution(double *C, int in_N);


struct DefaultCPP{};

// Generic RAJA implementation for CPU
template <typename Policy>
struct VectorAdd {
  template <typename T>
  void operator()(const T* a, const T* b, T* c, int N) {
    RAJA::forall<Policy>(RAJA::RangeSegment(0, N), [=] (int i) {
        c[i] = a[i] + b[i];
      });
  }
};

// Generic CPP implementation
template <>
struct VectorAdd<DefaultCPP> {
  template <typename T>
  void operator()(const T* a, const T* b, T* c, int N) {
    for (int i = 0; i < N; ++i) {
      c[i] = a[i] + b[i];
    }
  }
};

// RAJA implementation for CUDA
template <size_t BLOCK_SIZE>
struct VectorAdd<RAJA::cuda_exec<BLOCK_SIZE>> {  
  template <typename T>
  void operator()(const T* a, const T* b, T* c, int N) {
    RAJA::forall<RAJA::cuda_exec<BLOCK_SIZE>>(RAJA::RangeSegment(0, N), [=] __device__ (int i) {
        c[i] = a[i] + b[i];
      });
  }
};


template <typename T>
T* allocate(RAJA::Index_type size) {
  T* ptr;
#if defined(RAJA_ENABLE_CUDA)
  cudaMallocManaged((void**)&ptr, sizeof(T)*size,cudaMemAttachGlobal);
#else
  ptr = new T[size];
#endif
  return ptr;
}

template <typename T>
void deallocate(T* &ptr) {
  if (ptr) {
#if defined(RAJA_ENABLE_CUDA)
    cudaFree(ptr);
#else
    delete[] ptr;
#endif
    ptr = nullptr;
  }
}

int main(int RAJA_UNUSED_ARG(argc), char** RAJA_UNUSED_ARG(argv[]))
{

  std::cout<<"Example 1: Adding Two Vectors"<<std::endl;

  const int N = 1000;

  auto A = allocate<double>(N);
  auto B = allocate<double>(N);
  auto C = allocate<double>(N);
  
  //Populate vectors
  for(int i=0; i<N; ++i){
    A[i] = i;
    B[i] = i;
  }
  
  //----[Standard C++ loop]---------------
  std::cout<<"Standard C++ loop"<<std::endl;
  VectorAdd<DefaultCPP>()(A, B, C, N);
  checkSolution(C,N);
  std::cout<<"\n"<<std::endl;
  //======================================


  //----[RAJA: Sequential Policy]---------
  std::cout<<"RAJA: Sequential Policy"<<std::endl;
  VectorAdd<RAJA::seq_exec>()(A, B, C, N);
  checkSolution(C,N);
  std::cout<<"\n"<<std::endl;
  //======================================

#if defined(RAJA_ENABLE_OPENMP)
  //----[RAJA: OpenMP Policy]---------
  std::cout<<"RAJA: OpenMP Policy"<<std::endl;
  VectorAdd<RAJA::omp_parallel_for_exec>()(A, B, C, N);
  checkSolution(C,N);
  std::cout<<"\n"<<std::endl;
  //======================================
#endif

#if defined(RAJA_ENABLE_CUDA)
  //----[RAJA: CUDA Policy]---------
  std::cout<<"RAJA: CUDA Policy"<<std::endl;
  VectorAdd<RAJA::cuda_exec<CUDA_BLOCK_SIZE>>()(A, B, C, N);
  checkSolution(C,N);
  std::cout<<"\n"<<std::endl;
  //======================================
#endif

  deallocate(A);
  deallocate(B);
  deallocate(C);
  
  return 0;
}

void checkSolution(double *C, int in_N){

  for(int i=0; i<in_N; ++i){
    if( (C[i] - (i+i)) > 1e-9){
      std::cout<<"Error in result!"<<std::endl;
      return;
    }
  }
  std::cout<<"Result is correct"<<std::endl;

}
