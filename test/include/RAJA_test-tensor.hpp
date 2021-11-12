//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-19, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef RAJA_TEST_TENSOR
#define RAJA_TEST_TENSOR


#include "RAJA/RAJA.hpp"
#include "RAJA_gtest.hpp"


using TensorElementTypes = ::testing::Types<
        int,
        long,
        float,
        double
    >;

template<typename POL>
struct TensorTestHelper {

    template<typename BODY>
    static
    void exec(BODY const &body){
      body();
    }

    static constexpr bool is_device = false;
};

#ifdef RAJA_ENABLE_CUDA

template <typename BODY>
__global__
void test_launcher(BODY body_in)
{
  using RAJA::internal::thread_privatize;
  auto privatizer = thread_privatize(body_in);
  auto& body = privatizer.get_priv();
  body();
}

template<>
struct TensorTestHelper<RAJA::cuda_warp_register>
{

    RAJA_SUPPRESS_HD_WARN
    template<typename BODY>
    static
    void exec(BODY const &body){
      cudaDeviceSynchronize();

      test_launcher<<<1,32>>>(body);

      cudaDeviceSynchronize();

    }

    static constexpr bool is_device = true;
};
#endif


template<typename POL, typename BODY>
void tensor_do(BODY const &body){
  TensorTestHelper<POL>::exec(body);
}



#ifdef RAJA_ENABLE_CUDA

template<typename POL, typename T>
T* tensor_malloc(size_t len){
  if(TensorTestHelper<POL>::is_device){
    T *ptr;

    cudaErrchk(cudaMalloc(&ptr, len*sizeof(T)));

    return ptr;
  }
  else{
    return new T[len];
  }
}

template<typename POL, typename T>
void tensor_free(T *ptr){
  if(TensorTestHelper<POL>::is_device){
    cudaErrchk(cudaFree(ptr));
  }
  else{
    delete[] ptr;
  }
}

template<typename POL, typename T>
void tensor_copy_to_device(T *d_ptr, std::vector<T> const &h_vec){
  if(TensorTestHelper<POL>::is_device){
    cudaErrchk(cudaMemcpy(d_ptr, h_vec.data(), h_vec.size()*sizeof(T), cudaMemcpyHostToDevice));
  }
  else{
    memcpy(d_ptr, h_vec.data(), h_vec.size()*sizeof(T));
  }
}

template<typename POL, typename T>
void tensor_copy_to_host(std::vector<T> &h_vec, T const *d_ptr){
  if(TensorTestHelper<POL>::is_device){
    cudaErrchk(cudaMemcpy(h_vec.data(), d_ptr, h_vec.size()*sizeof(T), cudaMemcpyDeviceToHost));
  }
  else{
    memcpy(h_vec.data(), d_ptr, h_vec.size()*sizeof(T));
  }
}

#else

template<typename POL, typename T>
T* tensor_malloc(size_t len){
  return new T[len];
}

template<typename POL, typename T>
void tensor_free(T *ptr){
  delete[] ptr;
}

template<typename POL, typename T>
void tensor_copy_to_device(T *d_ptr, std::vector<T> const &h_vec){
  memcpy(d_ptr, h_vec.data(), h_vec.size()*sizeof(T));
}

template<typename POL, typename T>
void tensor_copy_to_host(std::vector<T> &h_vec, T const *d_ptr){
  memcpy(h_vec.data(), d_ptr, h_vec.size()*sizeof(T));
}

#endif



// Sugar to make things cleaner
template<typename POL, typename T>
T* tensor_malloc(std::vector<T> const &vec){
  return tensor_malloc<POL,T>(vec.size());
}




#endif
