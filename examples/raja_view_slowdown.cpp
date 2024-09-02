//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-24, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include <RAJA/RAJA.hpp>
#include "RAJA/util/Timer.hpp"
#include <iostream>

int main() {

  const int N = 10000;
  const int K = 17;

  auto timer = RAJA::Timer();

  //launch to intialize the stream
  RAJA::forall<RAJA::cuda_exec<256>>
    (RAJA::RangeSegment(0,1), [=] __device__ (int i) {
    printf("launch kernel\n");
  });


  int* array = new int[N * N];
  int* array_copy = new int[N * N];

  //big array, or image
  for (int i = 0; i < N * N; ++i) {
    array[i] = 1;
    array_copy[i] = 1;
  }

  //small array that acts as the blur
  int* kernel = new int[K * K];
  for (int i = 0; i < K * K; ++i) {
    kernel[i] = 2;
  }

  // copying to gpu
  int* d_array;
  int* d_array_copy;
  int* d_kernel;
  cudaMalloc((void**)&d_array, N * N * sizeof(int));
  cudaMalloc((void**)&d_array_copy, N * N * sizeof(int));
  cudaMalloc((void**)&d_kernel, K * K * sizeof(int));
  cudaMemcpy(d_array, array, N * N * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(d_array_copy, array_copy, N * N * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(d_kernel, kernel, K * K * sizeof(int), cudaMemcpyHostToDevice);


  constexpr int DIM = 2;
  RAJA::View<int, RAJA::Layout<DIM, int, 1>> array_view(d_array, N, N);
  RAJA::View<int, RAJA::Layout<DIM, int, 1>> array_view_copy(d_array_copy, N, N);
  RAJA::View<int, RAJA::Layout<DIM, int, 1>> kernel_view(d_kernel, K, K);


  using EXEC_POL5 = RAJA::KernelPolicy<
    RAJA::statement::CudaKernelFixed<256,
      RAJA::statement::For<1, RAJA::cuda_global_size_y_direct<16>,
        RAJA::statement::For<0, RAJA::cuda_global_size_x_direct<16>,
	  RAJA::statement::Lambda<0>
	  >
       >
     >
    >;

  RAJA::RangeSegment range_i(0, N);
  RAJA::RangeSegment range_j(0, N);


timer.start();

  RAJA::kernel<EXEC_POL5>
    (RAJA::make_tuple(range_i, range_j),
     [=] RAJA_DEVICE (int i, int j) {
      int sum = 0;

      //looping through the "blur"
      for (int m = 0; m < K; ++m) {
	for (int n = 0; n < K; ++n) {
	  int x = i + m;
	  int y = j + n;

	  // adding the "blur" to the "image" wherever the blur is located on the image
	  if (x < N && y < N) {
	    sum += kernel_view(m, n) * array_view(x, y);
	  }
	}
      }

      array_view(i, j) += sum;
    }
   );

timer.stop();

std::cout<<"Elapsed time with RAJA view : "<<timer.elapsed()<<std::endl;


timer.reset();
timer.start();

  RAJA::kernel<EXEC_POL5>
    (RAJA::make_tuple(range_i, range_j),
     [=] RAJA_DEVICE (int i, int j) {
      int sum = 0;

      // looping through the "blur"
      for (int m = 0; m < K; ++m) {
	for (int n = 0; n < K; ++n) {
	  int x = i + m;
	  int y = j + n;

	  // adding the "blur" to the "image" wherever the blur is located on the image
	  if (x < N && y < N) {
	    sum += d_kernel[m * K + n] * d_array_copy[x * N + y];
	  }
	}
      }

      d_array_copy[i * N + j] += sum;
    }
  );

timer.stop();
std::cout<<"Elapsed time with NO RAJA view : "<<timer.elapsed()<<std::endl;

  // copy from gpu to cpu
  cudaMemcpy(array, d_array, N * N * sizeof(int), cudaMemcpyDeviceToHost);
  cudaMemcpy(array_copy, d_array_copy, N * N * sizeof(int), cudaMemcpyDeviceToHost);


  cudaFree(d_array);
  cudaFree(d_array_copy);
  cudaFree(d_kernel);


  delete[] array;
  delete[] array_copy;
  delete[] kernel;

  return 0;
}
