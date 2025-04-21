//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-25, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include <cstdlib>
#include <iostream>
#include <iomanip>  
#include <vector>
#include <cassert>
#include <chrono>

#include "RAJA/RAJA.hpp"
#include "memoryManager.hpp"

/*
 *  Prefix Sum Example
 *
 *  Computes an exclusive prefix sum (scan) on an array of integers using
 *  multiple execution policies.
 *
 *  RAJA features shown:
 *    - `exclusive_scan` operation with different execution backends
 *    - Use of `make_span` to create RAJA-compatible views
 *    - Sequential, OpenMP, and CUDA execution variants
 *    - CUDA device memory allocation and transfer
 *
 *  If CUDA is enabled, device memory is allocated manually with `cudaMalloc`
 *  and the results are copied back to host memory for validation.
 */


/*
  CUDA_BLOCK_SIZE - specifies the number of threads in a CUDA thread block
*/
#if defined(RAJA_ENABLE_CUDA)
const int CUDA_BLOCK_SIZE = 256;
#endif

/*
  N - the length of the series to perform the prefix sum
*/
const int N = 100;

bool check_equal(const std::vector<int>& a, const std::vector<int>& b)
{
  if (a.size() != b.size()) return false;
  for (size_t i = 0; i < a.size(); ++i) {
    if (a[i] != b[i]) return false;
  }
  return true;
}

int main(int RAJA_UNUSED_ARG(argc), char** RAJA_UNUSED_ARG(argv))
{
  std::cout << "\n\nRAJA prefix sum (exclusive_scan) example using a series of " << N << " length...\n";

  std::vector<int> input(N, 1);
  std::vector<int> reference_output(N, 0);
  std::vector<int> test_output(N, 0);

  //----------------------------------------------------------------------------
  std::cout << "\n Running C-style prefix sum...\n";

  auto start = std::chrono::high_resolution_clock::now();

  reference_output[0] = 0;
  for (int i = 1; i < N; ++i) {
    reference_output[i] = reference_output[i - 1] + input[i - 1];
  }

  auto end = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
  std::cout << "   Reference complete. Time: " << duration.count() << " us\n";

  //----------------------------------------------------------------------------
  std::cout << "\n Running RAJA::exclusive_scan with seq_exec...\n";
  std::fill(test_output.begin(), test_output.end(), 0);

  start = std::chrono::high_resolution_clock::now();

  RAJA::exclusive_scan<RAJA::seq_exec>(
    RAJA::make_span(input),
    RAJA::make_span(test_output),
    RAJA::operators::plus<int>());

  end = std::chrono::high_resolution_clock::now();
  duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

  std::cout << "   Result: " 
            << (check_equal(reference_output, test_output) ? "PASS" : "FAIL")
            << " | Time: " << duration.count() << " us\n";


  //----------------------------------------------------------------------------
#if defined(RAJA_ENABLE_OPENMP)
  std::cout << "\n Running RAJA::exclusive_scan with omp_parallel_for_exec...\n";
  std::fill(test_output.begin(), test_output.end(), 0);
  
  start = std::chrono::high_resolution_clock::now();
  
  RAJA::exclusive_scan<RAJA::omp_parallel_for_exec>(
    RAJA::make_span(input),
    RAJA::make_span(test_output),
    RAJA::operators::plus<int>());
  
  end = std::chrono::high_resolution_clock::now();
  duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
  
  std::cout << "   Result: " 
            << (check_equal(reference_output, test_output) ? "PASS" : "FAIL")
            << " | Time: " << duration.count() << " us\n";
#endif
  

  //----------------------------------------------------------------------------
#if defined(RAJA_ENABLE_CUDA)
  std::cout << "\n Running RAJA::exclusive_scan with cuda_exec...\n";
  
  int* d_input;
  int* d_output;
  
  cudaMalloc((void**)&d_input, N * sizeof(int));
  cudaMalloc((void**)&d_output, N * sizeof(int));
  
  cudaMemcpy(d_input, input.data(), N * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemset(d_output, 0, N * sizeof(int));
  
  cudaDeviceSynchronize();
  start = std::chrono::high_resolution_clock::now();
  
  RAJA::exclusive_scan<RAJA::cuda_exec<CUDA_BLOCK_SIZE>>(
    RAJA::make_span(d_input, N),
    RAJA::make_span(d_output, N),
    RAJA::operators::plus<int>{});
  
  cudaDeviceSynchronize();  // Make sure the scan finishes before timing ends
  end = std::chrono::high_resolution_clock::now();
  duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
  
  cudaMemcpy(test_output.data(), d_output, N * sizeof(int), cudaMemcpyDeviceToHost);
  
  std::cout << "   Result: " 
            << (check_equal(reference_output, test_output) ? "PASS" : "FAIL")
            << " | Time: " << duration.count() << " us\n";
  
  cudaFree(d_input);
  cudaFree(d_output);
#endif
  

  std::cout << "\n DONE!...\n";
  return 0;
}
