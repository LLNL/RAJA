#define RAJA_ENABLE_OPENACC 1
#include "RAJA/policy/openacc.hpp"
#include "RAJA/pattern/forall.hpp"
#include "RAJA/pattern/forallN.hpp"

#include <algorithm>

#include <cuda.h>
#include <cuda_runtime.h>

int main() {
  namespace acc = RAJA::acc;

  const int N = 1024;
  using T = double;

  T *a, *b, *c;
  cudaMallocManaged(reinterpret_cast<void**>(&a), sizeof(T) * N);
  cudaMallocManaged(reinterpret_cast<void**>(&b), sizeof(T) * N);
  cudaMallocManaged(reinterpret_cast<void**>(&c), sizeof(T) * N);

  std::fill(a, a + N, T(1));
  std::fill(b, b + N, T(2));

  RAJA::forall<
    RAJA::acc_kernels_loop_exec<
      acc::config<acc::num::gangs<32>, acc::num::vectors<32>>,
      acc::config<acc::independent, acc::gang, acc::vector>>>
    (0, N, [=] (int i) {
    c[i] = a[i] + b[i];
  });

  bool valid = std::all_of(c, c + N, [] (T const &v) {
    return v == T(3);
  });

  std::cout << std::boolalpha << valid << std::endl;

  RAJA::forallN<
    RAJA::ExecList<
      RAJA::acc_loop_exec<acc::config<acc::independent>>,
      RAJA::acc_loop_exec<acc::config<acc::independent>>>,
    RAJA::ACC_Kernels<acc::config<acc::num::vectors<32>>>> (
      RAJA::RangeSegment{0,32},
      RAJA::RangeSegment{0,32},
      [=] (int out, int in) {
        int idx = out * 32 + in;
        c[idx] = a[idx] + b[idx];
      });

  bool valid = std::all_of(c, c + N, [] (T const &v) {
    return v == T(3);
  });

  std::cout << std::boolalpha << valid << std::endl;

  cudaFree(a);
  cudaFree(b);
  cudaFree(c);

  return EXIT_SUCCESS;
}
