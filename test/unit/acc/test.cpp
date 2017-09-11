
// control emmission of "fake pragmas" as deprecated warnings
// useful for determining correctness of SFINAE
//
// sample usage:
//   clang++ -std=c++14 -O3 test.cpp 2>&1 | grep 'deprecated-declarations'
//
//#define RAJA_ENABLE_VERBOSE

// didn't add to CMake yet

#define RAJA_ENABLE_OPENACC 1
#include "RAJA/policy/openacc.hpp"
#include "RAJA/pattern/forall.hpp"
#include "RAJA/pattern/forallN.hpp"

#include <algorithm>

int main() {
  namespace acc = RAJA::acc;

  const int N = 1024;
  using T = double;

  T *a, *b, *c;
  a = new T[N];
  b = new T[N];
  c = new T[N];

  std::fill(a, a + N, T(1));
  std::fill(b, b + N, T(2));

  RAJA::forall<
    RAJA::acc_parallel_loop_exec<
      acc::config<acc::NumGangs<32>, acc::NumVectors<32>>,
      acc::config<acc::Independent, acc::Gang, acc::Vector>>>
    (0, N, [=] (int i) {
    c[i] = a[i] + b[i];
  });

  bool valid = std::all_of(c, c + N, [] (T const &v) {
    return v == T(3);
  });

  std::cout << std::boolalpha << valid << std::endl;

  /*
  RAJA::forallN<
    RAJA::NestedPolicy<
      RAJA::ExecList<
        RAJA::acc_loop_exec<acc::config<acc::Independent>>,
        RAJA::acc_loop_exec<acc::config<acc::Independent>>>,
      RAJA::ACC_Parallel<acc::config<acc::NumVectors<32>>>>> (
    RAJA::RangeSegment{0,32},
    RAJA::RangeSegment{0,32},
    [=] (int out, int in) {
      int idx = out * 32 + in;
      c[idx] = a[idx] + b[idx];
    });
  */

  valid = std::all_of(c, c + N, [] (T const &v) {
    return v == T(3);
  });

  std::cout << std::boolalpha << valid << std::endl;

  delete[] a;
  delete[] b;
  delete[] c;


  return EXIT_SUCCESS;
}
