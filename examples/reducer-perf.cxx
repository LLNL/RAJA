#include <omp.h>
#include "RAJA/RAJA.hxx"
#include <iostream>
#include <vector>
#include <chrono>

int test_sum(int argc, char *argv[])
{
  std::cout << "Testing ReduceSum...";

  std::vector<int> v2(500000000, 1);
  int * __restrict__ v = v2.data();
  RAJA::ReduceSum<RAJA::omp_reduce, int> r(0);
  RAJA::ReduceSum<RAJA::omp_reduce, int> r1(0);
  RAJA::ReduceSum<RAJA::omp_reduce, int> r2(0);
  RAJA::ReduceSum<RAJA::omp_reduce, int> r3(0);
  RAJA::ReduceSum<RAJA::omp_reduce, int> r4(0);
  RAJA::ReduceSum<RAJA::omp_reduce, int> r5(0);
  // std::cout << "val=" << r << " addr=" << &r << std::endl;

  auto start = std::chrono::high_resolution_clock::now();
  RAJA::forall<RAJA::omp_parallel_for_exec>(RAJA::RangeSegment(0,v2.size()),[=](int i) {
    r += v[i];
    r1 += v[i];
    r2 += v[i];
    r3 += v[i];
    r4 += v[i];
    r5 += v[i];
  });
  auto stop = std::chrono::high_resolution_clock::now();

  std::cout << "done in " << std::chrono::duration_cast<std::chrono::duration<double, std::ratio<1>>>(stop - start).count() << "s" << std::endl;
  // std::cout << "Total=" << r << std::endl;
  return 0;
}

int test_max(int argc, char *argv[])
{
  std::cout << "Testing ReduceMax...";

  std::vector<int> v2(500000000, 1);
  int * __restrict__ v = v2.data();
  RAJA::ReduceMax<RAJA::omp_reduce, int> r(0);
  RAJA::ReduceMax<RAJA::omp_reduce, int> r1(0);
  RAJA::ReduceMax<RAJA::omp_reduce, int> r2(0);
  RAJA::ReduceMax<RAJA::omp_reduce, int> r3(0);
  RAJA::ReduceMax<RAJA::omp_reduce, int> r4(0);
  RAJA::ReduceMax<RAJA::omp_reduce, int> r5(0);
  // std::cout << "val=" << r << " addr=" << &r << std::endl;

  auto start = std::chrono::high_resolution_clock::now();
  RAJA::forall<RAJA::omp_parallel_for_exec>(RAJA::RangeSegment(0,v2.size()),[=](int i) {
    r.max(v[i]);
    r1.max(v[i]);
    r2.max(v[i]);
    r3.max(v[i]);
    r4.max(v[i]);
    r5.max(v[i]);
  });
  auto stop = std::chrono::high_resolution_clock::now();

  std::cout << "done in " << std::chrono::duration_cast<std::chrono::duration<double, std::ratio<1>>>(stop - start).count() << "s" << std::endl;
  // std::cout << "Total=" << r << std::endl;
  return 0;
}

int maint(int argc, char* argv[])
{
  test_sum();
  test_max();
}
