#include <omp.h>
#include "RAJA/RAJA.hxx"
#include <iostream>
#include <vector>
#include <chrono>

int test_sum()
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

int test_max()
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

int test_min()
{
  std::cout << "Testing ReduceMin...";

  std::vector<int> v2(500000000, 1);
  int * __restrict__ v = v2.data();
  RAJA::ReduceMin<RAJA::omp_reduce, int> r(0);
  RAJA::ReduceMin<RAJA::omp_reduce, int> r1(0);
  RAJA::ReduceMin<RAJA::omp_reduce, int> r2(0);
  RAJA::ReduceMin<RAJA::omp_reduce, int> r3(0);
  RAJA::ReduceMin<RAJA::omp_reduce, int> r4(0);
  RAJA::ReduceMin<RAJA::omp_reduce, int> r5(0);
  // std::cout << "val=" << r << " addr=" << &r << std::endl;

  auto start = std::chrono::high_resolution_clock::now();
  RAJA::forall<RAJA::omp_parallel_for_exec>(RAJA::RangeSegment(0,v2.size()),[=](int i) {
    r.min(v[i]);
    r1.min(v[i]);
    r2.min(v[i]);
    r3.min(v[i]);
    r4.min(v[i]);
    r5.min(v[i]);
  });
  auto stop = std::chrono::high_resolution_clock::now();

  std::cout << "done in " << std::chrono::duration_cast<std::chrono::duration<double, std::ratio<1>>>(stop - start).count() << "s" << std::endl;
  // std::cout << "Total=" << r << std::endl;
  return 0;
}

int test_maxloc()
{
  std::cout << "Testing ReduceMaxLoc...";

  std::vector<int> v2(500000000, 1);
  int * __restrict__ v = v2.data();
  RAJA::ReduceMaxLoc<RAJA::omp_reduce, int> r(0, 1);
  RAJA::ReduceMaxLoc<RAJA::omp_reduce, int> r1(0, 1);
  RAJA::ReduceMaxLoc<RAJA::omp_reduce, int> r2(0, 1);
  RAJA::ReduceMaxLoc<RAJA::omp_reduce, int> r3(0, 1);
  RAJA::ReduceMaxLoc<RAJA::omp_reduce, int> r4(0, 1);
  RAJA::ReduceMaxLoc<RAJA::omp_reduce, int> r5(0, 1);

  v[100] = 512;

  auto start = std::chrono::high_resolution_clock::now();
  RAJA::forall<RAJA::omp_parallel_for_exec>(RAJA::RangeSegment(0,v2.size()),[=](int i) {
    r.maxloc(v[i], i);
    r1.maxloc(v[i], i);
    r2.maxloc(v[i], i);
    r3.maxloc(v[i], i);
    r4.maxloc(v[i], i);
    r5.maxloc(v[i], i);
  });
  auto stop = std::chrono::high_resolution_clock::now();

  std::cout << "done in " << std::chrono::duration_cast<std::chrono::duration<double, std::ratio<1>>>(stop - start).count() << "s" << std::endl;
  // std::cout << "Total=" << r << std::endl;
  return 0;
}

int test_minloc()
{
  std::cout << "Testing ReduceMinLoc...";

  std::vector<int> v2(500000000, 1);
  int * __restrict__ v = v2.data();
  RAJA::ReduceMinLoc<RAJA::omp_reduce, int> r(0, 1);
  RAJA::ReduceMinLoc<RAJA::omp_reduce, int> r1(0, 1);
  RAJA::ReduceMinLoc<RAJA::omp_reduce, int> r2(0, 1);
  RAJA::ReduceMinLoc<RAJA::omp_reduce, int> r3(0, 1);
  RAJA::ReduceMinLoc<RAJA::omp_reduce, int> r4(0, 1);
  RAJA::ReduceMinLoc<RAJA::omp_reduce, int> r5(0, 1);

  v[100] = -1;

  auto start = std::chrono::high_resolution_clock::now();
  RAJA::forall<RAJA::omp_parallel_for_exec>(RAJA::RangeSegment(0,v2.size()),[=](int i) {
    r.minloc(v[i], i);
    r1.minloc(v[i], i);
    r2.minloc(v[i], i);
    r3.minloc(v[i], i);
    r4.minloc(v[i], i);
    r5.minloc(v[i], i);
  });
  auto stop = std::chrono::high_resolution_clock::now();

  std::cout << "done in " << std::chrono::duration_cast<std::chrono::duration<double, std::ratio<1>>>(stop - start).count() << "s" << std::endl;
  // std::cout << "Total=" << r << std::endl;
  return 0;
}

int main(int argc, char* argv[])
{
  test_sum();
  test_max();
  test_min();
  test_maxloc();
  test_minloc();

  return 0;
}
