#include <omp.h>
#include "RAJA/RAJA.hxx"
#include <iostream>
#include <vector>
#include <chrono>

int main(int argc, char *argv[])
{
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

    std::cout << "Total=" << r << std::endl;
    std::cout << "Time=" << std::chrono::duration_cast<std::chrono::duration<double, std::ratio<1>>>(stop - start).count() << std::endl;   
    return 0;
}
