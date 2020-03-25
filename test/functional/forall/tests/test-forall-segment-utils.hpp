#ifndef __TEST_FORALL_SEGMENT_UTILS_HPP__
#define __TEST_FORALL_SEGMENT_UTILS_HPP__


#include "camp/resource.hpp"
#include "gtest/gtest.h"

using namespace camp::resources;
using namespace camp;


// Unroll types for gtest testing::Types
template <class T>
struct Test;

template <class... T>
struct Test<list<T...>> {
  using Types = ::testing::Types<T...>;
};


// Define Index Types
using IdxTypes = list<RAJA::Index_type,
                      short,
                      unsigned short,
                      int,
                      unsigned int,
                      long,
                      unsigned long,
                      long int,
                      unsigned long int,
                      long long,
                      unsigned long long>;

using ListHost = list<camp::resources::Host>;

template<typename T>
void allocateForallTestData(T N,
                            Resource& work_res,
                            T** work_array,
                            T** check_array,
                            T** test_array)
{
  Resource host_res{Host()};

  *work_array = work_res.allocate<T>(N);

  *check_array = host_res.allocate<T>(N);
  *test_array = host_res.allocate<T>(N);
}

template<typename T>
void deallocateForallTestData(Resource& work_res,
                              T* work_array,
                              T* check_array,
                              T* test_array)
{
  Resource host_res{Host()};

  work_res.deallocate(work_array);

  host_res.deallocate(check_array);
  host_res.deallocate(test_array);
}

#endif  // __TEST_FORALL_SEGMENT_UTILS_HPP__
