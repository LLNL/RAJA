//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-21, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

/*
 *  Reproducer for compile error when using random number generators
 */

/*
 * Compile line followed by error output

/usr/tce/packages/clang/clang-10.0.1-gcc-8.3.1/bin/clang++ -std=c++14 -fopenmp -fopenmp-targets=nvptx64-nvidia-cuda -o reproducer-openmptarget-issue3.cpp.o -c reproducer-openmptarget-issue3.cpp

In file included from reproducer-openmptarget-issue3.cpp:8:
In file included from /usr/tce/packages/gcc/gcc-8.3.1/rh/usr/lib/gcc/ppc64le-redhat-linux/8/../../../../include/c++/8/random:51:
/usr/tce/packages/gcc/gcc-8.3.1/rh/usr/lib/gcc/ppc64le-redhat-linux/8/../../../../include/c++/8/bits/random.tcc:3325:30: error: call to 'log' is ambiguous
      const size_t __log2r = std::log(__r) / std::log(2.0L);
                             ^~~~~~~~
/usr/tce/packages/clang/clang-10.0.1/release/lib/clang/10.0.1/include/__clang_cuda_device_functions.h:1603:19: note: candidate function
__DEVICE__ double log(double __a) { return __nv_log(__a); }
                  ^
/usr/tce/packages/clang/clang-10.0.1/release/lib/clang/10.0.1/include/__clang_cuda_cmath.h:140:18: note: candidate function
__DEVICE__ float log(float __x) { return ::logf(__x); }
                 ^
In file included from reproducer-openmptarget-issue3.cpp:8:
In file included from /usr/tce/packages/gcc/gcc-8.3.1/rh/usr/lib/gcc/ppc64le-redhat-linux/8/../../../../include/c++/8/random:51:
/usr/tce/packages/gcc/gcc-8.3.1/rh/usr/lib/gcc/ppc64le-redhat-linux/8/../../../../include/c++/8/bits/random.tcc:3325:46: error: call to 'log' is ambiguous
      const size_t __log2r = std::log(__r) / std::log(2.0L);
                                             ^~~~~~~~
/usr/tce/packages/clang/clang-10.0.1/release/lib/clang/10.0.1/include/__clang_cuda_device_functions.h:1603:19: note: candidate function
__DEVICE__ double log(double __a) { return __nv_log(__a); }
                  ^
/usr/tce/packages/clang/clang-10.0.1/release/lib/clang/10.0.1/include/__clang_cuda_cmath.h:140:18: note: candidate function
__DEVICE__ float log(float __x) { return ::logf(__x); }
                 ^
In file included from reproducer-openmptarget-issue3.cpp:8:
In file included from /usr/tce/packages/gcc/gcc-8.3.1/rh/usr/lib/gcc/ppc64le-redhat-linux/8/../../../../include/c++/8/random:51:
/usr/tce/packages/gcc/gcc-8.3.1/rh/usr/lib/gcc/ppc64le-redhat-linux/8/../../../../include/c++/8/bits/random.tcc:3325:20: error: default initialization of an object of const type 'const std::size_t' (aka 'const unsigned long')
      const size_t __log2r = std::log(__r) / std::log(2.0L);
                   ^
/usr/tce/packages/gcc/gcc-8.3.1/rh/usr/lib/gcc/ppc64le-redhat-linux/8/../../../../include/c++/8/bits/random.h:179:16: note: in instantiation of function template specialization 'std::generate_canonical<double, 53, std::linear_congruential_engine<unsigned long, 16807, 0, 2147483647> >' requested here
          return std::generate_canonical<_DInputType,
                      ^
/usr/tce/packages/gcc/gcc-8.3.1/rh/usr/lib/gcc/ppc64le-redhat-linux/8/../../../../include/c++/8/bits/random.h:1823:12: note: in instantiation of member function 'std::__detail::_Adaptor<std::linear_congruential_engine<unsigned long, 16807, 0, 2147483647>, double>::operator()' requested here
          return (__aurng() * (__p.b() - __p.a())) + __p.a();
                  ^
/usr/tce/packages/gcc/gcc-8.3.1/rh/usr/lib/gcc/ppc64le-redhat-linux/8/../../../../include/c++/8/bits/random.h:1814:24: note: in instantiation of function template specialization 'std::uniform_real_distribution<double>::operator()<std::linear_congruential_engine<unsigned long, 16807, 0, 2147483647> >' requested here
        { return this->operator()(__urng, _M_param); }
                       ^
reproducer-openmptarget-issue3.cpp:27:21: note: in instantiation of function template specialization 'std::uniform_real_distribution<double>::operator()<std::linear_congruential_engine<unsigned long, 16807, 0, 2147483647> >' requested here
  double dval = dist(gen);
                    ^
3 errors generated.
 */

/*
 * Compile line followed by error output

/usr/tce/packages/clang/clang-10.0.1/bin/clang++ -std=c++14 -fopenmp -fopenmp-targets=nvptx64-nvidia-cuda -o reproducer-openmptarget-issue3.cpp.o -c reproducer-openmptarget-issue3.cpp

In file included from reproducer-openmptarget-issue3.cpp:8:
In file included from /usr/tce/packages/gcc/gcc-4.9.3/lib64/gcc/powerpc64le-unknown-linux-gnu/4.9.3/../../../../include/c++/4.9.3/random:51:
/usr/tce/packages/gcc/gcc-4.9.3/lib64/gcc/powerpc64le-unknown-linux-gnu/4.9.3/../../../../include/c++/4.9.3/bits/random.tcc:1476:27: error: call to 'abs' is ambiguous
                    const double __y = -std::abs(__n) * __param._M_sm - 1;
                                        ^~~~~~~~
/usr/include/stdlib.h:770:12: note: candidate function
extern int abs (int __x) __THROW __attribute__ ((__const__)) __wur;
           ^
/usr/tce/packages/gcc/gcc-4.9.3/lib64/gcc/powerpc64le-unknown-linux-gnu/4.9.3/../../../../include/c++/4.9.3/cstdlib:166:3: note: candidate function
  abs(long __i) { return __builtin_labs(__i); }
  ^
/usr/tce/packages/gcc/gcc-4.9.3/lib64/gcc/powerpc64le-unknown-linux-gnu/4.9.3/../../../../include/c++/4.9.3/cstdlib:174:3: note: candidate function
  abs(long long __x) { return __builtin_llabs (__x); }
  ^
In file included from reproducer-openmptarget-issue3.cpp:8:
In file included from /usr/tce/packages/gcc/gcc-4.9.3/lib64/gcc/powerpc64le-unknown-linux-gnu/4.9.3/../../../../include/c++/4.9.3/random:51:
/usr/tce/packages/gcc/gcc-4.9.3/lib64/gcc/powerpc64le-unknown-linux-gnu/4.9.3/../../../../include/c++/4.9.3/bits/random.tcc:1485:30: error: call to 'abs' is ambiguous
                    const double __y = 1 + std::abs(__n) * __param._M_scx;
                                           ^~~~~~~~
/usr/include/stdlib.h:770:12: note: candidate function
extern int abs (int __x) __THROW __attribute__ ((__const__)) __wur;
           ^
/usr/tce/packages/gcc/gcc-4.9.3/lib64/gcc/powerpc64le-unknown-linux-gnu/4.9.3/../../../../include/c++/4.9.3/cstdlib:166:3: note: candidate function
  abs(long __i) { return __builtin_labs(__i); }
  ^
/usr/tce/packages/gcc/gcc-4.9.3/lib64/gcc/powerpc64le-unknown-linux-gnu/4.9.3/../../../../include/c++/4.9.3/cstdlib:174:3: note: candidate function
  abs(long long __x) { return __builtin_llabs (__x); }
  ^
In file included from reproducer-openmptarget-issue3.cpp:8:
In file included from /usr/tce/packages/gcc/gcc-4.9.3/lib64/gcc/powerpc64le-unknown-linux-gnu/4.9.3/../../../../include/c++/4.9.3/random:51:
/usr/tce/packages/gcc/gcc-4.9.3/lib64/gcc/powerpc64le-unknown-linux-gnu/4.9.3/../../../../include/c++/4.9.3/bits/random.tcc:1727:42: error: call to 'abs' is ambiguous
                    const double __y = __param._M_s1 * std::abs(__n);
                                                       ^~~~~~~~
/usr/include/stdlib.h:770:12: note: candidate function
extern int abs (int __x) __THROW __attribute__ ((__const__)) __wur;
           ^
/usr/tce/packages/gcc/gcc-4.9.3/lib64/gcc/powerpc64le-unknown-linux-gnu/4.9.3/../../../../include/c++/4.9.3/cstdlib:166:3: note: candidate function
  abs(long __i) { return __builtin_labs(__i); }
  ^
/usr/tce/packages/gcc/gcc-4.9.3/lib64/gcc/powerpc64le-unknown-linux-gnu/4.9.3/../../../../include/c++/4.9.3/cstdlib:174:3: note: candidate function
  abs(long long __x) { return __builtin_llabs (__x); }
  ^
In file included from reproducer-openmptarget-issue3.cpp:8:
In file included from /usr/tce/packages/gcc/gcc-4.9.3/lib64/gcc/powerpc64le-unknown-linux-gnu/4.9.3/../../../../include/c++/4.9.3/random:51:
/usr/tce/packages/gcc/gcc-4.9.3/lib64/gcc/powerpc64le-unknown-linux-gnu/4.9.3/../../../../include/c++/4.9.3/bits/random.tcc:1739:42: error: call to 'abs' is ambiguous
                    const double __y = __param._M_s2 * std::abs(__n);
                                                       ^~~~~~~~
/usr/include/stdlib.h:770:12: note: candidate function
extern int abs (int __x) __THROW __attribute__ ((__const__)) __wur;
           ^
/usr/tce/packages/gcc/gcc-4.9.3/lib64/gcc/powerpc64le-unknown-linux-gnu/4.9.3/../../../../include/c++/4.9.3/cstdlib:166:3: note: candidate function
  abs(long __i) { return __builtin_labs(__i); }
  ^
/usr/tce/packages/gcc/gcc-4.9.3/lib64/gcc/powerpc64le-unknown-linux-gnu/4.9.3/../../../../include/c++/4.9.3/cstdlib:174:3: note: candidate function
  abs(long long __x) { return __builtin_llabs (__x); }
  ^
In file included from reproducer-openmptarget-issue3.cpp:8:
In file included from /usr/tce/packages/gcc/gcc-4.9.3/lib64/gcc/powerpc64le-unknown-linux-gnu/4.9.3/../../../../include/c++/4.9.3/random:51:
/usr/tce/packages/gcc/gcc-4.9.3/lib64/gcc/powerpc64le-unknown-linux-gnu/4.9.3/../../../../include/c++/4.9.3/bits/random.tcc:3474:30: error: call to 'log' is ambiguous
      const size_t __log2r = std::log(__r) / std::log(2.0L);
                             ^~~~~~~~
/usr/tce/packages/clang/clang-10.0.1/release/lib/clang/10.0.1/include/__clang_cuda_device_functions.h:1603:19: note: candidate function
__DEVICE__ double log(double __a) { return __nv_log(__a); }
                  ^
/usr/tce/packages/clang/clang-10.0.1/release/lib/clang/10.0.1/include/__clang_cuda_cmath.h:140:18: note: candidate function
__DEVICE__ float log(float __x) { return ::logf(__x); }
                 ^
In file included from reproducer-openmptarget-issue3.cpp:8:
In file included from /usr/tce/packages/gcc/gcc-4.9.3/lib64/gcc/powerpc64le-unknown-linux-gnu/4.9.3/../../../../include/c++/4.9.3/random:51:
/usr/tce/packages/gcc/gcc-4.9.3/lib64/gcc/powerpc64le-unknown-linux-gnu/4.9.3/../../../../include/c++/4.9.3/bits/random.tcc:3474:46: error: call to 'log' is ambiguous
      const size_t __log2r = std::log(__r) / std::log(2.0L);
                                             ^~~~~~~~
/usr/tce/packages/clang/clang-10.0.1/release/lib/clang/10.0.1/include/__clang_cuda_device_functions.h:1603:19: note: candidate function
__DEVICE__ double log(double __a) { return __nv_log(__a); }
                  ^
/usr/tce/packages/clang/clang-10.0.1/release/lib/clang/10.0.1/include/__clang_cuda_cmath.h:140:18: note: candidate function
__DEVICE__ float log(float __x) { return ::logf(__x); }
                 ^
In file included from reproducer-openmptarget-issue3.cpp:8:
In file included from /usr/tce/packages/gcc/gcc-4.9.3/lib64/gcc/powerpc64le-unknown-linux-gnu/4.9.3/../../../../include/c++/4.9.3/random:51:
/usr/tce/packages/gcc/gcc-4.9.3/lib64/gcc/powerpc64le-unknown-linux-gnu/4.9.3/../../../../include/c++/4.9.3/bits/random.tcc:3474:20: error: default initialization of an object of const type 'const std::size_t' (aka 'const unsigned long')
      const size_t __log2r = std::log(__r) / std::log(2.0L);
                   ^
/usr/tce/packages/gcc/gcc-4.9.3/lib64/gcc/powerpc64le-unknown-linux-gnu/4.9.3/../../../../include/c++/4.9.3/bits/random.h:190:16: note: in instantiation of function template specialization 'std::generate_canonical<double, 53, std::linear_congruential_engine<unsigned long, 16807, 0, 2147483647> >' requested here
          return std::generate_canonical<_DInputType,
                      ^
/usr/tce/packages/gcc/gcc-4.9.3/lib64/gcc/powerpc64le-unknown-linux-gnu/4.9.3/../../../../include/c++/4.9.3/bits/random.h:1985:12: note: in instantiation of member function 'std::__detail::_Adaptor<std::linear_congruential_engine<unsigned long, 16807, 0, 2147483647>, double>::operator()' requested here
          return (__aurng() * (__p.b() - __p.a())) + __p.a();
                  ^
/usr/tce/packages/gcc/gcc-4.9.3/lib64/gcc/powerpc64le-unknown-linux-gnu/4.9.3/../../../../include/c++/4.9.3/bits/random.h:1976:24: note: in instantiation of function template specialization 'std::uniform_real_distribution<double>::operator()<std::linear_congruential_engine<unsigned long, 16807, 0, 2147483647> >' requested here
        { return this->operator()(__urng, _M_param); }
                       ^
reproducer-openmptarget-issue3.cpp:27:21: note: in instantiation of function template specialization 'std::uniform_real_distribution<double>::operator()<std::linear_congruential_engine<unsigned long, 16807, 0, 2147483647> >' requested here
  double dval = dist(gen);
                    ^
7 errors generated.
 */

#include <random>

int main(int, char **)
{
  std::default_random_engine gen;
  std::uniform_real_distribution<double> dist(0.0, 1.0);

  double dval = dist(gen);

  return (dval > 1.0 || dval < 0.0) ? 1 : 0;
}
