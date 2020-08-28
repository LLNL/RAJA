//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

///
/// Source file containing hip insertion sort pairs issue reproducer.
///
/// Immediate Issue: Insertion sort with zip iterator gives wrong result in values.
///
/// Larger Issue: RAJA insertion sort hip test fails.
///
/// Seen when compiling with hip 3.6.0.
///

#include "RAJA/RAJA.hpp"
#include <cassert>
#include <cstdio>

__global__ void call_insertion_sort_pairs(double* keys_begin, double* keys_end, int* vals_begin)
{
   auto begin = RAJA::zip(keys_begin, vals_begin);
   auto end = RAJA::zip(keys_end, vals_begin+(keys_end-keys_begin));
   using zip_ref = RAJA::detail::IterRef<camp::decay<decltype(begin)>>;
   RAJA::operators::less<double> comp{};
   RAJA::insertion_sort(begin, end, RAJA::compare_first<zip_ref>(comp));
}

__global__ void call_shell_sort_pairs(double* keys_begin, double* keys_end, int* vals_begin)
{
   auto begin = RAJA::zip(keys_begin, vals_begin);
   auto end = RAJA::zip(keys_end, vals_begin+(keys_end-keys_begin));
   using zip_ref = RAJA::detail::IterRef<camp::decay<decltype(begin)>>;
   RAJA::operators::less<double> comp{};
   RAJA::shell_sort(begin, end, RAJA::compare_first<zip_ref>(comp));
}

int main(int, char**)
{
   std::printf("reproducer-insertion-sort-pairs\n");

   int N = 4;

   std::printf("N = %i\n", N);

   double* keys;
   assert(hipHostMalloc((void**)&keys, N*sizeof(double)) == hipSuccess);

   int* vals;
   assert(hipHostMalloc((void**)&vals, N*sizeof(int)) == hipSuccess);

   //
   // Attempt to sort using insertion sort
   // This fails and clobbers the vals array
   //
   std::printf("Before Insertion Sort\n");

   for (int i = 0; i < N; ++i) {
      keys[i] = N-i-1.0;
      vals[i] = N-i-1;
      std::printf("  %2i (%2.0f, %2i)\n", i, keys[i], vals[i]);
   }

   call_insertion_sort_pairs<<<1,1>>>(keys, keys+N, vals);
   assert(hipDeviceSynchronize() == hipSuccess);

   std::printf("After Insertion Sort\n");

   int wrong_insert = 0;
   for (int i = 0; i < N; ++i) {
      std::printf("  %2i (%2.0f, %2i)\n", i, keys[i], vals[i]);
      if (keys[i] != static_cast<double>(i)) wrong_insert = 1;
      if (vals[i] != i) wrong_insert = 1;
   }

   if (wrong_insert) {
      std::printf("Insertion sort got wrong answer.\n");
   } else {
      std::printf("Insertion sort got correct answer.\n");
   }

   //
   // Attempt to sort using shell sort
   // This succeeds despite shell sort using insertion sort when n <= 4
   //
   std::printf("Before Shell Sort\n");

   for (int i = 0; i < N; ++i) {
      keys[i] = N-i-1.0;
      vals[i] = N-i-1;
      std::printf("  %2i (%2.0f, %2i)\n", i, keys[i], vals[i]);
   }

   call_shell_sort_pairs<<<1,1>>>(keys, keys+N, vals);
   assert(hipDeviceSynchronize() == hipSuccess);

   std::printf("After Shell Sort\n");

   int wrong_shell = 0;
   for (int i = 0; i < N; ++i) {
      std::printf("  %2i (%2.0f, %2i)\n", i, keys[i], vals[i]);
      if (keys[i] != static_cast<double>(i)) wrong_shell = 1;
      if (vals[i] != i) wrong_shell = 1;
   }

   assert(hipHostFree(vals) == hipSuccess);
   assert(hipHostFree(keys) == hipSuccess);

   if (wrong_shell) {
      std::printf("Shell sort got wrong answer.\n");
   } else {
      std::printf("Shell sort got correct answer.\n");
   }

   return wrong_insert;
}
