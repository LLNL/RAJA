//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

///
/// Source file containing hip global var reference in device lambda issue reproducer.
///
/// Immediate Issue: Using a variable that is a local reference to a global variable
///                  causes compilation errors when used inside of a device lambda.
///
/// Larger Issue: Some codes using RAJA cannot compile with hip.
///
/// Seen when compiling with hip 3.6.0.
///

#include "RAJA/RAJA.hpp"
#include <cassert>
#include <cstdio>

namespace global {
int val = 0;
}

template < typename Kernel >
__global__ void call_kernel(Kernel kernel)
{
   kernel();
}

int main(int, char**)
{
   int& val = global::val;

   std::printf("reproducer-global-var-reference\n");

   int* output;
   assert(hipHostMalloc((void**)&output, sizeof(int)) == hipSuccess);

   //
   // Attempt to sort using insertion sort
   // This fails and clobbers the vals array
   //
   std::printf("Before setting output\n");

   output[0] = 1;
   std::printf("  %i %i\n", val, output[0]);

   call_kernel<<<1,1>>>([=] __host__ __device__() {
      output[0] = val;
   });
   assert(hipDeviceSynchronize() == hipSuccess);

   std::printf("After setting output\n");

   int wrong_val = (output[0] != val);
   std::printf("  %i %i\n", val, output[0]);

   if (wrong_val) {
      std::printf("Output got wrong answer.\n");
   } else {
      std::printf("Output got correct answer.\n");
   }

   return wrong_val;
}
