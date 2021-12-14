//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-21, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

///
/// Source file containing hip indirect device function call issue reproducer.
///
/// Immediate Issue: Indirect device function calls produce incorrect results.
///
/// Larger Issue: RAJA workgroup unordered tests fail, applications get wrong results.
///
/// Seen when compiling for gfx908 on ElCap EA systems (AMD CPU, AMD MI100)
///
/// Note removing dependence on threadIdx.x in device_function gives correct
/// results even with much more complex expressions.
///

#include <cassert>
#include <cstdio>
#include <hip/hip_runtime.h>

using function_ptr = void(*)(int*, int);

__device__ void device_function(int* ptr, int val)
{
   ptr[threadIdx.x] = val;
}

__global__ void set_device_function_ptr(function_ptr* func_ptr_ptr)
{
   *func_ptr_ptr = &device_function;
}

function_ptr get_device_function_pointer()
{
   function_ptr* func_ptr_ptr;
   assert(hipHostMalloc(&func_ptr_ptr, sizeof(function_ptr)) == hipSuccess);
   *func_ptr_ptr = nullptr;

   set_device_function_ptr<<<1,1>>>(func_ptr_ptr);
   assert(hipGetLastError() == hipSuccess);
   assert(hipDeviceSynchronize() == hipSuccess);

   function_ptr func_ptr = *func_ptr_ptr;

   assert(hipHostFree(func_ptr_ptr) == hipSuccess);

   assert(func_ptr != nullptr);

   return func_ptr;
}

__global__ void call_device_function_pointer(function_ptr func_ptr, int* int_ptr, int val)
{
   func_ptr(int_ptr, val);
}

int main(int, char**)
{
   int blocksize = 1;

   function_ptr func_ptr = get_device_function_pointer();
   assert(func_ptr != nullptr);

   int* int_ptr;
   assert(hipHostMalloc((void**)&int_ptr, blocksize*sizeof(int)) == hipSuccess);
   for (int i = 0; i < blocksize; ++i) {
      int_ptr[i] = 0;
   }

   dim3 blockSize(blocksize);
   dim3 gridSize(1);
   call_device_function_pointer<<<gridSize,blockSize>>>(func_ptr, int_ptr, 1);
   assert(hipGetLastError() == hipSuccess);
   assert(hipDeviceSynchronize() == hipSuccess);

   int wrong = 0;
   for (int i = 0; i < blocksize; ++i) {
      if (int_ptr[i] != 1) {
         wrong = 1;
         printf("Error at int_ptr[%i] expected %i got %i.\n",
                i, 1+i, int_ptr[i]);
      }
   }

   assert(hipHostFree(int_ptr) == hipSuccess);

   if (wrong) {
      printf("Got wrong answer.\n");
   } else {
      printf("Got correct answer.\n");
   }

   return wrong;
}
