//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-21, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include <cstdlib>
#include <iostream>
#include <limits>

#include "memoryManager.hpp"

#include "RAJA/RAJA.hpp"

/*
 *  Reduction Example
 *
 *  This example illustrates use of the RAJA Allocators.
 *
 *  RAJA features shown:
 *    -  Allocator types
 *    - `forall` loop iteration template method
 *    -  Index range segment
 *    -  Execution policies
 *    -  Reduction types
 *
 * If CUDA is enabled, CUDA unified memory is used.
 */

/*
  CUDA_BLOCK_SIZE - specifies the number of threads in a CUDA thread block
*/
#if defined(RAJA_ENABLE_CUDA)
const int CUDA_BLOCK_SIZE = 256;
#endif

#if defined(RAJA_ENABLE_HIP)
const int HIP_BLOCK_SIZE = 256;
#endif


// Allocator class derived from RAJA::Allocator
// using memoryManager allocations.
struct ExampleAllocator : RAJA::Allocator
{
  // Allocators may take any constructor args as they are passed through the
  // RAJA::*::set_*_allocator calls to the constructor.
  ExampleAllocator(const std::string& name)
    : m_name(name)
  {
    std::cout << "\t\t" << getName() << " constructor" << std::endl;
  }

  // Virtual destructor.
  // Care should be taken as this may be called after main has returned.
  virtual ~ExampleAllocator()
  {
    std::cout << "\t\t" << getName() << " destructor" << std::endl;
  }

  // Override the allocate method.
  void* allocate(size_t nbytes,
                 size_t alignment) override
  {
    std::cout << "\t\t" << getName() << " allocate nbytes " << nbytes
              << " alignment " << alignment << std::endl;
    void* ptr = memoryManager::allocate<char>(nbytes);
    std::cout << "\t\t" << getName() << "          ptr " << ptr << std::endl;
    return ptr;
  }

  // Override the deallocate method.
  void deallocate(void* ptr) override
  {
    std::cout << "\t\t" << getName() << " deallocate ptr " << ptr << std::endl;
    memoryManager::deallocate(ptr);
  }

  // Override the release method.
  // Used to release memory held by things like pool allocators.
  // Called on the old allocator when changing allocators via
  // RAJA::*::set_*_allocator and RAJA::*::reset_*_allocator calls.
  void release() override
  {
    std::cout << "\t\t" << getName() << " release" << std::endl;
  }

  // override the getName method
  const std::string& getName() const noexcept override
  {
    return m_name;
  }
private:
  std::string m_name;
};


#if defined(RAJA_ENABLE_CUDA) || defined(RAJA_ENABLE_HIP)

// Allocator class derived from RAJA::Allocator
// using memoryManager gpu allocations.
struct ExampleAllocatorGPU : RAJA::Allocator
{
  // Allocators may take any constructor args as they are passed through the
  // RAJA::*::set_*_allocator calls to the constructor.
  ExampleAllocatorGPU(const std::string& name)
    : m_name(name)
  {
    std::cout << "\t\t" << getName() << " constructor" << std::endl;
  }

  // Virtual destructor.
  // Care should be taken as this may be called after main has returned.
  virtual ~ExampleAllocatorGPU()
  {
    std::cout << "\t\t" << getName() << " destructor" << std::endl;
  }

  // Override the allocate method.
  void* allocate(size_t nbytes,
                 size_t alignment) override
  {
    std::cout << "\t\t" << getName() << " allocate nbytes " << nbytes
              << " alignment " << alignment << std::endl;
    void* ptr = memoryManager::allocate_gpu<char>(nbytes);
    std::cout << "\t\t" << getName() << "          ptr " << ptr << std::endl;
    return ptr;
  }

  // Override the deallocate method.
  void deallocate(void* ptr) override
  {
    std::cout << "\t\t" << getName() << " deallocate ptr " << ptr << std::endl;
    memoryManager::deallocate_gpu(ptr);
  }

  // Override the release method.
  // Used to release memory held by things like pool allocators.
  // Called on the old allocator when changing allocators via
  // RAJA::*::set_*_allocator and RAJA::*::reset_*_allocator calls.
  void release() override
  {
    std::cout << "\t\t" << getName() << " release" << std::endl;
  }

  // override the getName method
  const std::string& getName() const noexcept override
  {
    return m_name;
  }
private:
  std::string m_name;
};

#endif


int main(int RAJA_UNUSED_ARG(argc), char** RAJA_UNUSED_ARG(argv[]))
{

  std::cout << "\n\nRAJA allocators example...\n";

//
// Define array length
//
  const int N = 1000000;

//
// Allocate array data and initialize data to alternating sequence of 1, -1.
//
  int* a = memoryManager::allocate<int>(N);

  for (int i = 0; i < N; ++i) {
    if ( i % 2 == 0 ) {
      a[i] = 1;
    } else {
      a[i] = -1;
    }
  }

//
// Note: with this data initialization scheme, the following results will
//       be observed for all reduction kernels below:
//
//  - the sum will be zero
//

//
// Define index range for iterating over a elements in all examples
//
  RAJA::RangeSegment arange(0, N);


//----------------------------------------------------------------------------//
  {
    std::cout << "\n Running RAJA sequential reduction...\n";

    using EXEC_POL1   = RAJA::seq_exec;
    using REDUCE_POL1 = RAJA::seq_reduce;

    RAJA::ReduceSum<REDUCE_POL1, int> seq_sum(0);

    RAJA::forall<EXEC_POL1>(arange, [=](int i) {

      seq_sum += a[i];

    });

    std::cout << "\tsum = " << seq_sum.get() << std::endl;
  }


//----------------------------------------------------------------------------//

#if defined(RAJA_ENABLE_OPENMP)
  {
    std::cout << "\n Running RAJA OpenMP reduction...\n";

    using EXEC_POL2   = RAJA::omp_parallel_for_exec;
    using REDUCE_POL2 = RAJA::omp_reduce;

    RAJA::ReduceSum<REDUCE_POL2, int> omp_sum(0);

    RAJA::forall<EXEC_POL2>(arange, [=](int i) {

      omp_sum += a[i];

    });

    std::cout << "\tsum = " << omp_sum.get() << std::endl;
  }
#endif


//----------------------------------------------------------------------------//

#if defined(RAJA_ENABLE_CUDA)
  {
    std::cout << "\n Setting RAJA CUDA device allocator...\n";

    RAJA::cuda::set_device_allocator<ExampleAllocatorGPU>("CUDA_ExampleAllocatorGPU");

    std::cout << "\n Getting RAJA CUDA device allocator...\n";

    RAJA::Allocator& cuda_device_allocator = RAJA::cuda::get_device_allocator();

    std::cout << "\n Got RAJA CUDA device allocator " << cuda_device_allocator.getName() << "...\n";

    {
      std::cout << "\n Running RAJA CUDA reduction...\n";

      using EXEC_POL3   = RAJA::cuda_exec<CUDA_BLOCK_SIZE>;
      using REDUCE_POL3 = RAJA::cuda_reduce;

      std::cout << "\n Constructing RAJA CUDA reduction object...\n";
      RAJA::ReduceSum<REDUCE_POL3, int> cuda_sum(0);

      std::cout << "\n Running RAJA CUDA reduction kernel...\n";
      RAJA::forall<EXEC_POL3>(arange, [=] RAJA_DEVICE (int i) {

        cuda_sum += a[i];

      });


      std::cout << "\n Getting RAJA CUDA reduction result...\n";

      int result = cuda_sum.get();

      std::cout << "\tsum = " << result << std::endl;
    }

    std::cout << "\n Resetting RAJA CUDA device allocator...\n";

    RAJA::cuda::reset_device_allocator();

    std::cout << "\n Done with RAJA CUDA device allocator...\n";
  }
#endif

//----------------------------------------------------------------------------//

#if defined(RAJA_ENABLE_HIP)
  {
    std::cout << "\n Setting RAJA HIP device allocator...\n";

    RAJA::hip::set_device_allocator<ExampleAllocatorGPU>("HIP_ExampleAllocatorGPU");

    std::cout << "\n Getting RAJA HIP device allocator...\n";

    RAJA::Allocator& hip_device_allocator = RAJA::hip::get_device_allocator();

    std::cout << "\n Got RAJA HIP device allocator " << hip_device_allocator.getName() << "...\n";

    {
      int* d_a = memoryManager::allocate_gpu<int>(N);
      hipErrchk(hipMemcpy( d_a, a, N * sizeof(int), hipMemcpyHostToDevice ));

      using EXEC_POL3   = RAJA::hip_exec<HIP_BLOCK_SIZE>;
      using REDUCE_POL3 = RAJA::hip_reduce;

      std::cout << "\n Constructing RAJA HIP reduction object...\n";
      RAJA::ReduceSum<REDUCE_POL3, int> hip_sum(0);

      std::cout << "\n Running RAJA HIP reduction kernel...\n";
      RAJA::forall<EXEC_POL3>(arange, [=] RAJA_DEVICE (int i) {

        hip_sum += d_a[i];

      });

      std::cout << "\n Getting RAJA HIP reduction result...\n";

      int result = hip_sum.get();

      std::cout << "\tsum = " << result << std::endl;

      memoryManager::deallocate_gpu(d_a);
    }

    std::cout << "\n Resetting RAJA HIP device allocator...\n";

    RAJA::hip::reset_device_allocator();

    std::cout << "\n Done with RAJA HIP device allocator...\n";
  }
#endif

//----------------------------------------------------------------------------//

//
// Clean up.
//
  memoryManager::deallocate(a);

  std::cout << "\n DONE!...\n";

  return 0;
}
