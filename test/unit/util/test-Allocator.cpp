//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-21, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

///
/// Source file containing unit tests for Allocator class
///

#include "RAJA_test-base.hpp"

#include "RAJA/util/Allocator.hpp"
#include "RAJA/util/AllocatorPool.hpp"
#include "RAJA/util/AllocatorUmpire.hpp"


template < typename Resource >
struct ResourceAllocator
{
  ResourceAllocator(Resource const& res) : m_res(res) { }
  const char* getName() const noexcept
  {
    return "ResourceAllocator";
  }
  RAJA::Platform getPlatform() noexcept
  {
    return m_res.get_platform();
  }
  void* allocate(size_t nbytes)
  {
    return m_res.calloc(nbytes);
  }
  void deallocate(void* ptr)
  {
    m_res.deallocate(ptr);
  }
private:
  Resource m_res;
};


void AllocatorUnitTestExistingAllocator(RAJA::Allocator& aloc) {
  RAJA::Allocator const& caloc = aloc;

  ASSERT_TRUE(!caloc.getName().empty());

  ASSERT_NE(aloc.getPlatform(), RAJA::Platform::undefined);

  const size_t initial_highWatermark   = caloc.getHighWatermark();
  const size_t initial_currentSize     = caloc.getCurrentSize();
  const size_t initial_actualSize      = caloc.getActualSize();
  const size_t initial_allocationCount = caloc.getAllocationCount();

  ASSERT_GE(initial_highWatermark,   0u);
  ASSERT_EQ(initial_currentSize,     0u);
  ASSERT_GE(initial_actualSize,      0u);
  ASSERT_EQ(initial_allocationCount, 0u);

  {
    void* ptr = aloc.allocate(1);

    ASSERT_NE(ptr, nullptr);

    ASSERT_GE(caloc.getHighWatermark(),   initial_highWatermark);
    ASSERT_EQ(caloc.getCurrentSize(),     initial_currentSize + 1u);
    ASSERT_GE(caloc.getActualSize(),      initial_actualSize);
    ASSERT_EQ(caloc.getAllocationCount(), initial_allocationCount + 1u);

    aloc.deallocate(ptr);
  }

  ASSERT_GE(caloc.getHighWatermark(),   initial_highWatermark);
  ASSERT_EQ(caloc.getCurrentSize(),     initial_currentSize);
  ASSERT_GE(caloc.getActualSize(),      initial_actualSize);
  ASSERT_EQ(caloc.getAllocationCount(), initial_allocationCount);

  {
    double* ptr = aloc.template allocate<double>(1);

    ASSERT_NE(ptr, nullptr);

    ASSERT_GE(caloc.getHighWatermark(),   initial_highWatermark);
    ASSERT_EQ(caloc.getCurrentSize(),     initial_currentSize + sizeof(double));
    ASSERT_GE(caloc.getActualSize(),      initial_actualSize);
    ASSERT_EQ(caloc.getAllocationCount(), initial_allocationCount + 1u);

    aloc.deallocate(ptr);
  }

  ASSERT_GE(caloc.getHighWatermark(),   initial_highWatermark);
  ASSERT_EQ(caloc.getCurrentSize(),     initial_currentSize);
  ASSERT_GE(caloc.getActualSize(),      initial_actualSize);
  ASSERT_EQ(caloc.getAllocationCount(), initial_allocationCount);

  const size_t preRelease_highWatermark   = caloc.getHighWatermark();

  aloc.release();

  ASSERT_EQ(caloc.getHighWatermark(),   preRelease_highWatermark);
  ASSERT_EQ(caloc.getCurrentSize(),     0u);
  ASSERT_EQ(caloc.getActualSize(),      0u);
  ASSERT_EQ(caloc.getAllocationCount(), 0u);
}

void AllocatorUnitTestNewAllocator(RAJA::Allocator& aloc, RAJA::Platform platform) {
  RAJA::Allocator const& caloc = aloc;

  ASSERT_TRUE(!caloc.getName().empty());

  ASSERT_EQ(aloc.getPlatform(), platform);

  const size_t initial_highWatermark   = caloc.getHighWatermark();
  const size_t initial_currentSize     = caloc.getCurrentSize();
  const size_t initial_actualSize      = caloc.getActualSize();
  const size_t initial_allocationCount = caloc.getAllocationCount();

  ASSERT_EQ(initial_highWatermark,   0u);
  ASSERT_EQ(initial_currentSize,     0u);
  ASSERT_EQ(initial_actualSize,      0u);
  ASSERT_EQ(initial_allocationCount, 0u);

  {
    void* ptr = aloc.allocate(1u);

    ASSERT_NE(ptr, nullptr);

    ASSERT_EQ(caloc.getHighWatermark(),   initial_highWatermark + 1u);
    ASSERT_EQ(caloc.getCurrentSize(),     initial_currentSize + 1u);
    ASSERT_GE(caloc.getActualSize(),      initial_actualSize + 1u);
    ASSERT_EQ(caloc.getAllocationCount(), initial_allocationCount + 1u);

    aloc.deallocate(ptr);
  }

  ASSERT_EQ(caloc.getHighWatermark(),   initial_highWatermark + 1u);
  ASSERT_EQ(caloc.getCurrentSize(),     initial_currentSize);
  ASSERT_GE(caloc.getActualSize(),      initial_actualSize + 1u);
  ASSERT_EQ(caloc.getAllocationCount(), initial_allocationCount);

  {
    double* ptr = aloc.template allocate<double>(1);

    ASSERT_NE(ptr, nullptr);

    ASSERT_EQ(caloc.getHighWatermark(),   initial_highWatermark + sizeof(double));
    ASSERT_EQ(caloc.getCurrentSize(),     initial_currentSize + sizeof(double));
    ASSERT_GE(caloc.getActualSize(),      initial_actualSize + sizeof(double));
    ASSERT_EQ(caloc.getAllocationCount(), initial_allocationCount + 1u);

    aloc.deallocate(ptr);
  }

  ASSERT_EQ(caloc.getHighWatermark(),   initial_highWatermark + sizeof(double));
  ASSERT_EQ(caloc.getCurrentSize(),     initial_currentSize);
  ASSERT_GE(caloc.getActualSize(),      initial_actualSize + sizeof(double));
  ASSERT_EQ(caloc.getAllocationCount(), initial_allocationCount);

  const size_t preRelease_highWatermark   = caloc.getHighWatermark();

  aloc.release();

  ASSERT_EQ(caloc.getHighWatermark(),   preRelease_highWatermark);
  ASSERT_EQ(caloc.getCurrentSize(),     0u);
  ASSERT_EQ(caloc.getActualSize(),      0u);
  ASSERT_EQ(caloc.getAllocationCount(), 0u);
}

TEST(AllocatorUnitTest, get_allocators)
{
  std::vector<RAJA::Allocator*> allocators = RAJA::get_allocators();

  for (RAJA::Allocator* aloc : allocators) {

    AllocatorUnitTestExistingAllocator(*aloc);

  }
}

#if defined(RAJA_ENABLE_CUDA)

TEST(AllocatorUnitTest, allocators_CUDA)
{
  AllocatorUnitTestExistingAllocator(RAJA::cuda::get_device_allocator());
  RAJA::cuda::set_device_allocator<
        RAJA::AllocatorPool<ResourceAllocator<RAJA::resources::Cuda>>>(
      RAJA::resources::Cuda::get_default());
  AllocatorUnitTestNewAllocator(RAJA::cuda::get_device_allocator(),
                                RAJA::Platform::cuda);
  RAJA::cuda::reset_device_allocator();
  AllocatorUnitTestNewAllocator(RAJA::cuda::get_device_allocator(),
                                RAJA::Platform::cuda);

  AllocatorUnitTestExistingAllocator(RAJA::cuda::get_pinned_allocator());
  RAJA::cuda::set_pinned_allocator<
        RAJA::AllocatorPool<ResourceAllocator<RAJA::resources::Cuda>>>(
      RAJA::resources::Cuda::get_default());
  AllocatorUnitTestNewAllocator(RAJA::cuda::get_pinned_allocator(),
                                RAJA::Platform::cuda);
  RAJA::cuda::reset_pinned_allocator();
  AllocatorUnitTestNewAllocator(RAJA::cuda::get_pinned_allocator(),
                                RAJA::Platform::cuda);

  AllocatorUnitTestExistingAllocator(RAJA::cuda::get_device_zeroed_allocator());
  RAJA::cuda::set_device_zeroed_allocator<
        RAJA::AllocatorPool<ResourceAllocator<RAJA::resources::Cuda>>>(
      RAJA::resources::Cuda::get_default());
  AllocatorUnitTestNewAllocator(RAJA::cuda::get_device_zeroed_allocator(),
                                RAJA::Platform::cuda);
  RAJA::cuda::reset_device_zeroed_allocator();
  AllocatorUnitTestNewAllocator(RAJA::cuda::get_device_zeroed_allocator(),
                                RAJA::Platform::cuda);
}

#endif  // if defined(RAJA_ENABLE_CUDA)

#if defined(RAJA_ENABLE_HIP)

TEST(AllocatorUnitTest, allocators_HIP)
{
  AllocatorUnitTestExistingAllocator(RAJA::hip::get_device_allocator());
  RAJA::hip::set_device_allocator<
        RAJA::AllocatorPool<ResourceAllocator<RAJA::resources::Hip>>>(
      RAJA::resources::Hip::get_default());
  AllocatorUnitTestNewAllocator(RAJA::hip::get_device_allocator(),
                                RAJA::Platform::hip);
  RAJA::hip::reset_device_allocator();
  AllocatorUnitTestNewAllocator(RAJA::hip::get_device_allocator(),
                                RAJA::Platform::hip);

  AllocatorUnitTestExistingAllocator(RAJA::hip::get_pinned_allocator());
  RAJA::hip::set_pinned_allocator<
        RAJA::AllocatorPool<ResourceAllocator<RAJA::resources::Hip>>>(
      RAJA::resources::Hip::get_default());
  AllocatorUnitTestNewAllocator(RAJA::hip::get_pinned_allocator(),
                                RAJA::Platform::hip);
  RAJA::hip::reset_pinned_allocator();
  AllocatorUnitTestNewAllocator(RAJA::hip::get_pinned_allocator(),
                                RAJA::Platform::hip);

  AllocatorUnitTestExistingAllocator(RAJA::hip::get_device_zeroed_allocator());
  RAJA::hip::set_device_zeroed_allocator<
        RAJA::AllocatorPool<ResourceAllocator<RAJA::resources::Hip>>>(
      RAJA::resources::Hip::get_default());
  AllocatorUnitTestNewAllocator(RAJA::hip::get_device_zeroed_allocator(),
                                RAJA::Platform::hip);
  RAJA::hip::reset_device_zeroed_allocator();
  AllocatorUnitTestNewAllocator(RAJA::hip::get_device_zeroed_allocator(),
                                RAJA::Platform::hip);
}

#endif  // if defined(RAJA_ENABLE_HIP)
