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

#include <random>
#include <vector>
#include <algorithm>
#include <utility>

namespace
{

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


inline unsigned get_random_seed()
{
  static unsigned seed = std::random_device{}();
  return seed;
}


void AllocatorUnitTestAllocatorAllocation(RAJA::Allocator& aloc) {

  std::mt19937 rng(get_random_seed());
  using dist_type = std::uniform_int_distribution<size_t>;

  const size_t initial_highWatermark   = aloc.getHighWatermark();
  const size_t initial_currentSize     = aloc.getCurrentSize();
  const size_t initial_actualSize      = aloc.getActualSize();
  const size_t initial_allocationCount = aloc.getAllocationCount();

  ASSERT_GE(initial_highWatermark,   0u);
  ASSERT_EQ(initial_currentSize,     0u);
  ASSERT_GE(initial_actualSize,      0u);
  ASSERT_EQ(initial_allocationCount, 0u);

  size_t highWatermark   = initial_highWatermark;
  size_t currentSize     = initial_currentSize;
  size_t allocationCount = initial_allocationCount;

  size_t remaining_allocations = dist_type(0u, 1024u)(rng);

  std::vector<std::pair<void*, size_t>> allocations;
  allocations.reserve(remaining_allocations);

  while (remaining_allocations > 0u || !allocations.empty()) {

    bool allocate   = false;
    bool deallocate = false;
    if (remaining_allocations > 0u && !allocations.empty()) {
      if (dist_type(0u, 1u)(rng) == 0u) {
        allocate = true;
      } else {
        deallocate = true;
      }
    } else if (remaining_allocations > 0u) {
      allocate = true;
    } else if (!allocations.empty()) {
      deallocate = true;
    }

    if (allocate) {
      size_t allocation_size = dist_type(1u, 1024u)(rng);
      switch (dist_type(0u, 2u)(rng)) {
        case 0u:
        {
          allocations.emplace_back(aloc.allocate(allocation_size), allocation_size);
          break;
        }
        case 1u:
        {
          allocations.emplace_back(aloc.allocate<int>(allocation_size), allocation_size*sizeof(int));
          allocation_size *= sizeof(int);
          break;
        }
        case 2u:
        {
          allocations.emplace_back(aloc.allocate<double>(allocation_size), allocation_size*sizeof(double));
          allocation_size *= sizeof(double);
          break;
        }
        default:
        {
          ASSERT_TRUE(false);
        }
      }
      remaining_allocations -= 1u;

      currentSize     += allocation_size;
      highWatermark    = std::max(highWatermark, currentSize);
      allocationCount += 1u;

      ASSERT_EQ(aloc.getHighWatermark(),   highWatermark);
      ASSERT_EQ(aloc.getCurrentSize(),     currentSize);
      ASSERT_GE(aloc.getActualSize(),      currentSize);
      ASSERT_EQ(aloc.getAllocationCount(), allocationCount);
    }

    if (deallocate) {
      size_t allocation_id = dist_type(0u, allocations.size()-1u)(rng);
      void* ptr = allocations[allocation_id].first;
      size_t allocation_size = allocations[allocation_id].second;
      if (allocation_id != allocations.size()-1u) {
        allocations[allocation_id] = allocations[allocations.size()-1u];
      }
      aloc.deallocate(ptr);
      allocations.erase(--allocations.end());

      currentSize     -= allocation_size;
      allocationCount -= 1u;

      ASSERT_EQ(aloc.getHighWatermark(),   highWatermark);
      ASSERT_EQ(aloc.getCurrentSize(),     currentSize);
      ASSERT_GE(aloc.getActualSize(),      currentSize);
      ASSERT_EQ(aloc.getAllocationCount(), allocationCount);
    }

  }

  aloc.release();

  ASSERT_EQ(aloc.getHighWatermark(),   highWatermark);
  ASSERT_EQ(aloc.getCurrentSize(),     initial_currentSize);
  ASSERT_GE(aloc.getActualSize(),      initial_currentSize);
  ASSERT_EQ(aloc.getAllocationCount(), initial_allocationCount);
}

} // end namespace


void AllocatorUnitTestExistingAllocator(RAJA::Allocator& aloc) {

  ASSERT_TRUE(!aloc.getName().empty());

  ASSERT_NE(aloc.getPlatform(), RAJA::Platform::undefined);

  AllocatorUnitTestAllocatorAllocation(aloc);
}

void AllocatorUnitTestNewAllocator(RAJA::Allocator& aloc, RAJA::Platform platform) {
  RAJA::Allocator const& caloc = aloc;

  ASSERT_TRUE(!caloc.getName().empty());

  ASSERT_EQ(aloc.getPlatform(), platform);

  ASSERT_EQ(caloc.getHighWatermark(),   0u);
  ASSERT_EQ(caloc.getCurrentSize(),     0u);
  ASSERT_EQ(caloc.getActualSize(),      0u);
  ASSERT_EQ(caloc.getAllocationCount(), 0u);

  AllocatorUnitTestAllocatorAllocation(aloc);
}

TEST(AllocatorUnitTest, get_allocators)
{
  std::vector<RAJA::Allocator*> allocators = RAJA::get_allocators();

  for (RAJA::Allocator* aloc : allocators) {

    AllocatorUnitTestExistingAllocator(*aloc);

  }
}

TEST(AllocatorUnitTest, AllocatorPool)
{
  RAJA::AllocatorPool<ResourceAllocator<RAJA::resources::Host>> aloc(
      RAJA::resources::Host::get_default());
  AllocatorUnitTestNewAllocator(aloc, RAJA::Platform::host);
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
