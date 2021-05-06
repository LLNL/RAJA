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


TEST(AllocatorUnitTest, get_allocators)
{
  std::vector<RAJA::Allocator*> allocators = RAJA::get_allocators();

  for (RAJA::Allocator* aloc : allocators) {
    const RAJA::Allocator* caloc = aloc;

    ASSERT_TRUE(!caloc->getName().empty());

    ASSERT_NE(aloc->getPlatform(), RAJA::Platform::undefined);

    const size_t initial_highWatermark   = caloc->getHighWatermark();
    const size_t initial_currentSize     = caloc->getCurrentSize();
    const size_t initial_actualSize      = caloc->getActualSize();
    const size_t initial_allocationCount = caloc->getAllocationCount();

    ASSERT_GE(initial_highWatermark,   0u);
    ASSERT_EQ(initial_currentSize,     0u);
    ASSERT_GE(initial_actualSize,      0u);
    ASSERT_EQ(initial_allocationCount, 0u);

    {
      void* ptr = aloc->allocate(1);

      ASSERT_NE(ptr, nullptr);

      ASSERT_GE(caloc->getHighWatermark(),   initial_highWatermark);
      ASSERT_EQ(caloc->getCurrentSize(),     initial_currentSize + 1u);
      ASSERT_GE(caloc->getActualSize(),      initial_actualSize);
      ASSERT_EQ(caloc->getAllocationCount(), initial_allocationCount + 1u);

      aloc->deallocate(ptr);
    }

    ASSERT_GE(caloc->getHighWatermark(),   initial_highWatermark);
    ASSERT_EQ(caloc->getCurrentSize(),     initial_currentSize);
    ASSERT_GE(caloc->getActualSize(),      initial_actualSize);
    ASSERT_EQ(caloc->getAllocationCount(), initial_allocationCount);

    {
      double* ptr = aloc->template allocate<double>(1);

      ASSERT_NE(ptr, nullptr);

      ASSERT_GE(caloc->getHighWatermark(),   initial_highWatermark);
      ASSERT_EQ(caloc->getCurrentSize(),     initial_currentSize + sizeof(double));
      ASSERT_GE(caloc->getActualSize(),      initial_actualSize);
      ASSERT_EQ(caloc->getAllocationCount(), initial_allocationCount + sizeof(double));

      aloc->deallocate(ptr);
    }

    ASSERT_GE(caloc->getHighWatermark(),   initial_highWatermark);
    ASSERT_EQ(caloc->getCurrentSize(),     initial_currentSize);
    ASSERT_GE(caloc->getActualSize(),      initial_actualSize);
    ASSERT_EQ(caloc->getAllocationCount(), initial_allocationCount);

    const size_t preRelease_highWatermark   = caloc->getHighWatermark();

    aloc->release();

    ASSERT_EQ(caloc->getHighWatermark(),   preRelease_highWatermark);
    ASSERT_EQ(caloc->getCurrentSize(),     0u);
    ASSERT_EQ(caloc->getActualSize(),      0u);
    ASSERT_EQ(caloc->getAllocationCount(), 0u);
  }
}
