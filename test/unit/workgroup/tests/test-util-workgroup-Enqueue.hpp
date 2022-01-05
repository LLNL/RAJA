//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-22, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

///
/// Header file containing tests for RAJA workgroup enqueue.
///

#ifndef __TEST_UTIL_WORKGROUP_ENQUEUE__
#define __TEST_UTIL_WORKGROUP_ENQUEUE__

#include "RAJA_test-workgroup.hpp"

#include <random>


template < typename IndexType,
           typename ... Args >
struct EnqueueTestCallable
{
  EnqueueTestCallable(IndexType* _ptr, IndexType _val)
    : ptr(_ptr)
    , val(_val)
  { }

  EnqueueTestCallable(EnqueueTestCallable const&) = default;
  EnqueueTestCallable& operator=(EnqueueTestCallable const&) = default;

  EnqueueTestCallable(EnqueueTestCallable&& o) = default;
  EnqueueTestCallable& operator=(EnqueueTestCallable&& o) = default;

  RAJA_HOST_DEVICE void operator()(IndexType i, Args... args) const
  {
    RAJA_UNUSED_VAR(args...);
    ptr[i] = val;
  }

private:
  IndexType* ptr;
  IndexType  val;
};

#endif  //__TEST_UTIL_WORKGROUP_ENQUEUE__
