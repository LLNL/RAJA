//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-24, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

//
// test/include headers
//
#include "RAJA_test-atomic-types.hpp"
#include "RAJA_test-base.hpp"
#include "RAJA_test-camp.hpp"
#include "RAJA_test-forall-data.hpp"
#include "RAJA_test-forall-execpol.hpp"
#include "RAJA_test-index-types.hpp"

// "Store buffer" litmus test for DESUL ordered atomic
// ---------------------------------------------------
// Initial state: x = 0 && y = 0
//
//  Thread 1:        Thread 2:
// -----------      -----------
// store(x, 1)      store(y, 1)
// a = load(y)      b = load(x)
//
// Allowed results:
// ----------------
// Strong behaviors:
//  - a = 1, b = 1
//  - a = 0, b = 1
//  - a = 1, b = 0
// Weak behavior:
//  - a = 0, b = 0
//
// Acquire-release semantics are not enough to disallow the stores to be
// reordered after the load -- full sequential consistency is required in order
// to impose a "single total order" of the stores.

// Send policy: Relaxed, SeqCst (Strong)
// Recv policy: Relaxed, SeqCst (Strong)
template <typename T, typename AtomicPolicy>
struct StoreBufferLitmus {
  using RelaxedPolicy = RAJA::atomic_relaxed;
  size_t m_size;
  T *x;
  T *y;
  T *a;
  T *b;

  int strong_behavior_0{0};
  int strong_behavior_1{0};
  int interleaved_behavior{0};
  int weak_behavior{0};

  void allocate(camp::resources::Resource work_res, size_t size)
  {
    m_size = size;
    x = work_res.allocate<T>(size);
    y = work_res.allocate<T>(size);
    a = work_res.allocate<T>(size);
    b = work_res.allocate<T>(size);
  }

  void deallocate(camp::resources::Resource work_res)
  {
    work_res.deallocate(x);
    work_res.deallocate(y);
    work_res.deallocate(a);
    work_res.deallocate(b);
  }

  void pre_run(camp::resources::Resource work_res)
  {
    work_res.memset(x, 0, sizeof(T) * m_size);
    work_res.memset(y, 0, sizeof(T) * m_size);
    work_res.memset(a, 0, sizeof(T) * m_size);
    work_res.memset(b, 0, sizeof(T) * m_size);

#if defined(RAJA_ENABLE_CUDA)
    cudaErrchk(cudaDeviceSynchronize());
#endif

#if defined(RAJA_ENABLE_HIP)
    hipErrchk(hipDeviceSynchronize());
#endif
  }

  RAJA_HOST_DEVICE void run(int this_thread, int other_thread)
  {
    // Store-buffer 1
    RAJA::atomicAdd<AtomicPolicy>(&(x[other_thread]), T{1});
    // a[other_thread] = RAJA::atomicAdd<AtomicPolicy>(&(y[other_thread]),
    // T{0});
    a[other_thread] = y[other_thread];
    // Store-buffer 2
    RAJA::atomicAdd<AtomicPolicy>(&(y[this_thread]), T{1});
    b[this_thread] = x[this_thread];
  }

  void count_results(camp::resources::Resource work_res)
  {
    camp::resources::Resource host_res{camp::resources::Host()};

    T *a_host = host_res.allocate<T>(m_size);
    T *b_host = host_res.allocate<T>(m_size);

    work_res.memcpy(a_host, a, m_size * sizeof(T));
    work_res.memcpy(b_host, b, m_size * sizeof(T));

#if defined(RAJA_ENABLE_CUDA)
    cudaErrchk(cudaDeviceSynchronize());
#endif

#if defined(RAJA_ENABLE_HIP)
    hipErrchk(hipDeviceSynchronize());
#endif
    for (size_t i = 0; i < m_size; i++) {
      if (a_host[i] == 1 && b_host[i] == 0) {
        // Strong behavior: thread 1 happened before thread 2
        strong_behavior_0++;
      } else if (a_host[i] == 0 && b_host[i] == 1) {
        // Strong behavior: thread 2 happened before thread 1
        strong_behavior_1++;
      } else if (a_host[i] == 1 && b_host[i] == 1) {
        // Strong behavior: stores interleaved with receives
        interleaved_behavior++;
      } else if (a_host[i] == 0 && b_host[i] == 0) {
        // Weak behavior: stores reordered after receives
        weak_behavior++;
      } else {
        FAIL() << "Unexpected result for index " << i;
      }
    }

    host_res.deallocate(a_host);
    host_res.deallocate(b_host);
  }

  void verify()
  {
    std::cout << " - Strong behavior (a = 1, b = 0) = " << strong_behavior_0
              << "\n";
    std::cout << " - Strong behavior (a = 0, b = 1) = " << strong_behavior_1
              << "\n";
    std::cout << " - Strong behavior (a = 1, b = 1) = " << interleaved_behavior
              << "\n";
    std::cout << " - Weak behaviors = " << weak_behavior << "\n";

    if (std::is_same<AtomicPolicy, RAJA::atomic_relaxed>::value) {
      // In the relaxed case, we should observe some weak behaviors.
      ASSERT_GT(weak_behavior, 0);
    } else {
      // We should not expect any weak behaviors if using a strong ordering.
      EXPECT_EQ(weak_behavior, 0);
    }
  }
};

using SBLitmusTestOrderPols =
    camp::list<RAJA::atomic_relaxed, RAJA::atomic_seq_cst>;

using SBLitmusTestPols =
    camp::cartesian_product<AtomicDataTypeList, SBLitmusTestOrderPols>;
