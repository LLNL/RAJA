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

// "Message Passing" litmus test for DESUL ordered atomic
// ------------------------------------------------------
// Initial state: x = 0 && y = 0
//
//  Thread 1:        Thread 2:
// -----------      -----------
// store(x, 1)      a = load(flag)
// store(flag, 1)   b = load(x)
//
// Allowed results:
// ----------------
// Strong behaviors:
//  - a = 1, b = 1
//  - a = 0, b = 0
//  - a = 0, b = 1
// Weak behavior:
//  - a = 1, b = 0
//
// On weak architectures (POWER/ARM/GPUs), the store to "x" can be reordered
// after the store to "flag". Store-release and load-acquire on the "flag"
// variable should prevent observing the weak behavior.

// Send policy: Relaxed (Weak), Acquire, AcqRel, SeqCst
// Recv policy: Relaxed (Weak), Release, AcqRel, SeqCst
template <typename T, typename SendPolicy, typename RecvPolicy>
struct MessagePassingLitmus {
  using RelaxedPolicy = RAJA::atomic_relaxed;
  size_t m_size;
  T *x;
  T *flag;
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
    flag = work_res.allocate<T>(size);
    a = work_res.allocate<T>(size);
    b = work_res.allocate<T>(size);
  }

  void deallocate(camp::resources::Resource work_res)
  {
    work_res.deallocate(x);
    work_res.deallocate(flag);
    work_res.deallocate(a);
    work_res.deallocate(b);
  }

  void pre_run(camp::resources::Resource work_res)
  {
    work_res.memset(x, 0, sizeof(T) * m_size);
    work_res.memset(flag, 0, sizeof(T) * m_size);
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
    // Send action
    x[other_thread] = T{1};
    RAJA::atomicExchange<SendPolicy>(&(flag[other_thread]), T{1});
    // Recv action
    a[this_thread] = RAJA::atomicAdd<RecvPolicy>(&(flag[this_thread]), T{0});
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
      if (a_host[i] == 0 && b_host[i] == 0) {
        // Strong behavior: neither store from test_send is observable
        strong_behavior_0++;
      } else if (a_host[i] == 1 && b_host[i] == 1) {
        // Strong behavior: both stores from test_send are observable
        strong_behavior_1++;
      } else if (a_host[i] == 0 && b_host[i] == 1) {
        // Strong behavior: stores interleaved with receives
        interleaved_behavior++;
      } else if (a_host[i] == 1 && b_host[i] == 0) {
        // Weak behavior: second store observed before first store
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
    std::cout << " - Strong behavior (a = 0, b = 0) = " << strong_behavior_0
              << "\n";
    std::cout << " - Strong behavior (a = 1, b = 1) = " << strong_behavior_1
              << "\n";
    std::cout << " - Strong behavior (a = 0, b = 1) = " << interleaved_behavior
              << "\n";
    std::cout << " - Weak behaviors = " << weak_behavior << "\n";

    if (std::is_same<SendPolicy, RAJA::atomic_relaxed>::value &&
        std::is_same<RecvPolicy, RAJA::atomic_relaxed>::value) {
      // In the relaxed case, we should observe some weak behaviors.
      ASSERT_GT(weak_behavior, 0);
    } else {
      // We should not expect any weak behaviors if using a strong ordering.
      EXPECT_EQ(weak_behavior, 0);
    }
  }
};

using MPLitmusTestOrderPols =
    camp::list<camp::list<RAJA::atomic_relaxed, RAJA::atomic_relaxed>,
               camp::list<RAJA::atomic_release, RAJA::atomic_acquire>,
               camp::list<RAJA::atomic_acq_rel, RAJA::atomic_acq_rel>,
               camp::list<RAJA::atomic_seq_cst, RAJA::atomic_seq_cst> >;

using MPLitmusTestRecvPols = camp::list<RAJA::atomic_relaxed>;

using MPLitmusTestPols =
    camp::cartesian_product<AtomicDataTypeList, MPLitmusTestOrderPols>;
