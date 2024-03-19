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

#include "test-forall-atomic-litmus-driver.hpp"

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
  using DataType = T;
  using RelaxedPolicy = RAJA::atomic_relaxed;
  constexpr static int PERMUTE_THREAD_FLAG = 97;
  size_t m_size;
  int m_stride;
  T *x;
  T *flag;
  T *a;
  T *b;

  size_t strong_behavior_0{0};
  size_t strong_behavior_1{0};
  size_t interleaved_behavior{0};
  size_t weak_behavior{0};

  void allocate(camp::resources::Resource work_res, size_t size, int stride)
  {
    m_size = size;
    m_stride = stride;
    x = work_res.allocate<T>(size * stride);
    flag = work_res.allocate<T>(size * stride);
    a = work_res.allocate<T>(size * stride);
    b = work_res.allocate<T>(size * stride);
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
    work_res.memset(x, 0, sizeof(T) * m_size * m_stride);
    work_res.memset(flag, 0, sizeof(T) * m_size * m_stride);
    work_res.memset(a, 0, sizeof(T) * m_size * m_stride);
    work_res.memset(b, 0, sizeof(T) * m_size * m_stride);

#if defined(RAJA_ENABLE_CUDA)
    cudaErrchk(cudaDeviceSynchronize());
#endif

#if defined(RAJA_ENABLE_HIP)
    hipErrchk(hipDeviceSynchronize());
#endif
  }

  RAJA_HOST_DEVICE void run(int this_thread, int other_thread, int iter)
  {
    bool send_first = (this_thread % 2 == 0);
    // Send action
    if (send_first) {
      this->run_send(other_thread, iter);
      this->run_recv(this_thread, iter);
    } else {
      this->run_recv(this_thread, iter);
      this->run_send(other_thread, iter);
    }
  }

  RAJA_HOST_DEVICE void run_send(int other_thread, int iter)
  {
    int other_thread_idx = other_thread * m_stride + iter;
    int permute_other_thread = (other_thread * PERMUTE_THREAD_FLAG) % m_size;
    int permute_idx = permute_other_thread * m_stride + iter;
    RAJA::atomicAdd<RelaxedPolicy>(&(x[other_thread_idx]), T{1});
    RAJA::atomicAdd<SendPolicy>(&(flag[permute_idx]), T{1});
  }

  RAJA_HOST_DEVICE void run_recv(int this_thread, int iter)
  {
    int this_thread_idx = this_thread * m_stride + iter;
    int permute_this_thread = (this_thread * PERMUTE_THREAD_FLAG) % m_size;
    int permute_idx = permute_this_thread * m_stride + iter;
    a[this_thread_idx] =
        RAJA::atomicAdd<RecvPolicy>(&(flag[permute_idx]), T{0});
    b[this_thread_idx] =
        RAJA::atomicAdd<RelaxedPolicy>(&(x[this_thread_idx]), T{0});
  }

  void count_results(camp::resources::Resource work_res)
  {

#ifdef RAJA_ENABLE_HIP
    using GPUExec = RAJA::hip_exec<256>;
    using ReducePolicy = RAJA::hip_reduce;
#endif

#ifdef RAJA_ENABLE_CUDA
    using GPUExec = RAJA::cuda_exec<256>;
    using ReducePolicy = RAJA::cuda_reduce;
#endif
    RAJA::ReduceSum<ReducePolicy, size_t> strong_cnt_0(0);
    RAJA::ReduceSum<ReducePolicy, size_t> strong_cnt_1(0);
    RAJA::ReduceSum<ReducePolicy, size_t> interleaved_cnt(0);
    RAJA::ReduceSum<ReducePolicy, size_t> weak_cnt(0);
    RAJA::ReduceSum<ReducePolicy, size_t> unexpected_cnt(0);

    T *a_local = a;
    T *b_local = b;

    auto forall_len = RAJA::TypedRangeSegment<int>(0, m_size * m_stride);

    RAJA::forall<GPUExec>(forall_len, [=] RAJA_HOST_DEVICE(int i) {
      if (a_local[i] == 0 && b_local[i] == 0) {
        // Strong behavior: neither store from test_send is observable
        strong_cnt_0 += 1;
      } else if (a_local[i] == 1 && b_local[i] == 1) {
        // Strong behavior: both stores from test_send are observable
        strong_cnt_1 += 1;
      } else if (a_local[i] == 0 && b_local[i] == 1) {
        // Strong behavior: stores interleaved with receives
        interleaved_cnt += 1;
      } else if (a_local[i] == 1 && b_local[i] == 0) {
        // Weak behavior: second store observed before first store
        weak_cnt += 1;
      } else {
        unexpected_cnt += 1;
      }
    });

    EXPECT_EQ(unexpected_cnt.get(), 0);

    strong_behavior_0 += strong_cnt_0.get();
    strong_behavior_1 += strong_cnt_1.get();
    interleaved_behavior += interleaved_cnt.get();
    weak_behavior += weak_cnt.get();
  }

  void verify()
  {
    std::cerr << " - Strong behavior (a = 0, b = 0) = " << strong_behavior_0
              << "\n";
    std::cerr << " - Strong behavior (a = 1, b = 1) = " << strong_behavior_1
              << "\n";
    std::cerr << " - Strong behavior (a = 0, b = 1) = " << interleaved_behavior
              << "\n";
    std::cerr << " - Weak behaviors = " << weak_behavior << "\n";

    if (std::is_same<SendPolicy, RAJA::atomic_relaxed>::value &&
        std::is_same<RecvPolicy, RAJA::atomic_relaxed>::value) {
      // In the relaxed case, we should observe some weak behaviors.
      // Don't fail the test, but do print out a message.
      if (weak_behavior == 0) {
        std::cerr << "Warning - no weak behaviors detected in the control case."
                  << "\nThis litmus test may be insufficient to exercise "
                     "ordered memory atomics.\n";
      } else {
        double overall_behavior_counts = strong_behavior_0 + strong_behavior_1 +
                                         interleaved_behavior + weak_behavior;
        std::cerr << "\n Weak behaviors detected in "
                  << 100 * (weak_behavior / overall_behavior_counts)
                  << "% of cases.\n";
      }
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

using MPLitmusTestPols =
    camp::cartesian_product<AtomicDataTypeList, MPLitmusTestOrderPols>;

TYPED_TEST_SUITE_P(ForallAtomicLitmusTest);

template <typename T>
class ForallAtomicLitmusTest : public ::testing::Test
{
};

TYPED_TEST_P(ForallAtomicLitmusTest, MessagePassingTest)
{
  using Type = typename camp::at<TypeParam, camp::num<0>>::type;
  using SendRecvPol = typename camp::at<TypeParam, camp::num<1>>::type;
  using SendPol = typename camp::at<SendRecvPol, camp::num<0>>::type;
  using RecvPol = typename camp::at<SendRecvPol, camp::num<1>>::type;

  using MPTest = MessagePassingLitmus<Type, SendPol, RecvPol>;
  LitmusTestDriver<MPTest>::run();
}

REGISTER_TYPED_TEST_SUITE_P(ForallAtomicLitmusTest, MessagePassingTest);

using MessagePassingTestTypes = Test<MPLitmusTestPols>::Types;

INSTANTIATE_TYPED_TEST_SUITE_P(Hip,
                               ForallAtomicLitmusTest,
                               MessagePassingTestTypes);
