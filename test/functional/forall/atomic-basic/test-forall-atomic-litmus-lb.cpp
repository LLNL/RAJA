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

// "Load buffer" litmus test for DESUL ordered atomic
// --------------------------------------------------
// Initial state: x = 0 && y = 0
//
//  Thread 1:        Thread 2:
// -----------      -----------
// a = load(y)      b = load(x)
// store(x, 1)      store(y, 1)
//
// Allowed results:
// ----------------
// Strong behaviors:
//  - a = 0, b = 1
//  - a = 1, b = 0
//  - a = 0, b = 0
// Weak behavior:
//  - a = 1, b = 1

// Send policy: Relaxed (Weak), Release, AcqRel, SeqCst
// Recv policy: Relaxed (Weak), Acquire, AcqRel, SeqCst
template <typename T, typename SendPolicy, typename RecvPolicy>
struct LoadBufferLitmus {
  using DataType = T;
  using RelaxedPolicy = RAJA::atomic_relaxed;
  constexpr static int PERMUTE_THREAD_FLAG = 97;
  size_t m_size;
  int m_stride;
  T *x;
  T *y;
  T *a;
  T *b;

  int strong_behavior_0{0};
  int strong_behavior_1{0};
  int interleaved_behavior{0};
  int weak_behavior{0};

  void allocate(camp::resources::Resource work_res, size_t size, int stride)
  {
    m_size = size;
    m_stride = stride;
    x = work_res.allocate<T>(size * stride);
    y = work_res.allocate<T>(size * stride);
    a = work_res.allocate<T>(size * stride);
    b = work_res.allocate<T>(size * stride);
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
    work_res.memset(x, 0, sizeof(T) * m_size * m_stride);
    work_res.memset(y, 0, sizeof(T) * m_size * m_stride);
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
    bool swap = this_thread % 2 == 0;
    if (swap) {
      load_buffer_1(other_thread, iter);
      load_buffer_2(this_thread, iter);
    } else {
      load_buffer_2(this_thread, iter);
      load_buffer_1(other_thread, iter);
    }
  }

  RAJA_HOST_DEVICE void load_buffer_1(int thread, int iter)
  {
    int thread_idx = thread * m_stride + iter;
    int permute_thread = (thread * PERMUTE_THREAD_FLAG) % m_size;
    int permute_idx = permute_thread * m_stride + iter;
    a[thread_idx] = RAJA::atomicAdd<RelaxedPolicy>(&(y[thread_idx]), T{0});
    RAJA::atomicAdd<SendPolicy>(&(x[permute_idx]), T{1});
  }

  RAJA_HOST_DEVICE void load_buffer_2(int thread, int iter)
  {
    int thread_idx = thread * m_stride + iter;
    int permute_thread = (thread * PERMUTE_THREAD_FLAG) % m_size;
    int permute_idx = permute_thread * m_stride + iter;
    b[thread_idx] = RAJA::atomicAdd<RecvPolicy>(&(x[permute_idx]), T{0});
    RAJA::atomicAdd<RelaxedPolicy>(&(y[thread_idx]), T{1});
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
      if (a_local[i] == 1 && b_local[i] == 0) {
        // Strong behavior: thread 2 happened before thread 1
        strong_cnt_0 += 1;
      } else if (a_local[i] == 0 && b_local[i] == 1) {
        // Strong behavior: thread 1 happened before thread 2
        strong_cnt_1 += 1;
      } else if (a_local[i] == 0 && b_local[i] == 0) {
        // Strong behavior: stores interleaved with receives
        interleaved_cnt += 1;
      } else if (a_local[i] == 1 && b_local[i] == 1) {
        // Weak behavior: stores reordered after receives
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
    std::cerr << " - Strong behavior (a = 1, b = 0) = " << strong_behavior_0
              << "\n";
    std::cerr << " - Strong behavior (a = 0, b = 1) = " << strong_behavior_1
              << "\n";
    std::cerr << " - Strong behavior (a = 1, b = 1) = " << interleaved_behavior
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

using LBLitmusTestOrderPols =
    camp::list<camp::list<RAJA::atomic_relaxed, RAJA::atomic_relaxed>,
               camp::list<RAJA::atomic_release, RAJA::atomic_acquire>,
               camp::list<RAJA::atomic_acq_rel, RAJA::atomic_acq_rel>,
               camp::list<RAJA::atomic_seq_cst, RAJA::atomic_seq_cst> >;

using LBLitmusTestPols =
    camp::cartesian_product<AtomicDataTypeList, LBLitmusTestOrderPols>;

TYPED_TEST_SUITE_P(ForallAtomicLitmusTest);

template <typename T>
class ForallAtomicLitmusTest : public ::testing::Test
{
};

TYPED_TEST_P(ForallAtomicLitmusTest, LoadBufferTest)
{
  using Type = typename camp::at<TypeParam, camp::num<0>>::type;
  using SendRecvPol = typename camp::at<TypeParam, camp::num<1>>::type;
  using SendPol = typename camp::at<SendRecvPol, camp::num<0>>::type;
  using RecvPol = typename camp::at<SendRecvPol, camp::num<1>>::type;

  using LBTest = LoadBufferLitmus<Type, SendPol, RecvPol>;
  LitmusTestDriver<LBTest>::run();
}

REGISTER_TYPED_TEST_SUITE_P(ForallAtomicLitmusTest, LoadBufferTest);

using LoadBufferTestTypes = Test<LBLitmusTestPols>::Types;

INSTANTIATE_TYPED_TEST_SUITE_P(Hip,
                               ForallAtomicLitmusTest,
                               LoadBufferTestTypes);
