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

// "2+2 Write" litmus test for DESUL ordered atomic
// ---------------------------------------------------
// Initial state: x = 0 && y = 0
//
//  Thread 1:        Thread 2:
// -----------      -----------
// store(x, 1)      store(y, 1)
// store(y, 2)      store(x, 2)
//
// Allowed results:
// ----------------
// Strong behaviors:
//  - a = 1, x = 2
//  - a = 2, x = 1
//  - a = 2, x = 2
// Weak behavior:
//  - a = 1, x = 1

// Send policy: Relaxed (Weak), AcqRel, SeqCst
// Recv policy: Relaxed (Weak), AcqRel, SeqCst
template <typename T, typename AtomicPolicy>
struct Write2x2Litmus {
  using DataType = T;
  using RelaxedPolicy = RAJA::atomic_relaxed;
  size_t m_size;
  int m_stride;
  T *x;
  T *y;
  T *a;

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
  }

  void deallocate(camp::resources::Resource work_res)
  {
    work_res.deallocate(x);
    work_res.deallocate(y);
    work_res.deallocate(a);
  }

  void pre_run(camp::resources::Resource work_res)
  {
    work_res.memset(x, 0, sizeof(T) * m_size * m_stride);
    work_res.memset(y, 0, sizeof(T) * m_size * m_stride);
    work_res.memset(a, 0, sizeof(T) * m_size * m_stride);

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
      store_1(other_thread, iter);
      store_2(this_thread, iter);
    } else {
      store_2(this_thread, iter);
      store_1(other_thread, iter);
    }
  }

  RAJA_HOST_DEVICE void store_1(int thread, int iter)
  {
    int thread_idx = thread * m_stride + iter;
    RAJA::atomicExchange<RelaxedPolicy>(&(x[thread_idx]), T{1});
    RAJA::atomicExchange<AtomicPolicy>(&(y[thread_idx]), T{2});
  }

  RAJA_HOST_DEVICE void store_2(int thread, int iter)
  {
    int thread_idx = thread * m_stride + iter;
    RAJA::atomicExchange<RelaxedPolicy>(&(y[thread_idx]), T{1});
    RAJA::atomicExchange<AtomicPolicy>(&(x[thread_idx]), T{2});
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

    T *x_local = x;
    T *y_local = y;

    auto forall_len = RAJA::TypedRangeSegment<int>(0, m_size * m_stride);

    RAJA::forall<GPUExec>(forall_len, [=] RAJA_HOST_DEVICE(int i) {
      if (x_local[i] == 1 && y_local[i] == 2) {
        // Strong behavior: thread 1 happened before thread 2
        strong_cnt_0 += 1;
      } else if (x_local[i] == 2 && y_local[i] == 1) {
        // Strong behavior: thread 2 happened before thread 1
        strong_cnt_1 += 1;
      } else if (x_local[i] == 2 && y_local[i] == 2) {
        // Strong behavior: interleaved stores in-order
        interleaved_cnt += 1;
      } else if (x_local[i] == 1 && y_local[i] == 1) {
        // Weak behavior: stores on each thread were reordered
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

    if (std::is_same<AtomicPolicy, RAJA::atomic_relaxed>::value) {
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

using Write2x2LitmusTestOrderPols =
    camp::list<RAJA::atomic_relaxed,
               RAJA::atomic_acq_rel,
               RAJA::atomic_seq_cst >;

using Write2x2LitmusTestPols =
    camp::cartesian_product<AtomicDataTypeList, Write2x2LitmusTestOrderPols>;

TYPED_TEST_SUITE_P(ForallAtomicLitmusTest);

template <typename T>
class ForallAtomicLitmusTest : public ::testing::Test
{
};

TYPED_TEST_P(ForallAtomicLitmusTest, Write2x2Test)
{
  using Type = typename camp::at<TypeParam, camp::num<0>>::type;
  using AtomicPol = typename camp::at<TypeParam, camp::num<1>>::type;

  using Write2x2Test = Write2x2Litmus<Type, AtomicPol>;
  LitmusTestDriver<Write2x2Test>::run();
}

REGISTER_TYPED_TEST_SUITE_P(ForallAtomicLitmusTest, Write2x2Test);

using Write2x2TestTypes = Test<Write2x2LitmusTestPols>::Types;

INSTANTIATE_TYPED_TEST_SUITE_P(Hip,
                               ForallAtomicLitmusTest,
                               Write2x2TestTypes);
