#include "gtest/gtest.h"

#include "RAJA/RAJA.hxx"
#include "RAJA/MemUtils_CPU.hxx"

#include <stdlib.h>

template <typename T>
class LoggerTest : public ::testing::Test
{
protected:
  virtual void SetUp()
  {
    array_length = 3153;
    small = 7548;
    small_count = 0;

    test_array = (RAJA::Real_ptr) RAJA::allocate_aligned(RAJA::DATA_ALIGN,
                   array_length * sizeof(RAJA::Real_type));

    for (RAJA::Index_type i = 0; i < array_length; ++i) {
      test_array[i] = RAJA::Real_type(rand() % 65536);
    }

    for (RAJA::Index_type i = 0; i < array_length; ++i) {
      if (test_array[i] <= small) {
        small_count++;
      }
    }

    RAJA::Internal::s_exit_enabled = false;
  }

  virtual void TearDown()
  {
    RAJA::Internal::s_exit_enabled = true;

    RAJA::free_aligned(test_array);
  }

  RAJA::Real_ptr test_array;
  RAJA::Index_type array_length;
  RAJA::Index_type small;
  RAJA::Index_type small_count;
};

// it wouls be nice to encapsulate this
std::atomic<RAJA::Index_type> log_counter;
std::atomic<RAJA::Index_type> err_counter;

TYPED_TEST_CASE_P(LoggerTest);

TYPED_TEST_P(LoggerTest, BasicForall)
{
  using ExecPolicy = typename std::tuple_element<0, TypeParam>::type;
  using LoggerPolicy = typename std::tuple_element<1, TypeParam>::type;

  log_counter.store(0);
  err_counter.store(0);

  RAJA::Logger<LoggerPolicy> mylog([](int udata, const char* msg) {
    log_counter++;
    EXPECT_EQ(udata, atoi(msg));
  });

  RAJA::Logger<LoggerPolicy> myerr([](int udata, const char* msg) {
    err_counter++;
    EXPECT_EQ(udata, atoi(msg));
  });

  RAJA::forall<ExecPolicy>(0, this->array_length, [=](RAJA::Index_type idx) {
    if (this->test_array[idx] <= this->small) {
      mylog.log(idx, "%i", idx);
    }
    if (this->test_array[idx] <= this->small) {
      myerr.error(idx, "%i", idx);
    }
  });

  RAJA::check_logs();

  EXPECT_EQ(log_counter.load(), this->small_count);
  EXPECT_GT(err_counter.load(), 0);
}

REGISTER_TYPED_TEST_CASE_P(LoggerTest, BasicForall);


typedef ::testing::Types<
          std::tuple<RAJA::seq_exec, RAJA::seq_logger>,
          std::tuple<RAJA::simd_exec, RAJA::seq_logger> >
        SequentialTypes;

INSTANTIATE_TYPED_TEST_CASE_P(Sequential, LoggerTest, SequentialTypes);


#if defined(RAJA_ENABLE_OPENMP)
typedef ::testing::Types<
          std::tuple<RAJA::omp_parallel_for_exec, RAJA::omp_logger> >
        OpenMPTypes;

INSTANTIATE_TYPED_TEST_CASE_P(OpenMP, LoggerTest, OpenMPTypes);
#endif

#if defined(RAJA_ENABLE_CILK)
typedef ::testing::Types<
          std::tuple<RAJA::cilk_for_exec, RAJA::cilk_logger> >
        CilkTypes;

INSTANTIATE_TYPED_TEST_CASE_P(Cilk, LoggerTest, CilkTypes);
#endif
