#include "gtest/gtest.h"

#include "RAJA/RAJA.hxx"

#include <stdlib.h>

template <typename T>
class LoggerTest : public ::testing::Test
{
protected:
  virtual void SetUp()
  {
    array_length = 513;
    small = 14235;
    small_count = 0;

    cudaMallocManaged(&test_array, array_length * sizeof(RAJA::Real_type));

    for (RAJA::Index_type i = 0; i < array_length; ++i) {
      test_array[i] = RAJA::Real_type(rand() % 65536);
    }

    for (RAJA::Index_type i = 0; i < array_length; ++i) {
      if (test_array[i] <= small) {
        small_count++;
      }
    }
  }

  virtual void TearDown()
  {
    cudaFree(test_array);
  }

  RAJA::Real_ptr test_array;
  RAJA::Index_type array_length;
  RAJA::Index_type small;
  RAJA::Index_type small_count;
};

// it wouls be nice to encapsulate this
std::atomic<RAJA::Index_type> small_counter;

std::atomic<bool> error_flag;

TYPED_TEST_CASE_P(LoggerTest);

template < typename TypeParam >
void forall_test(RAJA::Index_type array_length,
                 RAJA::Real_ptr test_array,
                 RAJA::Index_type small)
{
  using ExecPolicy = typename std::tuple_element<0, TypeParam>::type;
  using LoggerPolicy = typename std::tuple_element<1, TypeParam>::type;

  RAJA::Logger<LoggerPolicy> mylog([](int udata, const char* msg) {
    small_counter++;
    fprintf(stderr, "RAJA logger: udata %i, msg %p.\n", udata, msg);
    fprintf(stderr, "RAJA logger: msg %s.\n",  msg);
    if (msg == nullptr || udata != atoi(msg)) {
      error_flag.store(true);
    }
  });

  RAJA::forall<ExecPolicy>(0, array_length, [=] __host__ __device__ (RAJA::Index_type idx) {
    if (test_array[idx] <= small) {
      mylog.log(idx, "%i", idx);
    } else if (test_array[idx] < 0) {
      mylog.error(idx, "%i", idx);
    }
  });

  RAJA::check_logs();
}

TYPED_TEST_P(LoggerTest, BasicForall)
{
  using ExecPolicy = typename std::tuple_element<0, TypeParam>::type;
  using LoggerPolicy = typename std::tuple_element<1, TypeParam>::type;

  small_counter.store(0);
  error_flag.store(false);

  forall_test<TypeParam>(this->array_length,
                         this->test_array,
                         this->small);

  EXPECT_EQ(small_counter.load(), this->small_count);
  EXPECT_FALSE(error_flag.load());

  if (std::is_same<ExecPolicy, RAJA::cuda_exec<128> >::value) {
    // exit(0);
  }
}

REGISTER_TYPED_TEST_CASE_P(LoggerTest, BasicForall);


typedef ::testing::Types<
          std::tuple<RAJA::seq_exec, RAJA::cuda_logger>,
          std::tuple<RAJA::simd_exec, RAJA::cuda_logger> >
        SequentialTypes;

INSTANTIATE_TYPED_TEST_CASE_P(Sequential, LoggerTest, SequentialTypes);


#if defined(RAJA_ENABLE_CUDA)
typedef ::testing::Types<
          std::tuple<RAJA::cuda_exec<128>, RAJA::cuda_logger> >
        CudaTypes;

INSTANTIATE_TYPED_TEST_CASE_P(Cuda, LoggerTest, CudaTypes);
#endif