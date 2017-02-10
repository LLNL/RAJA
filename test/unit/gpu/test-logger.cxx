
#include "gtest/gtest.h"

#include "RAJA/RAJA.hxx"
#include "RAJA/internal/defines.hxx"

#include <stdlib.h>

#include "type_helper.hxx"

// taken from test-scan.exe
using TestTypes = std::tuple<std::true_type, std::false_type>;

using ExecTypes = std::tuple<RAJA::cuda_exec<128>, RAJA::seq_exec, RAJA::simd_exec>;

using PrintfTypes = std::tuple<char,
                               // signed char,
                               // short,
                               int,
                               long,
                               // long long,
                               // unsigned char,
                               // unsigned short,
                               unsigned int,
                               unsigned long,
                               // unsigned long long,
                               // intmax_t,
                               // uintmax_t,
                               // size_t,
                               // ptrdiff_t,
                               float,
                               double>; // compiler chokes with too many types

template <typename T1, typename T2>
using Cross = typename types::product<T1, T2>;

using Types = Cross<Cross<TestTypes, ExecTypes>::type, PrintfTypes>::type;

template <typename T>
struct ForTesting {
};

template <typename... Ts>
struct ForTesting<std::tuple<Ts...>> {
  using type = testing::Types<Ts...>;
};

using CrossTypes = ForTesting<Types>::type;


template <typename T>
class LoggerTest : public ::testing::Test
{
protected:
  virtual void SetUp()
  {
    array_length = 123153;
    small = 7548;
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

    RAJA::Internal::s_exit_enabled = false;
  }

  virtual void TearDown()
  {
    RAJA::Internal::s_exit_enabled = true;

    cudaFree(test_array);
  }

  RAJA::Real_ptr test_array;
  RAJA::Index_type array_length;
  RAJA::Index_type small;
  RAJA::Index_type small_count;
};

// it wouls be nice to encapsulate this
std::atomic<RAJA::Index_type> log_counter;
std::atomic<RAJA::Index_type> err_counter;
const char* s_fmt = nullptr;

#define FMT_EXTRA " %s, %d strange error *&%% {}[]()\t\v\nhi %p\10\101.\n\1"
#define FMT_EXTRA_VALUES , "hi", 20500, ((void*)0x1d1ea98f7cUL)


TYPED_TEST_CASE_P(LoggerTest);

template < typename T, typename ExecPolicy, typename LoggerPolicy >
void log_test(RAJA::Index_type array_length,
                 RAJA::Real_ptr test_array,
                 RAJA::Index_type small,
                 RAJA::Index_type small_count)
{
  log_counter.store(0);

  const char* fmt = nullptr;
  if (std::is_same<char, T>::value) {
    fmt = "%c" FMT_EXTRA;
    s_fmt = fmt;
  } else if (std::is_same<signed char, T>::value) {
    fmt = "%hhi" FMT_EXTRA;
    s_fmt = fmt;
  } else if (std::is_same<short, T>::value) {
    fmt = "%hi" FMT_EXTRA;
    s_fmt = fmt;
  } else if (std::is_same<int, T>::value) {
    fmt = "%i" FMT_EXTRA;
    s_fmt = fmt;
  } else if (std::is_same<long, T>::value) {
    fmt = "%li" FMT_EXTRA;
    s_fmt = fmt;
  } else if (std::is_same<long long, T>::value) {
    fmt = "%lli" FMT_EXTRA;
    s_fmt = fmt;
  } else if (std::is_same<intmax_t, T>::value) {
    fmt = "%ji" FMT_EXTRA;
    s_fmt = fmt;
  } else if (std::is_same<ptrdiff_t, T>::value) {
    fmt = "%ti" FMT_EXTRA;
    s_fmt = fmt;
  } else if (std::is_same<unsigned char, T>::value) {
    fmt = "%hhu" FMT_EXTRA;
    s_fmt = fmt;
  } else if (std::is_same<unsigned short, T>::value) {
    fmt = "%hu" FMT_EXTRA;
    s_fmt = fmt;
  } else if (std::is_same<unsigned int, T>::value) {
    fmt = "%u" FMT_EXTRA;
    s_fmt = fmt;
  } else if (std::is_same<unsigned long, T>::value) {
    fmt = "%lu" FMT_EXTRA;
    s_fmt = fmt;
  } else if (std::is_same<unsigned long long, T>::value) {
    fmt = "%llu" FMT_EXTRA;
    s_fmt = fmt;
  } else if (std::is_same<uintmax_t, T>::value) {
    fmt = "%ju" FMT_EXTRA;
    s_fmt = fmt;
  } else if (std::is_same<size_t, T>::value) {
    fmt = "%zi" FMT_EXTRA;
    s_fmt = fmt;
  } else if (std::is_same<float, T>::value) {
    fmt = "%.10e" FMT_EXTRA;
    s_fmt = fmt;
  } else if (std::is_same<double, T>::value) {
    fmt = "%.16le" FMT_EXTRA;
    s_fmt = fmt;
  } else if (std::is_same<long double, T>::value) {
    fmt = "%.22Le" FMT_EXTRA;
    s_fmt = fmt;
  } else if (std::is_pointer<T>::value) {
    fmt = "%p" FMT_EXTRA;
    s_fmt = fmt;
  } else {
    ASSERT_TRUE(false);
  }

  RAJA::Logger<LoggerPolicy> mylog([](int udata, const char* msg) {
    if (msg != nullptr) {
      char msg_check[512];
      T multiplier = std::is_floating_point<T>::value ? 3.14159265358979323846 : 1;
      T val = udata * multiplier;
      sprintf(msg_check, s_fmt, val FMT_EXTRA_VALUES);
      if (strcmp(msg, msg_check) == 0) {
        log_counter++;
      } else {
        printf("udata = %i, msg %s, msg_check %s\n", udata, msg, msg_check);
      }
    }
  });

  RAJA::forall<ExecPolicy>(0, array_length, [=] __host__ __device__ (RAJA::Index_type idx) {
    T multiplier = std::is_floating_point<T>::value ? 3.14159265358979323846 : 1;
    T val = idx * multiplier;
    if (test_array[idx] <= small) {
      mylog.log(idx, fmt, val FMT_EXTRA_VALUES);
    } else if (test_array[idx] < 0) {
      mylog.error(idx, fmt, val FMT_EXTRA_VALUES);
    }
  });

  RAJA::check_logs();

  EXPECT_EQ(log_counter.load(), small_count);
}

#define ERROR_FMT_EXTRA
#define ERROR_FMT_EXTRA_VALUES

template < typename T, typename ExecPolicy, typename LoggerPolicy >
void error_test(RAJA::Index_type array_length,
                 RAJA::Real_ptr test_array,
                 RAJA::Index_type small,
                 RAJA::Index_type small_count)
{
  log_counter.store(0);
  err_counter.store(0);

  const char* fmt = nullptr;
  if (std::is_same<char, T>::value) {
    fmt = "%c" ERROR_FMT_EXTRA;
    s_fmt = fmt;
  } else if (std::is_same<signed char, T>::value) {
    fmt = "%hhi" ERROR_FMT_EXTRA;
    s_fmt = fmt;
  } else if (std::is_same<short, T>::value) {
    fmt = "%hi" ERROR_FMT_EXTRA;
    s_fmt = fmt;
  } else if (std::is_same<int, T>::value) {
    fmt = "%i" ERROR_FMT_EXTRA;
    s_fmt = fmt;
  } else if (std::is_same<long, T>::value) {
    fmt = "%li" ERROR_FMT_EXTRA;
    s_fmt = fmt;
  } else if (std::is_same<long long, T>::value) {
    fmt = "%lli" ERROR_FMT_EXTRA;
    s_fmt = fmt;
  } else if (std::is_same<intmax_t, T>::value) {
    fmt = "%ji" ERROR_FMT_EXTRA;
    s_fmt = fmt;
  } else if (std::is_same<ptrdiff_t, T>::value) {
    fmt = "%ti" ERROR_FMT_EXTRA;
    s_fmt = fmt;
  } else if (std::is_same<unsigned char, T>::value) {
    fmt = "%hhu" ERROR_FMT_EXTRA;
    s_fmt = fmt;
  } else if (std::is_same<unsigned short, T>::value) {
    fmt = "%hu" ERROR_FMT_EXTRA;
    s_fmt = fmt;
  } else if (std::is_same<unsigned int, T>::value) {
    fmt = "%u" ERROR_FMT_EXTRA;
    s_fmt = fmt;
  } else if (std::is_same<unsigned long, T>::value) {
    fmt = "%lu" ERROR_FMT_EXTRA;
    s_fmt = fmt;
  } else if (std::is_same<unsigned long long, T>::value) {
    fmt = "%llu" ERROR_FMT_EXTRA;
    s_fmt = fmt;
  } else if (std::is_same<uintmax_t, T>::value) {
    fmt = "%ju" ERROR_FMT_EXTRA;
    s_fmt = fmt;
  } else if (std::is_same<size_t, T>::value) {
    fmt = "%zi" ERROR_FMT_EXTRA;
    s_fmt = fmt;
  } else if (std::is_same<float, T>::value) {
    fmt = "%.10e" ERROR_FMT_EXTRA;
    s_fmt = fmt;
  } else if (std::is_same<double, T>::value) {
    fmt = "%.16le" ERROR_FMT_EXTRA;
    s_fmt = fmt;
  } else if (std::is_same<long double, T>::value) {
    fmt = "%.22Le" ERROR_FMT_EXTRA;
    s_fmt = fmt;
  } else if (std::is_pointer<T>::value) {
    fmt = "%p" ERROR_FMT_EXTRA;
    s_fmt = fmt;
  } else {
    ASSERT_TRUE(false);
  }

  RAJA::Logger<LoggerPolicy> mylog([](int udata, const char* msg) {
    if (msg != nullptr) {
      char msg_check[512];
      T multiplier = std::is_floating_point<T>::value ? 3.14159265358979323846 : 1;
      T val = udata * multiplier;
      sprintf(msg_check, s_fmt, val ERROR_FMT_EXTRA_VALUES);
      if (strcmp(msg, msg_check) == 0) {
        log_counter++;
      } else {
        printf("log udata = %i, msg %s, msg_check %s\n", udata, msg, msg_check);
      }
    }
  });

  RAJA::Logger<LoggerPolicy> myerr([](int udata, const char* msg) {
    if (msg != nullptr) {
      char msg_check[512];
      T multiplier = std::is_floating_point<T>::value ? 3.14159265358979323846 : 1;
      T val = udata * multiplier;
      sprintf(msg_check, s_fmt, val ERROR_FMT_EXTRA_VALUES);
      if (strcmp(msg, msg_check) == 0) {
        err_counter++;
      } else {
        printf("err udata = %i, msg %s, msg_check %s\n", udata, msg, msg_check);
      }
    }
  });

  RAJA::forall<ExecPolicy>(0, array_length, [=] __host__ __device__ (RAJA::Index_type idx) {
    T multiplier = std::is_floating_point<T>::value ? 3.14159265358979323846 : 1;
    T val = idx * multiplier;
    if (test_array[idx] <= small) {
      mylog.log(idx, fmt, val ERROR_FMT_EXTRA_VALUES);
    }
    if (test_array[idx] <= small) {
      myerr.error(idx, fmt, val ERROR_FMT_EXTRA_VALUES);
    }
  });

  RAJA::check_logs();

  EXPECT_EQ(log_counter.load(), small_count);
  EXPECT_GT(err_counter.load(), 0);
}

TYPED_TEST_P(LoggerTest, BasicForall)
{
  using TestPolicy = typename std::tuple_element<0, TypeParam>::type;
  using ExecPolicy = typename std::tuple_element<1, TypeParam>::type;
  using type = typename std::tuple_element<2, TypeParam>::type;

  using LoggerPolicy = RAJA::cuda_logger;

  if (TestPolicy()) {
    log_test<type, ExecPolicy, LoggerPolicy>(
        this->array_length, this->test_array,
        this->small, this->small_count);
  } else {
    error_test<type, ExecPolicy, LoggerPolicy>(
        this->array_length, this->test_array,
        this->small, this->small_count);
  }
}

REGISTER_TYPED_TEST_CASE_P(LoggerTest, BasicForall);

INSTANTIATE_TYPED_TEST_CASE_P(Logger, LoggerTest, CrossTypes);

int s_num_found = 0;

void get_logs()
{
  RAJA::Internal::s_exit_enabled = false;

  s_num_found = 0;

  RAJA::Logger<RAJA::cuda_logger> err_check([](int udata, const char* msg) {
    EXPECT_EQ(udata, 89416);
    EXPECT_EQ(s_num_found++, 1);
  });

  RAJA::Logger<RAJA::cuda_logger> log_check([](int udata, const char* msg) {
    EXPECT_EQ(udata, 15163);
    EXPECT_EQ(s_num_found++, 0);
  });

  RAJA::forall<RAJA::cuda_exec<128>>(0, 1, [=] __host__ __device__ (int i) {
    log_check.log(15163, "");
    err_check.error(89416, "");
    ((int*)0xffffffffffffffff)[0] = ((int*)0)[0]; // generate error
  });

  RAJA::check_logs();

  ASSERT_EQ(s_num_found, 2);

  RAJA::Internal::s_exit_enabled = true;

  RAJA::Internal::CudaLogManager::deallocateInstance();
  cudaDeviceReset();
}

TEST(ErrorTest, GetLogs)
{
  get_logs();
}
