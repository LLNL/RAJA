
// define that removes calls to exit in error handlers
#define RAJA_LOGGER_CUDA_TESTING

#include "gtest/gtest.h"

#include "RAJA/RAJA.hxx"
#include "RAJA/internal/defines.hxx"

#include <stdlib.h>

#include "type_helper.hxx"

// taken from test-scan.exe
using ExecTypes = std::tuple<RAJA::cuda_exec<128>, RAJA::seq_exec, RAJA::simd_exec>;

using PrintfTypes = std::tuple<char,
                               signed char,
                               short,
                               int,
                               long,
                               long long,
                               unsigned char,
                               unsigned short,
                               unsigned int,
                               unsigned long,
                               unsigned long long,
                               // intmax_t,
                               // uintmax_t,
                               size_t,
                               ptrdiff_t,
                               float,
                               double>; // compiler chokes with more than 16 types

template <typename T1, typename T2>
using Cross = typename types::product<T1, T2>;

using Types = Cross<ExecTypes, PrintfTypes>::type;

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
const char* s_fmt = nullptr;

TYPED_TEST_CASE_P(LoggerTest);

template < typename T, typename ExecPolicy, typename LoggerPolicy >
void forall_test(RAJA::Index_type array_length,
                 RAJA::Real_ptr test_array,
                 RAJA::Index_type small,
                 RAJA::Index_type small_count)
{
  small_counter.store(0);

  const char* fmt = nullptr;
  if (std::is_same<char, T>::value) {
    fmt = "%c";
    s_fmt = fmt;
  } else if (std::is_same<signed char, T>::value) {
    fmt = "%hhi";
    s_fmt = fmt;
  } else if (std::is_same<short, T>::value) {
    fmt = "%hi";
    s_fmt = fmt;
  } else if (std::is_same<int, T>::value) {
    fmt = "%i";
    s_fmt = fmt;
  } else if (std::is_same<long, T>::value) {
    fmt = "%li";
    s_fmt = fmt;
  } else if (std::is_same<long long, T>::value) {
    fmt = "%lli";
    s_fmt = fmt;
  } else if (std::is_same<intmax_t, T>::value) {
    fmt = "%ji";
    s_fmt = fmt;
  } else if (std::is_same<ptrdiff_t, T>::value) {
    fmt = "%ti";
    s_fmt = fmt;
  } else if (std::is_same<unsigned char, T>::value) {
    fmt = "%hhu";
    s_fmt = fmt;
  } else if (std::is_same<unsigned short, T>::value) {
    fmt = "%hu";
    s_fmt = fmt;
  } else if (std::is_same<unsigned int, T>::value) {
    fmt = "%u";
    s_fmt = fmt;
  } else if (std::is_same<unsigned long, T>::value) {
    fmt = "%lu";
    s_fmt = fmt;
  } else if (std::is_same<unsigned long long, T>::value) {
    fmt = "%llu";
    s_fmt = fmt;
  } else if (std::is_same<uintmax_t, T>::value) {
    fmt = "%ju";
    s_fmt = fmt;
  } else if (std::is_same<size_t, T>::value) {
    fmt = "%zi";
    s_fmt = fmt;
  } else if (std::is_same<float, T>::value) {
    fmt = "%.10e";
    s_fmt = "%e";
  } else if (std::is_same<double, T>::value) {
    fmt = "%.16le";
    s_fmt = "%le";
  } else if (std::is_same<long double, T>::value) {
    fmt = "%.22Le";
    s_fmt = "%Le";
  } else if (std::is_pointer<T>::value) {
    fmt = "%p";
    s_fmt = fmt;
  } else {
    ASSERT_TRUE(false);
  }

  RAJA::Logger<LoggerPolicy> mylog([](int udata, const char* msg) {
    if (msg != nullptr) {
      T multiplier = std::is_floating_point<T>::value ? 3.14159265358979323846 : 1;
      T val = udata * multiplier;
      T msg_val = static_cast<T>(-1);
      int ns = sscanf(msg, s_fmt, &msg_val);
      if (ns != 1 && std::is_same<T, char>::value) {
        msg_val = msg[0]; // case where scanf can't read null char
      }
      if (val == msg_val) {
        small_counter++;
      } else {
        printf("udata = %i, val = %.16e, msg_val (%s) = %.16e\n", udata, (double)val, msg, (double)msg_val);
      }
    }
  });

  RAJA::forall<ExecPolicy>(0, array_length, [=] __host__ __device__ (RAJA::Index_type idx) {
    T multiplier = std::is_floating_point<T>::value ? 3.14159265358979323846 : 1;
    T val = idx * multiplier;
    if (test_array[idx] <= small) {
      mylog.log(idx, fmt, val);
    } else if (test_array[idx] < 0) {
      mylog.error(idx, fmt, val);
    }
  });

  RAJA::check_logs();

  EXPECT_EQ(small_counter.load(), small_count);
}


TYPED_TEST_P(LoggerTest, BasicForall)
{
  using ExecPolicy = typename std::tuple_element<0, TypeParam>::type;
  using type = typename std::tuple_element<1, TypeParam>::type;

  using LoggerPolicy = RAJA::cuda_logger;

  forall_test<type, ExecPolicy, LoggerPolicy>(
      this->array_length, this->test_array,
      this->small, this->small_count);
}

REGISTER_TYPED_TEST_CASE_P(LoggerTest, BasicForall);

INSTANTIATE_TYPED_TEST_CASE_P(Logger, LoggerTest, CrossTypes);
