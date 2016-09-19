#include <algorithm>
#include <iostream>
#include <iterator>
#include <numeric>
#include <random>
#include <tuple>
#include <type_traits>

#include <cstdlib>

#include <gtest/gtest.h>
#include <RAJA/RAJA.hxx>

#include "data_storage.hxx"
#include "type_helper.hxx"

const int N = 1024;

// Unit Test Space Exploration

using ExecTypes = std::tuple<
#ifdef RAJA_ENABLE_CUDA
    RAJA::cuda_exec<128>
#else
    RAJA::seq_exec
#endif
#ifdef RAJA_ENABLE_OPENMP
    ,
    RAJA::omp_parallel_for_exec
#endif
    >;

using DataTypes = std::tuple<int, float, double>;
using InOrderTypes = std::tuple<std::false_type, std::true_type>;

template <typename T1, typename T2>
using Cross = typename types::product<T1, T2>;
using Types = Cross<Cross<ExecTypes, DataTypes>::type, InOrderTypes>::type;

template <typename T>
struct ForTesting {
};

template <typename... Ts>
struct ForTesting<std::tuple<Ts...>> {
  using type = testing::Types<Ts...>;
};

using CrossTypes = ForTesting<Types>::type;

// Inclusive Scan

template <typename Exec, typename Storage>
void inclusive(Storage& data, bool inPlace = false)
{
  if (inPlace)
    RAJA::inclusive_scan_inplace<Exec>(data.ibegin(), data.iend());
  else
    RAJA::inclusive_scan<Exec>(data.ibegin(), data.iend(), data.obegin());
}

template <typename Data,
          typename Storage,
          typename Fn,
          typename T = typename std::remove_pointer<Data>::type::type>
void compareInclusive(Data original, Storage data, Fn function, T init)
{
  auto in = original->ibegin();
  auto out = data->obegin();
  T sum = *in;
  int index = 0;
  while ((out + index) != data->oend()) {
    ASSERT_EQ(sum, *(out + index)) << "Expected value differs at index "
                                   << index;
    ++index;
    sum = function(sum, *(in + index));
  }
}

template <typename Data,
          typename Storage,
          typename T = typename std::remove_pointer<Data>::type::type>
void compareInclusive(Data original, Storage data)
{
  compareInclusive(original, data, std::plus<T>{}, 0);
}

template <typename Tuple>
class InclusiveScanTest : public testing::Test
{
  using Exec = typename std::tuple_element<0, Tuple>::type;
  using T = typename std::tuple_element<1, Tuple>::type;
  using Bool = typename std::tuple_element<2, Tuple>::type;
  constexpr const static bool InPlace = Bool::value;

protected:
  using Storage = storage<Exec, T, InPlace>;

  virtual void SetUp()
  {
    data = new Storage{N};
    original = new Storage{N};
    std::iota(data->ibegin(), data->iend(), 1);
    data->update();
    std::shuffle(data->ibegin(),
                 data->iend(),
                 std::mt19937{std::random_device{}()});
    data->update();
    std::copy(data->ibegin(), data->iend(), original->ibegin());
    original->update();
    inclusive<Exec>(*data, InPlace);
    data->update();
  }

  virtual void TearDown()
  {
    delete data;
    delete original;
  }

  Storage* data;
  Storage* original;
};

TYPED_TEST_CASE_P(InclusiveScanTest);

TYPED_TEST_P(InclusiveScanTest, InclusiveCorrectness)
{
  compareInclusive(this->original, this->data);
}

REGISTER_TYPED_TEST_CASE_P(InclusiveScanTest, InclusiveCorrectness);

INSTANTIATE_TYPED_TEST_CASE_P(Scan, InclusiveScanTest, CrossTypes);

// Exclusive Scan


template <typename Exec, typename Storage>
void exclusive(Storage& data, bool inPlace = false)
{
  if (inPlace)
    RAJA::exclusive_scan_inplace<Exec>(data.ibegin(), data.iend());
  else
    RAJA::exclusive_scan<Exec>(data.ibegin(), data.iend(), data.obegin());
}

template <typename Data,
          typename Storage,
          typename Fn,
          typename T = typename std::remove_pointer<Data>::type::type>
void compareExclusive(Data original, Storage data, Fn function, T init)
{
  auto in = original->ibegin();
  auto out = data->obegin();
  T sum = init;
  int index = 0;
  while ((out + index) != data->oend()) {
    ASSERT_EQ(sum, *(out + index)) << "Expected value differs at index "
                                   << index;
    sum = function(sum, *(in + index));
    ++index;
  }
}

template <typename Data,
          typename Storage,
          typename T = typename std::remove_pointer<Data>::type::type>
void compareExclusive(Data original, Storage data)
{
  compareExclusive(original, data, std::plus<T>{}, 0);
}

template <typename Tuple>
class ExclusiveScanTest : public testing::Test
{
  using Exec = typename std::tuple_element<0, Tuple>::type;
  using T = typename std::tuple_element<1, Tuple>::type;
  using Bool = typename std::tuple_element<2, Tuple>::type;
  constexpr const static bool InPlace = Bool::value;

protected:
  using Storage = storage<Exec, T, InPlace>;

  virtual void SetUp()
  {
    data = new Storage{N};
    original = new Storage{N};
    std::iota(data->ibegin(), data->iend(), 1);
    data->update();
    std::shuffle(data->ibegin(),
                 data->iend(),
                 std::mt19937{std::random_device{}()});
    data->update();
    std::copy(data->ibegin(), data->iend(), original->ibegin());
    original->update();
    exclusive<Exec>(*data, InPlace);
    data->update();
  }

  virtual void TearDown()
  {
    delete data;
    delete original;
  }

  Storage* data;
  Storage* original;
};

TYPED_TEST_CASE_P(ExclusiveScanTest);

TYPED_TEST_P(ExclusiveScanTest, ExclusiveCorrectness)
{
  compareExclusive(this->original, this->data);
}

REGISTER_TYPED_TEST_CASE_P(ExclusiveScanTest, ExclusiveCorrectness);

INSTANTIATE_TYPED_TEST_CASE_P(Scan, ExclusiveScanTest, CrossTypes);
