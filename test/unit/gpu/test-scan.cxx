#include <algorithm>
#include <iostream>
#include <iterator>
#include <numeric>
#include <random>
#include <tuple>
#include <type_traits>

#include <cstdlib>

#include <gtest/gtest.h>

#include <RAJA/RAJA.hpp>

#include "data_storage.hxx"
#include "type_helper.hxx"

const int N = 1024;

// Unit Test Space Exploration

using ExecTypes = std::tuple<RAJA::cuda_exec<128>, RAJA::seq_exec>;

using ReduceTypes = std::tuple<RAJA::operators::plus<int>,
                               RAJA::operators::plus<float>,
                               RAJA::operators::plus<double>,
                               RAJA::operators::minimum<int>,
                               RAJA::operators::minimum<float>,
                               RAJA::operators::minimum<double>,
                               RAJA::operators::maximum<int>,
                               RAJA::operators::maximum<float>,
                               RAJA::operators::maximum<double>>;

using InPlaceTypes = std::tuple<std::false_type, std::true_type>;

template <typename T1, typename T2>
using Cross = typename types::product<T1, T2>;

using Types = Cross<Cross<ExecTypes, ReduceTypes>::type, InPlaceTypes>::type;

template <typename T>
struct ForTesting {
};

template <typename... Ts>
struct ForTesting<std::tuple<Ts...>> {
  using type = testing::Types<Ts...>;
};

using CrossTypes = ForTesting<Types>::type;


// dispatchers

template <typename Exec, typename Fn, typename Storage>
void inclusive(Storage* data, bool inPlace = false)
{
  if (inPlace)
    RAJA::inclusive_scan_inplace<Exec>(data->ibegin(), data->iend(), Fn{});
  else
    RAJA::inclusive_scan<Exec>(data->ibegin(),
                               data->iend(),
                               data->obegin(),
                               Fn{});
}

template <typename Exec, typename Fn, typename Storage>
void exclusive(Storage* data, bool inPlace = false)
{
  if (inPlace)
    RAJA::exclusive_scan_inplace<Exec>(data->ibegin(), data->iend(), Fn{});
  else
    RAJA::exclusive_scan<Exec>(data->ibegin(),
                               data->iend(),
                               data->obegin(),
                               Fn{});
}

// comparators

template <typename Data,
          typename Storage,
          typename Fn,
          typename T = typename std::remove_pointer<Data>::type::type>
void compareInclusive(Data original, Storage data, Fn function)
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

// test implementations

template <typename Tuple>
class ScanTest : public testing::Test
{
public:
  using Exec = typename std::tuple_element<0, Tuple>::type;
  using Fn = typename std::tuple_element<1, Tuple>::type;
  using Bool = typename std::tuple_element<2, Tuple>::type;
  constexpr const static bool InPlace = Bool::value;
  using T = typename Fn::result_type;
  using Storage = storage<Exec, T, InPlace>;

protected:
  virtual void SetUp()
  {
    std::iota(data->ibegin(), data->iend(), 1);
    std::shuffle(data->ibegin(),
                 data->iend(),
                 std::mt19937{std::random_device{}()});
    std::copy(data->ibegin(), data->iend(), original->ibegin());
  }

  std::unique_ptr<Storage> data = std::unique_ptr<Storage>(new Storage(N));
  std::unique_ptr<Storage> original = std::unique_ptr<Storage>(new Storage(N));
  Fn function = Fn();
};

template <typename Tuple>
class InclusiveScanTest : public ScanTest<Tuple>
{
protected:
  virtual void SetUp()
  {
    ScanTest<Tuple>::SetUp();
    inclusive<typename ScanTest<Tuple>::Exec, typename ScanTest<Tuple>::Fn>(
        this->data.get(), ScanTest<Tuple>::InPlace);
  }
};

template <typename Tuple>
class ExclusiveScanTest : public ScanTest<Tuple>
{
protected:
  virtual void SetUp()
  {
    ScanTest<Tuple>::SetUp();
    exclusive<typename ScanTest<Tuple>::Exec, typename ScanTest<Tuple>::Fn>(
        this->data.get(), ScanTest<Tuple>::InPlace);
  }
};

TYPED_TEST_CASE_P(InclusiveScanTest);
TYPED_TEST_CASE_P(ExclusiveScanTest);

TYPED_TEST_P(InclusiveScanTest, InclusiveCorrectness)
{
  compareInclusive(this->original.get(),
                   this->data.get(),
                   this->function);
}
TYPED_TEST_P(ExclusiveScanTest, ExclusiveCorrectness)
{
  auto init = decltype(this->function)::identity;
  compareExclusive(this->original.get(),
                   this->data.get(),
                   this->function,
                   init);
}

REGISTER_TYPED_TEST_CASE_P(InclusiveScanTest, InclusiveCorrectness);
REGISTER_TYPED_TEST_CASE_P(ExclusiveScanTest, ExclusiveCorrectness);

INSTANTIATE_TYPED_TEST_CASE_P(Scan, InclusiveScanTest, CrossTypes);
INSTANTIATE_TYPED_TEST_CASE_P(Scan, ExclusiveScanTest, CrossTypes);
