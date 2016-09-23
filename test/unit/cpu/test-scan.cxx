#include <algorithm>
#include <iostream>
#include <iterator>
#include <numeric>
#include <random>
#include <tuple>
#include <type_traits>

#include <cstdlib>

#include <gtest/gtest.h>
#include <RAJA/LegacyCompatibility.hxx>
#include <RAJA/RAJA.hxx>

#include "data_storage.hxx"
#include "type_helper.hxx"

const int N = 1024;

// Unit Test Space Exploration

#if 0
//#ifdef RAJA_ENABLE_OPENMP
using ExecTypes = std::tuple<RAJA::seq_exec, RAJA::omp_parallel_for_exec>;
#else
using ExecTypes = std::tuple<RAJA::seq_exec>;
#endif
namespace API
{
struct type {
};
struct Iterators : type {
};
struct Iterables : type {
};
struct Containers : type {
};
}

using APITypes = std::tuple<API::Iterators, API::Iterables, API::Containers>;

/*
        using ReduceTypes = std::tuple<RAJA::operators::safe_plus<int>,
                               RAJA::operators::safe_plus<float>,
                               RAJA::operators::safe_plus<double>,
                               RAJA::operators::minimum<int>,
                               RAJA::operators::minimum<float>,
                               RAJA::operators::minimum<double>,
                               RAJA::operators::maximum<int>,
                               RAJA::operators::maximum<float>,
                               RAJA::operators::maximum<double>>;
*/
using ReduceTypes = std::tuple<RAJA::operators::safe_plus<float>,
                               RAJA::operators::maximum<int>>;

using InPlaceTypes = std::tuple<std::false_type, std::true_type>;

template <typename T1, typename T2>
using Cross = typename types::product<T1, T2>::type;

using Types =
    Cross<Cross<Cross<ExecTypes, ReduceTypes>, InPlaceTypes>, APITypes>;

template <typename T>
struct ForTesting {
};

template <typename... Ts>
struct ForTesting<std::tuple<Ts...>> {
  using type = testing::Types<Ts...>;
};

using CrossTypes = ForTesting<Types>::type;

template <typename APIType, bool inPlace>
struct Params {
};
template <>
struct Params<API::Iterators, true> {
  template <typename Fn, typename Storage>
  static auto get(Storage* data)
      -> decltype(std::make_tuple(data->in.begin(), data->in.end(), Fn{}))
  {
    return std::make_tuple(data->in.begin(), data->in.end(), Fn{});
  }
};
template <>
struct Params<API::Iterators, false> {
  template <typename Fn, typename Storage>
  static auto get(Storage* data) -> decltype(std::make_tuple(data->in.begin(),
                                                             data->in.end(),
                                                             data->out.begin(),
                                                             Fn{}))
  {
    return std::make_tuple(data->in.begin(),
                           data->in.end(),
                           data->out.begin(),
                           Fn{});
  }
};
template <>
struct Params<API::Iterables, true> {
  template <typename Fn, typename Storage>
  static auto get(Storage* data)
      -> decltype(std::make_tuple(RAJA::RangeSegment{0, data->in.size()},
                                  data->in.begin(),
                                  Fn{},
                                  Fn::identity))
  {
    return std::make_tuple(RAJA::RangeSegment{0, data->in.size()},
                           data->in.begin(),
                           Fn{},
                           Fn::identity);
  }
};

template <>
struct Params<API::Iterables, false> {
  template <typename Fn, typename Storage>
  static auto get(Storage* data)
      -> decltype(std::make_tuple(RAJA::RangeSegment{0, data->in.size()},
                                  data->in.begin(),
                                  data->out.begin(),
                                  Fn{},
                                  Fn::identity))
  {
    return std::make_tuple(RAJA::RangeSegment{0, data->in.size()},
                           data->in.begin(),
                           data->out.begin(),
                           Fn{},
                           Fn::identity);
  }
};
template <>
struct Params<API::Containers, true> {
  template <typename Fn, typename Storage>
  static auto get(Storage* data)
      -> decltype(std::make_tuple(data->in, Fn{}, Fn::identity))
  {
    return std::make_tuple(data->in, Fn{}, Fn::identity);
  }
};

template <>
struct Params<API::Containers, false> {
  template <typename Fn, typename Storage>
  static auto get(Storage* data)
      -> decltype(std::make_tuple(data->in, data->out, Fn{}, Fn::identity))
  {
    return std::make_tuple(data->in, data->out, Fn{}, Fn::identity);
  }
};

template <typename Fn, typename... Args>
void dispatch(Fn&& f, std::tuple<Args...>&& t)
{
  VarOps::invoke(t, f);
};

template <typename T>
struct getFunType {
};

template <typename... Args>
struct getFunType<std::tuple<Args...>> {
  using type = void(Args...);
};

template <bool inPlace>
struct inclusive {
};

template <>
struct inclusive<true> {
  template <typename Exec, typename Fn, typename APIType, typename Storage>
  static void exec(Storage* data)
  {
    auto&& args = Params<APIType, true>::template get<Fn>(data);
    using Args = typename std::decay<decltype(args)>::type;
    using FnType = typename getFunType<Args>::type;
    dispatch((FnType*)RAJA::inclusive_scan_inplace<Exec>,
             std::forward<Args>(args));
  }
};

template <>
struct inclusive<false> {
  template <typename Exec, typename Fn, typename APIType, typename Storage>
  static void exec(Storage* data)
  {
    auto&& args = Params<APIType, true>::template get<Fn>(data);
    using Args = typename std::decay<decltype(args)>::type;
    using FnType = typename getFunType<Args>::type;
    dispatch((FnType*)RAJA::inclusive_scan<Exec>, std::forward<Args>(args));
  }
};

template <bool inPlace>
struct exclusive {
};

template <>
struct exclusive<true> {
  template <typename Exec, typename Fn, typename APIType, typename Storage>
  static void exec(Storage* data)
  {
    auto&& args = Params<APIType, true>::template get<Fn>(data);
    using Args = typename std::decay<decltype(args)>::type;
    using FnType = typename getFunType<Args>::type;
    dispatch((FnType*)RAJA::exclusive_scan_inplace<Exec>,
             std::forward<Args>(args));
  }
};

template <>
struct exclusive<false> {
  template <typename Exec, typename Fn, typename APIType, typename Storage>
  static void exec(Storage* data)
  {
    auto&& args = Params<APIType, true>::template get<Fn>(data);
    using Args = typename std::decay<decltype(args)>::type;
    using FnType = typename getFunType<Args>::type;
    dispatch((FnType*)RAJA::exclusive_scan<Exec>, std::forward<Args>(args));
  }
};

template <typename Data,
          typename Storage,
          typename Fn,
          typename T = typename std::remove_pointer<Data>::type::type>
void compareInclusive(Data original, Storage data, Fn function, T init)
{
  auto in = original->in.begin();
  auto out = data->out.begin();
  T sum = *in;
  int index = 0;
  while ((out + index) != data->out.end()) {
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
  auto in = original->in.begin();
  auto out = data->out.begin();
  T sum = init;
  int index = 0;
  while ((out + index) != data->out.end()) {
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
  using API = typename std::tuple_element<3, Tuple>::type;
  constexpr const static bool InPlace = Bool::value;
  using T = typename Fn::result_type;
  using Storage = storage<Exec, T, InPlace>;

protected:
  virtual void SetUp()
  {
    std::iota(data->in.begin(), data->in.end(), 1);
    std::shuffle(data->in.begin(),
                 data->in.end(),
                 std::mt19937{std::random_device{}()});
    std::copy(data->in.begin(), data->in.end(), original->in.begin());
  }
  // eventually replace with std::make_unique<Storage>(N)
  std::unique_ptr<Storage> data = std::unique_ptr<Storage>(new Storage{N});
  std::unique_ptr<Storage> original = std::unique_ptr<Storage>(new Storage{N});
  Fn function = Fn{};
};

template <typename Tuple>
class InclusiveScanTest : public ScanTest<Tuple>
{
  using Parent = ScanTest<Tuple>;

protected:
  virtual void SetUp()
  {
    Parent::SetUp();
    ::inclusive<Parent::InPlace>::template exec<typename Parent::Exec,
                                                typename Parent::Fn,
                                                typename Parent::API>(
        this->data.get());
  }
};

template <typename Tuple>
class ExclusiveScanTest : public ScanTest<Tuple>
{
  using Parent = ScanTest<Tuple>;

protected:
  virtual void SetUp()
  {
    Parent::SetUp();
    exclusive<Parent::InPlace>::template exec<typename Parent::Exec,
                                              typename Parent::Fn,
                                              typename Parent::API>(
        this->data.get());
  }
};

TYPED_TEST_CASE_P(InclusiveScanTest);
TYPED_TEST_CASE_P(ExclusiveScanTest);

TYPED_TEST_P(InclusiveScanTest, InclusiveCorrectness)
{
  auto init = decltype(this->function)::identity;
  compareInclusive(this->original.get(),
                   this->data.get(),
                   this->function,
                   init);
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
