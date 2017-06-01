//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016, Lawrence Livermore National Security, LLC.
//
// Produced at the Lawrence Livermore National Laboratory
//
// LLNL-CODE-689114
//
// All rights reserved.
//
// This file is part of RAJA.
//
// For additional details, please also read RAJA/README.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// * Redistributions of source code must retain the above copyright notice,
//   this list of conditions and the disclaimer below.
//
// * Redistributions in binary form must reproduce the above copyright notice,
//   this list of conditions and the disclaimer (as noted below) in the
//   documentation and/or other materials provided with the distribution.
//
// * Neither the name of the LLNS/LLNL nor the names of its contributors may
//   be used to endorse or promote products derived from this software without
//   specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL LAWRENCE LIVERMORE NATIONAL SECURITY,
// LLC, THE U.S. DEPARTMENT OF ENERGY OR CONTRIBUTORS BE LIABLE FOR ANY
// DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
// DAMAGES  (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS
// OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
// HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
// STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING
// IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
//
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

///
/// Source file containing tests for RAJA GPU scans.
///

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

#include "data_storage.hpp"
#include "type_helper.hpp"

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
  compareInclusive(this->original.get(), this->data.get(), this->function);
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
