//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-19, Lawrence Livermore National Security, LLC.
//
// Produced at the Lawrence Livermore National Laboratory
//
// LLNL-CODE-689114
//
// All rights reserved.
//
// This file is part of RAJA.
//
// For details about use and distribution, please read RAJA/LICENSE.
//
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
#ifndef __RAJA_PRIVATIZER_HPP
#define __RAJA_PRIVATIZER_HPP

#include "RAJA/config.hpp"
#include "camp/camp.hpp"
#include "camp/concepts.hpp"

namespace RAJA
{

namespace internal
{

// template <typename T>
// struct HasPrivatizer : DefineConcept(typename T::privatizer(camp::val<T>()))
// {
// };
// DefineTypeTraitFromConcept(has_privatizer, HasPrivatizer);

template <typename T>
class has_privatizer
{
private:
  template <typename C>
  static auto Test(void*)
      -> decltype(camp::val<typename C::privatizer>(), camp::true_type{});

  template <typename>
  static camp::false_type Test(...);

public:
  static bool const value = decltype(Test<T>(0))::value;
};


static_assert(!has_privatizer<int>::value, "if this fires, abandon all hope");

struct GenericWrapperBase {
};

template <typename T>
struct Privatizer {
  using value_type = camp::decay<T>;
  using reference_type = value_type&;
  value_type priv;
  static_assert(!has_privatizer<T>::value,
                "Privatizer selected inappropriately, this is almost certainly "
                "a bug");
  static_assert(!std::is_base_of<GenericWrapperBase, T>::value,
                "Privatizer selected inappropriately, this is almost certainly "
                "a bug");

  RAJA_SUPPRESS_HD_WARN
  RAJA_HOST_DEVICE Privatizer(const T& o) : priv{o} {}

  RAJA_SUPPRESS_HD_WARN
  RAJA_HOST_DEVICE reference_type get_priv() { return priv; }
};

/**
 * @brief Create a private copy of the argument to be stored on the current
 * thread's stack in a class of the Privatizer concept
 *
 * @param item data to privatize
 *
 * @return Privatizer<T>
 *
 * This function will be invoked such that ADL can be used to extend its
 * functionality.  Anywhere it is called it should be invoked by:
 *
 * `using RAJA::internal::thread_privatize; thread_privatize()`
 *
 * This allows other namespaces to add new versions to support functionality
 * that does not belong here.
 *
 */
template <typename T,
          typename std::enable_if<!has_privatizer<T>::value>::type* = nullptr>
RAJA_HOST_DEVICE auto thread_privatize(const T& item) -> Privatizer<T>
{
  return Privatizer<T>{item};
}

RAJA_SUPPRESS_HD_WARN
template <typename T,
          typename std::enable_if<has_privatizer<T>::value>::type* = nullptr>
RAJA_HOST_DEVICE auto thread_privatize(const T& item) -> typename T::privatizer
{
  return typename T::privatizer{item};
}

}  // namespace internal

}  // namespace RAJA

#endif /* __RAJA_PRIVATIZER_HPP */
