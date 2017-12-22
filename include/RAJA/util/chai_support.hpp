/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   RAJA header file defining internal type related constructs for
 *          interacting with CHAI.
 *
 ******************************************************************************
 */

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-17, Lawrence Livermore National Security, LLC.
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

#ifndef RAJA_DETAIL_RAJA_CHAI_HPP
#define RAJA_DETAIL_RAJA_CHAI_HPP

#include "RAJA/config.hpp"

#ifdef RAJA_ENABLE_CHAI

#include "chai/ArrayManager.hpp"
#include "chai/ExecutionSpaces.hpp"

#include "RAJA/policy/PolicyBase.hpp"

#include "RAJA/index/IndexSet.hpp"
#include "RAJA/internal/ForallNPolicy.hpp"
#include "RAJA/internal/LegacyCompatibility.hpp"


namespace RAJA
{
namespace detail
{

struct max_platform {
  RAJA_HOST_DEVICE
  RAJA_INLINE
  constexpr RAJA::Platform operator()(const RAJA::Platform& l,
                                      const RAJA::Platform& r) const
  {
    return (l > r) ? l : r;
  }
};

template <Platform p>
struct get_space_impl {
};

template <>
struct get_space_impl<Platform::host> {
  static constexpr chai::ExecutionSpace value = chai::CPU;
};

#if defined(RAJA_ENABLE_CUDA)
template <>
struct get_space_impl<Platform::cuda> {
  static constexpr chai::ExecutionSpace value = chai::GPU;
};
#endif

template <>
struct get_space_impl<Platform::undefined> {
  static constexpr chai::ExecutionSpace value = chai::NONE;
};


template <typename... Ts>
struct get_space_from_list {
  static constexpr chai::ExecutionSpace value =
      get_space_impl<VarOps::foldl(max_platform(), Ts::platform...)>::value;
};

template <typename T, typename = void>
struct get_space {
  // catch-all: undefined CHAI space
  static constexpr chai::ExecutionSpace value = chai::NONE;
};

template <typename T>
struct get_space<T,
                 typename std::
                     enable_if<std::is_base_of<RAJA::PolicyBase, T>::value
                               && !RAJA::type_traits::is_indexset_policy<T>::
                                      value>::type>
    : public get_space_impl<T::platform> {
};

template <typename SEG, typename EXEC>
struct get_space<RAJA::ExecPolicy<SEG, EXEC>> : public get_space<EXEC> {
};

template <typename SEG, typename EXEC>
struct get_space_from_list<RAJA::ExecPolicy<SEG, EXEC>> {
  static constexpr chai::ExecutionSpace value = get_space<EXEC>::value;
};

template <typename TAGS, typename... POLICIES>
struct get_space<RAJA::NestedPolicy<RAJA::ExecList<POLICIES...>, TAGS>>
    : public get_space_from_list<POLICIES...> {
};


template <typename... POLICIES>
struct get_space_from_list<camp::list<POLICIES...>> {
  static constexpr chai::ExecutionSpace value =
      get_space_from_list<typename POLICIES::policy_type...>::value;
};
}
}

#endif  // RAJA_ENABLE_CHAI


namespace RAJA
{
namespace detail
{


/*!
 * Function to set the CHAI execution space based on the policy.
 *
 * This function is always defined, and is a NOP if CHAI is not enabled.
 */
template <typename ExecutionPolicy>
RAJA_INLINE void setChaiExecutionSpace()
{
#if defined(RAJA_ENABLE_CHAI)
  chai::ArrayManager* rm = chai::ArrayManager::getInstance();
  using EP = typename std::decay<ExecutionPolicy>::type;
  // printf("RAJA::setChaiExecutionSpace to %d\n",
  // (int)(detail::get_space<EP>::value));
  rm->setExecutionSpace(detail::get_space<EP>::value);
#endif
}

/*!
 * Function to set the CHAI execution space to chai::NONE.
 *
 * This function is always defined, and is a NOP if CHAI is not enabled.
 */
RAJA_INLINE
void clearChaiExecutionSpace()
{
#if defined(RAJA_ENABLE_CHAI)
  chai::ArrayManager* rm = chai::ArrayManager::getInstance();
  // std::cout << "RAJA::clearChaiExecutionSpace" << std::endl;
  rm->setExecutionSpace(chai::NONE);
#endif
}
}
}


#endif
