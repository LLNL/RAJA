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
// For additional details, please also read RAJA/LICENSE.
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
struct get_space_from_platform {
};

template <>
struct get_space_from_platform<Platform::host> {
  static constexpr chai::ExecutionSpace value = chai::CPU;
};

#if defined(RAJA_ENABLE_CUDA)
template <>
struct get_space_from_platform<Platform::cuda> {
  static constexpr chai::ExecutionSpace value = chai::GPU;
};
#endif

template <>
struct get_space_from_platform<Platform::undefined> {
  static constexpr chai::ExecutionSpace value = chai::NONE;
};





/*!
 * Returns the platform for the specified execution policy.
 * This is a catch-all, so anything undefined gets Platform::undefined
 */
template <typename T, typename = void>
struct get_platform {
  // catch-all: undefined CHAI space
  static constexpr Platform value = Platform::undefined;
};



/*!
 * Takes a list of policies, extracts their platforms, and provides the
 * reduction of them all.
 */
template <typename ... Policies>
struct get_platform_from_list {
  static constexpr Platform value =
      VarOps::foldl(max_platform(), get_platform<Policies>::value...);
};

/*!
 * Define an empty list as Platform::undefined;
 */
template <>
struct get_platform_from_list<> {
  static constexpr Platform value = Platform::undefined;
};



/*!
 * Specialization to define the platform for anything derived from PolicyBase,
 * which should catch all standard policies.
 *
 * (not for MultiPolicy or nested::Policy)
 */
template <typename T>
struct get_platform<T,
                 typename std::
                     enable_if<std::is_base_of<RAJA::PolicyBase, T>::value
                               && !RAJA::type_traits::is_indexset_policy<T>::
                                      value>::type>{

  static constexpr Platform value = T::platform;
};


/*!
 * Specialization to define the platform for an IndexSet execution policy.
 *
 * Examines both segment iteration and segment execution policies.
 */
template <typename SEG, typename EXEC>
struct get_platform<RAJA::ExecPolicy<SEG, EXEC>>  :
  public get_platform_from_list<SEG, EXEC>
{};


/*!
 * specialization for combining the execution polices for a forallN policy.
 *
 */
template <typename TAGS, typename... POLICIES>
struct get_platform<RAJA::NestedPolicy<RAJA::ExecList<POLICIES...>, TAGS>>
  : public get_platform_from_list<POLICIES...>
{};



template<typename T>
using get_space = get_space_from_platform<get_platform<T>::value>;


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
