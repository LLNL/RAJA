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
    return (l == RAJA::Platform::cuda) ? l : r;
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
template<>
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
}
}

#endif
