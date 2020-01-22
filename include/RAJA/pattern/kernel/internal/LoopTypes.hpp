/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   Header file for loop kernel internals: LoopData structure and
 *          related helper functions.
 *
 ******************************************************************************
 */

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef RAJA_pattern_kernel_internal_LoopTypes_HPP
#define RAJA_pattern_kernel_internal_LoopTypes_HPP

#include "RAJA/config.hpp"

#include "camp/camp.hpp"


namespace RAJA
{
namespace internal
{


template <typename ArgTypes,
          typename OffsetTypes>
struct LoopTypes;

template <typename ... ArgTypes,
          typename ... OffsetTypes>
struct LoopTypes<camp::list<ArgTypes...>, camp::list<OffsetTypes...>> {

  using Self = LoopTypes<camp::list<ArgTypes...>, camp::list<OffsetTypes...>>;


  using arg_types_t = camp::list<ArgTypes...>;
  using offset_types_t = camp::list<OffsetTypes...>;
};


template<typename Data>
using makeInitialLoopTypes = LoopTypes<camp::list<>, camp::list<>>;


}  // end namespace internal
}  // end namespace RAJA


#endif /* RAJA_pattern_kernel_internal_LoopData_HPP */
