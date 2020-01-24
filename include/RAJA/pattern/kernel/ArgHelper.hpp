/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   Header file for kernel lambda executor.
 *
 ******************************************************************************
 */


//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//


#ifndef RAJA_pattern_kernel_arghelper_HPP
#define RAJA_pattern_kernel_arghelper_HPP

#include "RAJA/config.hpp"

#include <iostream>
#include <type_traits>

#include "camp/camp.hpp"
#include "camp/concepts.hpp"
#include "camp/tuple.hpp"

#include "RAJA/util/macros.hpp"
#include "RAJA/util/types.hpp"

#include "RAJA/pattern/kernel/Param.hpp"
#include "RAJA/pattern/kernel/LambdaArgs.hpp"

namespace RAJA
{

namespace internal
{

using RAJA::statement::seg_t;
using RAJA::statement::param_t;
using RAJA::statement::offset_t;
using RAJA::statement::LambdaArgs;

//Extracts arguments from segments, and parameters
template<typename T>
struct extractor;

template<camp::idx_t id>
struct extractor<LambdaArgs<offset_t, id>>
{

  template<typename Data>
  RAJA_HOST_DEVICE
  static auto extract_arg(Data &&data) ->
    decltype(camp::get<id>(data.offset_tuple))
  {
    return camp::get<id>(data.offset_tuple);
  }

};

template<camp::idx_t id>
struct extractor<LambdaArgs<seg_t, id>>
{
  template<typename Data>
  RAJA_HOST_DEVICE
  static auto extract_arg(Data &&data) ->
    decltype(camp::get<id>(data.segment_tuple).begin()[camp::get<id>(data.offset_tuple)])
  {
    return camp::get<id>(data.segment_tuple).begin()[camp::get<id>(data.offset_tuple)];
  }
};

template<camp::idx_t id>
struct extractor<LambdaArgs<param_t, id>>
{
  template<typename Data>
  RAJA_HOST_DEVICE
  static auto extract_arg(Data &&data)->
    typename std::add_lvalue_reference<camp::tuple_element_t<id,typename camp::decay<Data>::param_tuple_t>>::type
  {
    return camp::get<id>(data.param_tuple);
  }
};

}  // namespace internal

}  // end namespace RAJA


#endif /* RAJA_pattern_kernel_HPP */
