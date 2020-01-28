/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   Header file for kernel conditional templates
 *
 ******************************************************************************
 */

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef RAJA_pattern_kernel_LambdaArgs_HPP
#define RAJA_pattern_kernel_LambdaArgs_HPP


#include "RAJA/config.hpp"

#include "RAJA/pattern/kernel/internal.hpp"

#include <iostream>
#include <type_traits>

namespace RAJA
{

namespace statement
{

struct seg_t
{};

struct param_t
{};

struct offset_t
{};

template<typename T, camp::idx_t ...>
struct LambdaArgs
{
};

template<camp::idx_t ... args> 
using Segs = camp::list<LambdaArgs<seg_t, args>...>;

template<camp::idx_t ... args> 
using Offsets = camp::list<LambdaArgs<offset_t, args>...>;

template<camp::idx_t ... args> 
using Params = camp::list<LambdaArgs<param_t, args>...>;


}  // namespace statement
}  // end namespace RAJA


#endif /* RAJA_pattern_kernel_HPP */
