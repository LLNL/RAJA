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
