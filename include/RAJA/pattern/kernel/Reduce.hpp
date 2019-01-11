/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   Header file for statement wrappers and executors.
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


#ifndef RAJA_pattern_kernel_Reduce_HPP
#define RAJA_pattern_kernel_Reduce_HPP

#include "RAJA/config.hpp"

#include <iostream>
#include <type_traits>

#include "RAJA/pattern/kernel/internal.hpp"

namespace RAJA
{

namespace statement
{


/*!
 * A RAJA::kernel statement that implements a reduction of a Param.
 * This reduces a value down to a "root" thread, and then only executes
 * the enclosed statements on the thread which contains the reduced value.
 *
 */
template <typename ReducePolicy,
          template <typename...> class ReduceOperator,
          typename ParamId,
          typename... EnclosedStmts>
struct Reduce : public internal::Statement<camp::nil, EnclosedStmts...> {

  static_assert(std::is_base_of<internal::ParamBase, ParamId>::value,
                "Inappropriate ParamId, ParamId must be of type "
                "RAJA::Statement::Param< # >");

  using execution_policy_t = camp::nil;
};


}  // end namespace statement


}  // end namespace RAJA


#endif /* RAJA_pattern_kernel_Reduce_HPP */
