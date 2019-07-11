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

#ifndef RAJA_pattern_kernel_Param_HPP
#define RAJA_pattern_kernel_Param_HPP


#include "RAJA/config.hpp"

#include "RAJA/pattern/kernel/internal.hpp"

#include <iostream>
#include <type_traits>

namespace RAJA
{
namespace internal
{

struct ParamBase {
};

}// end namespace internal

namespace statement
{

/*!
 * An expression that returns the value of the specified RAJA::kernel
 * parameter.
 *
 * This allows run-time values to affect the control logic within
 * RAJA::kernel execution policies.
 */
template <camp::idx_t ParamId>
struct Param : public internal::ParamBase {

  constexpr static camp::idx_t param_idx = ParamId;

  template <typename Data>
  RAJA_HOST_DEVICE RAJA_INLINE static auto eval(Data const &data)
      -> decltype(camp::get<ParamId>(data.param_tuple))
  {
    return camp::get<ParamId>(data.param_tuple);
  }
};

}  // end namespace statement
}  // end namespace RAJA


#endif /* RAJA_pattern_kernel_HPP */
