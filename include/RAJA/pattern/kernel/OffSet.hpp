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

#ifndef RAJA_pattern_kernel_OffSet_HPP
#define RAJA_pattern_kernel_OffSet_HPP


#include "RAJA/config.hpp"

#include "RAJA/pattern/kernel/internal.hpp"

#include <iostream>
#include <type_traits>

namespace RAJA
{
namespace internal
{
struct OffSetBase {
};
};
namespace statement
{


/*!
 * An expression that returns the value of the specified RAJA::kernel
 * OffSet.
 *
 * This allows run-time values to affect the control logic within
 * RAJA::kernel execution policies.
 */
template<camp::idx_t offId>
struct OffSet : public internal::OffSetBase
{
  constexpr static camp::idx_t offset_idx = offId;
};

template<camp::idx_t... offId>
struct OffSetList : public internal::OffSetBase
{
};

}  // namespace statement
}  // end namespace RAJA


#endif /* RAJA_pattern_kernel_HPP */
