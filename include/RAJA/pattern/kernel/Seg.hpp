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

#ifndef RAJA_pattern_kernel_Seg_HPP
#define RAJA_pattern_kernel_Seg_HPP


#include "RAJA/config.hpp"

#include "RAJA/pattern/kernel/internal.hpp"

#include <iostream>
#include <type_traits>

namespace RAJA
{
namespace internal
{

struct SegBase {
};

};

namespace statement
{

/*!
 * An expression that returns the value of the specified RAJA::kernel
 * segment.
 *
 * This allows run-time values to affect the control logic within
 * RAJA::kernel execution policies.
 */
template <camp::idx_t SegId>
struct Seg : public internal::SegBase {

  constexpr static camp::idx_t seg_idx = SegId;

  template <typename Data>
  RAJA_HOST_DEVICE RAJA_INLINE static auto eval(Data const &data)
      -> decltype(camp::get<Seg>(data.offset_tuple))
  {
    return camp::get<SegId>(data.offset_tuple);
  }
};

template <camp::idx_t... SegId>
struct SegList : public internal::SegBase {

};

}  // namespace statement
}  // end namespace RAJA


#endif /* RAJA_pattern_kernel_HPP */
