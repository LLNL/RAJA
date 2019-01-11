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

#ifndef RAJA_region_sequential_HPP
#define RAJA_region_sequential_HPP

namespace RAJA
{
namespace policy
{
namespace sequential
{

/*!
 * \brief RAJA::region implementation for sequential
 *
 * Generates sequential region
 *
 * \code
 *
 * RAJA::region<seq_region>([=](){
 *
 *  // region body - may contain multiple loops
 *
 *  });
 *
 * \endcode
 *
 * \tparam Policy region policy
 *
 */

template <typename Func>
RAJA_INLINE void region_impl(const seq_region &, Func &&body)
{
  body();
}

}  // namespace sequential

}  // namespace policy

}  // namespace RAJA

#endif  // closing endif for header file include guard
