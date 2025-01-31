/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   Header file for parallel region in kernel.
 *
 ******************************************************************************
 */

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-24, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef RAJA_pattern_kernel_region_HPP
#define RAJA_pattern_kernel_region_HPP

#include <iostream>
#include <type_traits>

#include "RAJA/config.hpp"
#include "RAJA/pattern/region.hpp"

namespace RAJA
{

namespace statement
{

template <typename RegionPolicy, typename... EnclosedStmts>
struct Region : public internal::Statement<camp::nil> {
};


}  // end namespace statement

namespace internal
{

// Statement executor to create a region within kernel

// Note: RAJA region's lambda must capture by reference otherwise
// internal function calls are undefined.
template <typename RegionPolicy, typename... EnclosedStmts, typename Types>
struct StatementExecutor<statement::Region<RegionPolicy, EnclosedStmts...>,
                         Types> {

  template <typename Data>
  static RAJA_INLINE void exec(Data &&data)
  {

    RAJA::region<RegionPolicy>([&]() {
      using data_t = camp::decay<Data>;
      execute_statement_list<camp::list<EnclosedStmts...>, Types>(data_t(data));
    });
  }
};


}  // namespace internal
}  // end namespace RAJA

#endif /* RAJA_pattern_kernel_HPP */
