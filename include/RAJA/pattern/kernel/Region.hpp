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
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef RAJA_pattern_kernel_region_HPP
#define RAJA_pattern_kernel_region_HPP

#include "RAJA/config.hpp"
#include "RAJA/pattern/region.hpp"
#include "RAJA/policy/openmp/policy.hpp"
#include "RAJA/policy/sequential/policy.hpp"

#include <iostream>
#include <type_traits>

namespace RAJA
{

namespace statement
{

template<typename RegionPolicy, typename... EnclosedStmts>
struct Region : public internal::Statement<camp::nil> {
};

struct OmpSyncThreads : public internal::Statement<camp::nil> {  
};

}  // end namespace statement

namespace internal
{

//Statement executor to create a region within kernel

//Note: RAJA region's lambda must capture by reference otherwise
//internal function calls are undefined.
template<typename RegionPolicy, typename... EnclosedStmts>
struct StatementExecutor<statement::Region<RegionPolicy, EnclosedStmts...> > {

template<typename Data>
static RAJA_INLINE void exec(Data &&data)
{

  RAJA::region<RegionPolicy>([&]() {
      using data_t = camp::decay<Data>;
      execute_statement_list<camp::list<EnclosedStmts...>>(data_t(data));
    });
}

};

#if defined(RAJA_ENABLE_OPENMP)
//Statement executor to synchronize omp threads inside a kernel region
template<>
struct StatementExecutor<statement::OmpSyncThreads> {

template<typename Data>
static RAJA_INLINE void exec(Data &&)
{
  #pragma omp barrier
}

};
#endif

}  // namespace internal
}  // end namespace RAJA

#endif /* RAJA_pattern_kernel_HPP */
