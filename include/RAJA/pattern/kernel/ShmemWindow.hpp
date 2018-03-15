/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   Header file for shared memory window.
 *
 ******************************************************************************
 */


//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-18, Lawrence Livermore National Security, LLC.
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


#ifndef RAJA_pattern_kernel_ShmemWindow_HPP
#define RAJA_pattern_kernel_ShmemWindow_HPP


#include "RAJA/config.hpp"
#include "RAJA/util/StaticLayout.hpp"

#include <iostream>
#include <type_traits>

namespace RAJA
{

namespace statement
{


/*!
 * A kernel::forall statement that sets the shared memory window.
 *
 *
 */

template <typename... EnclosedStmts>
struct SetShmemWindow
    : public internal::Statement<camp::nil, EnclosedStmts...> {
};


}  // end namespace statement

namespace internal
{


template <typename... EnclosedStmts>
struct StatementExecutor<statement::SetShmemWindow<EnclosedStmts...>> {


  template <typename Data>
  static RAJA_INLINE void exec(Data &&data)
  {

    // Call setWindow on all of our shmem objects
    shmem_set_windows(data.param_tuple, data.get_begin_index_tuple());

    // Invoke the enclosed statements
    execute_statement_list<camp::list<EnclosedStmts...>>(
        std::forward<Data>(data));
  }
};


}  // namespace internal
}  // end namespace RAJA


#endif /* RAJA_pattern_kernel_HPP */
