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


#ifndef RAJA_pattern_nested_ShmemWindow_HPP
#define RAJA_pattern_nested_ShmemWindow_HPP


#include "RAJA/config.hpp"
#include "RAJA/util/StaticLayout.hpp"

#include <iostream>
#include <type_traits>

namespace RAJA
{

namespace nested
{


/*!
 * A nested::forall statement that sets the shared memory window.
 *
 *
 */

template <typename... EnclosedStmts>
struct SetShmemWindow
    : public internal::Statement<camp::nil, EnclosedStmts...> {
};


namespace internal
{


template <typename... EnclosedStmts>
struct StatementExecutor<SetShmemWindow<EnclosedStmts...>> {


  template <typename Data>
  static RAJA_INLINE void exec(Data &&data)
  {
    // Grab pointer to shared shmem window
    using loop_data_t = camp::decay<Data>;
    using index_tuple_t = typename loop_data_t::index_tuple_t;
    index_tuple_t *shmem_window =
        static_cast<index_tuple_t *>(detail::getSharedMemoryWindow());

    if (shmem_window != nullptr) {
      // Set the window by copying the current index_tuple to the shared
      // location
      *shmem_window = data.get_begin_index_tuple();
    }

    // Invoke the enclosed statements
    execute_statement_list<camp::list<EnclosedStmts...>>(
        std::forward<Data>(data));
  }
};


}  // namespace internal
}  // end namespace nested
}  // end namespace RAJA


#endif /* RAJA_pattern_nested_HPP */
