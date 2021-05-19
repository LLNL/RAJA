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
// Copyright (c) 2016-19, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef RAJA_policy_tensor_kernel_For_HPP
#define RAJA_policy_tensor_kernel_For_HPP

#include "RAJA/config.hpp"

#include <iostream>
#include <type_traits>

#include "RAJA/pattern/kernel/internal.hpp"
#include "RAJA/pattern/kernel/Lambda.hpp"
#include "RAJA/policy/tensor/policy.hpp"

namespace RAJA
{

namespace internal
{

/*!
 * A generic RAJA::kernel forall_impl executor for statement::For
 *
 *
 */
template <camp::idx_t ArgumentId,
          typename TENSOR_TYPE,
          camp::idx_t DIM,
          camp::idx_t TILE_SIZE,
          typename... EnclosedStmts,
          typename Types>
struct StatementExecutor<
    statement::For<ArgumentId, RAJA::policy::tensor::tensor_exec<seq_exec, TENSOR_TYPE, DIM, TILE_SIZE>, EnclosedStmts...>,
    Types> {

  using stmt_list_t = StatementList<EnclosedStmts...>;


  template <typename Data>
  static RAJA_INLINE void exec(Data &&data)
  {

    using DataT = camp::decay<Data>;


    auto begin = camp::get<ArgumentId>(data.segment_tuple).begin();
    auto end = camp::get<ArgumentId>(data.segment_tuple).end();
    auto distance = std::distance(begin, end);

    using diff_t = decltype(distance);

    // compute the vector index type and new LoopTypes
    using value_type = camp::at_v<typename DataT::index_types_t, ArgumentId>;
    using tensor_index_type = TensorIndex<value_type, TENSOR_TYPE, DIM>;
    using NewTypes = setSegmentType<Types, ArgumentId, tensor_index_type>;


    // negative TILE_SIZE value uses entire loop range in one tile
    // this lets the expression templates do all of the looping
    // this is the default behavior unless the user defines a TILE_SIZE
    static_assert(TILE_SIZE != 0, "TILE_SIZE cannot be zero");
    if(TILE_SIZE < 0){
      // all in one shot. no loop necessary

      // assign the entire range
      data.template assign_offset<ArgumentId>(0);
      camp::get<ArgumentId>(data.vector_sizes) = distance;

      // execute
      execute_statement_list<stmt_list_t, NewTypes>(data);

    }
    else{


      // Streaming loop for complete vector widths
      camp::get<ArgumentId>(data.vector_sizes) = diff_t(TILE_SIZE);
      for (diff_t i = 0; i < distance; i+= diff_t(TILE_SIZE)) {

        if(i+diff_t(TILE_SIZE) > distance){
          camp::get<ArgumentId>(data.vector_sizes) = distance-i;
        }

        data.template assign_offset<ArgumentId>(i);
        execute_statement_list<stmt_list_t, NewTypes>(data);

      }
    }


  }
};


}  // namespace internal
}  // end namespace RAJA


#endif
