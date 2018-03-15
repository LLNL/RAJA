/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   RAJA header file containing tiling policies and mechanics
 *          for forallN templates.
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

#ifndef RAJA_pattern_kernel_HPP
#define RAJA_pattern_kernel_HPP


#include "RAJA/config.hpp"
#include "RAJA/util/defines.hpp"
#include "RAJA/util/types.hpp"

#include "RAJA/pattern/kernel/internal.hpp"

#include "RAJA/util/chai_support.hpp"

#include "RAJA/pattern/shared_memory.hpp"

#include "camp/camp.hpp"
#include "camp/concepts.hpp"
#include "camp/tuple.hpp"

#include <iostream>
#include <type_traits>

namespace RAJA
{

/*!
 * A RAJA::kernel execution policy.
 *
 * This is just a list of RAJA::kernel statements.
 */
template <typename... Stmts>
using KernelPolicy = internal::StatementList<Stmts...>;


///
/// Template list of argument indices
///
template <camp::idx_t... ArgumentId>
using ArgList = camp::idx_seq<ArgumentId...>;


template <typename PolicyType,
          typename SegmentTuple,
          typename ParamTuple,
          typename... Bodies>
RAJA_INLINE void kernel_param(SegmentTuple &&segments,
                              ParamTuple &&params,
                              Bodies &&... bodies)
{
  detail::setChaiExecutionSpace<PolicyType>();

  // TODO: test that all policy members model the Executor policy concept
  // TODO: add a static_assert for functors which cannot be invoked with
  //       index_tuple
  // TODO: add assert that all Lambda<i> match supplied loop bodies

  using segment_tuple_t = camp::decay<SegmentTuple>;
  using param_tuple_t = camp::decay<ParamTuple>;

  using loop_data_t = internal::LoopData<PolicyType,
                                         segment_tuple_t,
                                         param_tuple_t,
                                         camp::decay<Bodies>...>;


  // Create the LoopData object, which contains our policy object,
  // our segments, loop bodies, and the tuple of loop indices
  // it is passed through all of the kernel mechanics by-referenece,
  // and only copied to provide thread-private instances.
  loop_data_t loop_data(std::forward<SegmentTuple>(segments),
                        std::forward<ParamTuple>(params),
                        std::forward<Bodies>(bodies)...);

  // Setup shared memory objects passed in through parameter tuple
  RAJA::internal::shmem_setup_buffers(loop_data.param_tuple);

  // initialize the shmem tuple to the beginning of each loop iteration
  RAJA::internal::shmem_set_windows(loop_data.param_tuple,
                                    loop_data.get_begin_index_tuple());

  // Execute!
  internal::execute_statement_list<PolicyType>(loop_data);


  detail::clearChaiExecutionSpace();
}

template <typename PolicyType, typename SegmentTuple, typename... Bodies>
RAJA_INLINE void kernel(SegmentTuple &&segments, Bodies &&... bodies)
{
  RAJA::kernel_param<PolicyType>(std::forward<SegmentTuple>(segments),
                                 RAJA::make_tuple(),
                                 std::forward<Bodies>(bodies)...);
}


}  // end namespace RAJA


#include "RAJA/pattern/kernel/Collapse.hpp"
#include "RAJA/pattern/kernel/Conditional.hpp"
#include "RAJA/pattern/kernel/For.hpp"
#include "RAJA/pattern/kernel/Hyperplane.hpp"
#include "RAJA/pattern/kernel/Lambda.hpp"
#include "RAJA/pattern/kernel/ShmemWindow.hpp"
#include "RAJA/pattern/kernel/Tile.hpp"


#endif /* RAJA_pattern_kernel_HPP */
