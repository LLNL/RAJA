/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   Header file containing RAJA segment template methods for
 *          SIMD execution.
 *
 *          These methods should work on any platform. They make no
 *          asumptions about data alignment.
 *
 *          Note: Reduction operations should not be used with simd
 *          policies. Limited support.
 *
 *
 ******************************************************************************
 */

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-19, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef RAJA_forall_simd_register_HPP
#define RAJA_forall_simd_register_HPP

#include "RAJA/config.hpp"

#include <iterator>
#include <type_traits>

#include "RAJA/util/types.hpp"

#include "RAJA/internal/fault_tolerance.hpp"

#include "RAJA/policy/tensor/policy.hpp"

#include "RAJA/util/resource.hpp"

namespace RAJA
{
namespace policy
{
namespace tensor
{


template <typename TENSOR_TYPE, camp::idx_t DIM, camp::idx_t TILE_SIZE, typename Iterable, typename Func>
RAJA_INLINE
resources::EventProxy<typename resources::get_resource<seq_exec>::type>
forall_impl(resources::Host &host_res,
const tensor_exec<seq_exec, TENSOR_TYPE, DIM, TILE_SIZE>&,
                             Iterable &&iter,
                             Func &&loop_body)
{

 using Res = typename resources::get_resource<seq_exec>::type;

  auto begin = std::begin(iter);
  auto end = std::end(iter);
  auto distance = std::distance(begin, end);
  using diff_t = decltype(distance);
  using value_type = typename Iterable::value_type;
  using tensor_type = TENSOR_TYPE;
  using tensor_index_type = RAJA::expt::TensorIndex<value_type, tensor_type, DIM>;

  // negative TILE_SIZE value uses entire loop range in one tile
  // this lets the expression templates do all of the looping
  // this is the default behavior unless the user defines a TILE_SIZE
  static_assert(TILE_SIZE != 0, "TILE_SIZE cannot be zero");
  if(TILE_SIZE < 0){
    // all in one shot. no loop necessary
    loop_body(tensor_index_type::range(*begin, *end));
  }
  else{

    // loop over tiles
    for (diff_t i = 0; i < distance; i+= diff_t(TILE_SIZE)) {

      auto tile_begin = *(begin+i);
      auto tile_end = tile_begin+diff_t(TILE_SIZE);
      if(tile_end > distance){
        tile_end = distance;
      }
      loop_body(tensor_index_type::range(tile_begin, tile_end));
    }

  }

  return RAJA::resources::EventProxy<Res>(host_res);
}


}  // namespace tensor

}  // namespace policy


namespace expt {
  template <typename TENSOR_TYPE, camp::idx_t TENSOR_DIM, camp::idx_t TILE_SIZE, typename Iterable>
  struct LoopExecute<RAJA::policy::tensor::tensor_exec<seq_exec, TENSOR_TYPE, TENSOR_DIM, TILE_SIZE>, Iterable> {


    template <typename BODY>
    static RAJA_INLINE RAJA_HOST_DEVICE void exec(
        LaunchContext const RAJA_UNUSED_ARG(&ctx),
        Iterable const &iter,
        BODY const &loop_body)
    {

      auto begin = iter.begin();
      auto end = iter.end();
      auto distance = end-begin;
      using diff_t = decltype(distance);

      using value_type = typename Iterable::value_type;
      using tensor_type = TENSOR_TYPE;
      using tensor_index_type = expt::TensorIndex<value_type, tensor_type, TENSOR_DIM>;


      static_assert(TILE_SIZE != 0, "TILE_SIZE cannot be zero");
      if(TILE_SIZE < 0){
        // all in one shot. no loop necessary
        loop_body(tensor_index_type::range(*begin, *end));
      }
      else{

        // loop over tiles
        for (diff_t i = 0; i < distance; i+= diff_t(TILE_SIZE)) {

          auto tile_begin = *(begin+i);
          auto tile_end = tile_begin+diff_t(TILE_SIZE);
          if(tile_end > distance){
            tile_end = *(begin+distance);
          }
          loop_body(tensor_index_type::range(tile_begin, tile_end));
        }

      }

    }
  };

} // namespace expt

}  // namespace RAJA

#endif  // closing endif for header file include guard
