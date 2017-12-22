#ifndef RAJA_policy_cuda_nested_Tile_HPP
#define RAJA_policy_cuda_nested_Tile_HPP

#include "RAJA/config.hpp"
#include "RAJA/util/defines.hpp"
#include "RAJA/util/types.hpp"

#ifdef RAJA_ENABLE_CUDA

#include "RAJA/pattern/nested/internal.hpp"
#include "RAJA/pattern/nested/Tile.hpp"

#include "camp/camp.hpp"
#include "camp/concepts.hpp"
#include "camp/tuple.hpp"

#include <iostream>
#include <type_traits>

namespace RAJA
{
namespace nested
{
namespace internal{




template <camp::idx_t ArgumentId, typename TPol, typename ... InnerPolicies>
struct CudaStatementExecutor<Tile<ArgumentId, TPol, seq_exec, InnerPolicies...>> {

  using TileType = Tile<ArgumentId, TPol, seq_exec, InnerPolicies...>;
  using inner_policy_t = Policy<InnerPolicies...>;

  const inner_policy_t inner_policy;

  template <typename WrappedBody>
  RAJA_DEVICE
  void operator()(TileType const &fp, WrappedBody const &wrap, CudaExecInfo &exec_info)
  {
    // Get the segment referenced by this Tile statement
    auto const &iter = camp::get<ArgumentId>(wrap.data.segment_tuple);

    //using segment_type = typename std::remove_reference<decltype(iter)>::type;
    using segment_type = RAJA::TypedRangeSegment<RAJA::Index_type, RAJA::Index_type>;
    IterableTiler<segment_type> tiled_iterable(iter, fp.tile_policy.get_chunk_size());

    // Create a wrapper for inside this policy
    auto inner_wrapper = internal::cuda_make_statement_list_wrapper(fp.enclosed_statements, wrap.data);

    // Pull out iterators
    auto begin = tiled_iterable.begin();
    auto end = tiled_iterable.end();

    // compute trip count
    auto len = end - begin;

    // Iterate through tiles
    for (decltype(len) i = 0;i < len;++ i) {

      // Assign our new tiled segment
      camp::get<ArgumentId>(wrap.data.segment_tuple) = *(begin+i);

      // Execute our enclosed statement list
      inner_wrapper(exec_info);
    }


    // Set range back to original values
    camp::get<ArgumentId>(wrap.data.segment_tuple) = tiled_iterable.it;

  }
};




} // end namespace internal
}  // end namespace nested
}  // end namespace RAJA

#endif // RAJA_ENABLE_CUDA
#endif /* RAJA_pattern_nested_HPP */
