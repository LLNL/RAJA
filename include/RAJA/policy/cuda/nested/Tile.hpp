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


  template <typename WrappedBody, typename Data, typename IndexCalc>
  static
  RAJA_DEVICE
  void exec(WrappedBody const &wrap, Data &data, IndexCalc const &index_calc)
  {
    // Get the segment referenced by this Tile statement
    auto const &iter = camp::get<ArgumentId>(data.segment_tuple);

    auto chunk_size = TPol::chunk_size;
    using segment_type = RAJA::TypedRangeSegment<RAJA::Index_type, RAJA::Index_type>;
    IterableTiler<segment_type> tiled_iterable(iter, chunk_size);

    // Pull out iterators
    auto begin = tiled_iterable.begin();
    auto end = tiled_iterable.end();

    // compute trip count
    auto len = end - begin;

    // Iterate through tiles
    for (decltype(len) i = 0;i < len;++ i) {

      // Assign our new tiled segment
      camp::get<ArgumentId>(data.segment_tuple) = *(begin+i);

      // Assign the beginning index to the index_tuple for proper use
      // in shmem windows
      camp::get<ArgumentId>(data.index_tuple) = *camp::get<ArgumentId>(data.segment_tuple).begin();

      // Execute our enclosed statement list
      wrap(data, index_calc);
    }


    // Set range back to original values
    camp::get<ArgumentId>(data.segment_tuple) = tiled_iterable.it;

  }
};




} // end namespace internal
}  // end namespace nested
}  // end namespace RAJA

#endif // RAJA_ENABLE_CUDA
#endif /* RAJA_pattern_nested_HPP */
