/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   Header file for tile wrapper and iterator.
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

#ifndef RAJA_pattern_kernel_Tile_HPP
#define RAJA_pattern_kernel_Tile_HPP

#include "RAJA/config.hpp"
#include "RAJA/util/defines.hpp"
#include "RAJA/util/types.hpp"

#include "camp/camp.hpp"
#include "camp/concepts.hpp"
#include "camp/tuple.hpp"

#include <iostream>
#include <type_traits>

namespace RAJA
{
namespace statement
{


/*!
 * A kernel::forall statement that implements a tiling (or blocking) loop.
 *
 */
template <camp::idx_t ArgumentId,
          typename TilePolicy,
          typename ExecPolicy,
          typename... EnclosedStmts>
struct Tile : public internal::Statement<ExecPolicy, EnclosedStmts...> {
  using tile_policy_t = TilePolicy;
  using exec_policy_t = ExecPolicy;
};

///! tag for a tiling loop
template <camp::idx_t chunk_size_>
struct tile_fixed {
  static constexpr camp::idx_t chunk_size = chunk_size_;
};


}  // end namespace statement

namespace internal
{


template <camp::idx_t ArgumentId, typename Data, typename... EnclosedStmts>
struct TileWrapper : public GenericWrapper<Data, EnclosedStmts...> {

  using Base = GenericWrapper<Data, EnclosedStmts...>;
  using Base::Base;

  template <typename InSegmentType>
  RAJA_INLINE void operator()(InSegmentType s)
  {
    // Assign the tile's segment to the tuple
    camp::get<ArgumentId>(Base::data.segment_tuple) = s;

    // Assign the beginning index to the index_tuple for proper use
    // in shmem windows
    camp::get<ArgumentId>(Base::data.offset_tuple) = 0;

    // Execute enclosed statements
    Base::exec();
  }
};


template <typename Iterable>
struct IterableTiler {
  using value_type = camp::decay<Iterable>;

  class iterator
  {
    // NOTE: this must be held by value for NVCC support, *even on the host*
    const IterableTiler itiler;
    const Index_type block_id;

  public:
    using value_type = camp::decay<Iterable>;
    using difference_type = camp::idx_t;
    using pointer = value_type *;
    using reference = value_type &;
    using iterator_category = std::random_access_iterator_tag;

    RAJA_HOST_DEVICE
    RAJA_INLINE
    constexpr iterator(IterableTiler const &itiler_, Index_type block_id_)
        : itiler{itiler_}, block_id{block_id_}
    {
    }

    RAJA_HOST_DEVICE
    RAJA_INLINE
    value_type operator*()
    {
      auto start = block_id * itiler.block_size;
      return itiler.it.slice(start, itiler.block_size);
    }

    RAJA_HOST_DEVICE
    RAJA_INLINE difference_type operator-(const iterator &rhs) const
    {
      return static_cast<difference_type>(block_id)
             - static_cast<difference_type>(rhs.block_id);
    }

    RAJA_HOST_DEVICE
    RAJA_INLINE iterator operator-(const difference_type &rhs) const
    {
      return iterator(itiler, block_id - rhs);
    }

    RAJA_HOST_DEVICE
    RAJA_INLINE iterator operator+(const difference_type &rhs) const
    {
      return iterator(itiler,
                      block_id + rhs >= itiler.num_blocks ? itiler.num_blocks
                                                          : block_id + rhs);
    }

    RAJA_HOST_DEVICE
    RAJA_INLINE value_type operator[](difference_type rhs) const
    {
      return *((*this) + rhs);
    }

    RAJA_HOST_DEVICE
    RAJA_INLINE bool operator!=(const IterableTiler &rhs) const
    {
      return block_id != rhs.block_id;
    }

    RAJA_HOST_DEVICE
    RAJA_INLINE bool operator<(const IterableTiler &rhs) const
    {
      return block_id < rhs.block_id;
    }
  };

  RAJA_HOST_DEVICE
  RAJA_INLINE
  IterableTiler(const Iterable &it_, camp::idx_t block_size_)
      : it{it_}, block_size{block_size_}
  {
    using std::begin;
    using std::end;
    using std::distance;
    dist = it.end() - it.begin();  // distance(begin(it), end(it));
    num_blocks = dist / block_size;
    // if (dist % block_size) num_blocks += 1;
    if (dist - num_blocks * block_size > 0) {
      num_blocks += 1;
    }
  }

  RAJA_HOST_DEVICE
  RAJA_INLINE
  iterator begin() { return iterator(*this, 0); }

  RAJA_HOST_DEVICE
  RAJA_INLINE
  iterator end() { return iterator(*this, num_blocks); }

  value_type it;
  camp::idx_t block_size;
  camp::idx_t num_blocks;
  camp::idx_t dist;
};


template <camp::idx_t ArgumentId,
          typename TPol,
          typename EPol,
          typename... EnclosedStmts>
struct StatementExecutor<statement::
                             Tile<ArgumentId, TPol, EPol, EnclosedStmts...>> {


  template <typename Data>
  static RAJA_INLINE void exec(Data &data)
  {
    // Get the segment we are going to tile
    auto const &segment = camp::get<ArgumentId>(data.segment_tuple);

    // Get the tiling policies chunk size
    auto chunk_size = TPol::chunk_size;

    // Create a tile iterator
    IterableTiler<decltype(segment)> tiled_iterable(segment, chunk_size);

    // Wrap in case forall_impl needs to thread_privatize
    TileWrapper<ArgumentId, Data, EnclosedStmts...> tile_wrapper(data);

    // Loop over tiles, executing enclosed statement list
    forall_impl(EPol{}, tiled_iterable, tile_wrapper);

    // Set range back to original values
    camp::get<ArgumentId>(data.segment_tuple) = tiled_iterable.it;
  }
};
}  // end namespace internal
}  // end namespace RAJA

#endif /* RAJA_pattern_kernel_HPP */
