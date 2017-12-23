//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-17, Lawrence Livermore National Security, LLC.
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

#ifndef RAJA_pattern_nested_tile_HPP
#define RAJA_pattern_nested_tile_HPP

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
namespace nested
{


template <camp::idx_t Index, typename TilePolicy, typename ExecPolicy>
struct Tile {
  const TilePolicy tile_policy;
  const ExecPolicy exec_policy;
  RAJA_HOST_DEVICE constexpr Tile(TilePolicy const &tp = TilePolicy{},
                                  ExecPolicy const &ep = ExecPolicy{})
      : tile_policy{tp}, exec_policy{ep}
  {
  }
};

///! tag for a tiling loop
template <camp::idx_t chunk_size_>
struct tile_fixed {
  static constexpr camp::idx_t chunk_size = chunk_size_;

  RAJA_HOST_DEVICE constexpr tile_fixed() {}
  constexpr camp::idx_t get_chunk_size() const { return chunk_size; }
};

///! tag for a tiling loop
template <camp::idx_t default_chunk_size>
struct tile {
  camp::idx_t chunk_size;

  RAJA_HOST_DEVICE constexpr tile(camp::idx_t chunk_size_ = default_chunk_size)
      : chunk_size{chunk_size_}
  {
  }
  camp::idx_t get_chunk_size() const { return chunk_size; }
};

template <camp::idx_t Index, typename BaseWrapper>
struct TileWrapper : GenericWrapper<Index, BaseWrapper> {
  using Base = GenericWrapper<Index, BaseWrapper>;
  using Base::Base;
  template <typename InSegmentType>
  void operator()(InSegmentType s)
  {
    camp::get<Index>(Base::wrapper.data.segment_tuple) = s;
    Base::wrapper();
  }
};

/**
 * @brief specialization of internal::thread_privatize for tile
 */
template <camp::idx_t Index, typename BW>
auto thread_privatize(const nested::TileWrapper<Index, BW> &item)
    -> NestedPrivatizer<nested::TileWrapper<Index, BW>>
{
  return NestedPrivatizer<nested::TileWrapper<Index, BW>>{item};
}

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

    constexpr iterator(IterableTiler const &itiler_, Index_type block_id_)
        : itiler{itiler_}, block_id{block_id_}
    {
    }

    value_type operator*()
    {
      auto start = block_id * itiler.block_size;
      return itiler.it.slice(start, itiler.block_size);
    }

    inline difference_type operator-(const iterator &rhs) const
    {
      return static_cast<difference_type>(block_id)
             - static_cast<difference_type>(rhs.block_id);
    }

    inline iterator operator-(const difference_type &rhs) const
    {
      return iterator(itiler, block_id - rhs);
    }

    inline iterator operator+(const difference_type &rhs) const
    {
      return iterator(itiler,
                      block_id + rhs >= itiler.num_blocks ? itiler.num_blocks
                                                          : block_id + rhs);
    }

    inline value_type operator[](difference_type rhs) const
    {
      return *((*this) + rhs);
    }

    inline bool operator!=(const IterableTiler &rhs) const
    {
      return block_id != rhs.block_id;
    }

    inline bool operator<(const IterableTiler &rhs) const
    {
      return block_id < rhs.block_id;
    }
  };

  IterableTiler(const Iterable &it_, camp::idx_t block_size_)
      : it{it_}, block_size{block_size_}
  {
    using std::begin;
    using std::end;
    using std::distance;
    dist = distance(begin(it), end(it));
    num_blocks = dist / block_size;
    if (dist % block_size) num_blocks += 1;
  }

  iterator begin() { return iterator(*this, 0); }

  iterator end() { return iterator(*this, num_blocks); }

  value_type it;
  camp::idx_t block_size;
  camp::idx_t num_blocks;
  camp::idx_t dist;
};

template <typename TPol, typename EPol, camp::idx_t Index>
struct Executor<Tile<Index, TPol, EPol>> {
  using TileType = Tile<Index, TPol, EPol>;
  template <typename WrappedBody>
  void operator()(TileType const &fp, WrappedBody const &wrap)
  {
    auto const &st = camp::get<Index>(wrap.data.segment_tuple);

    IterableTiler<decltype(st)> tiled_iterable(st, fp.tile_policy.get_chunk_size());

    using ::RAJA::policy::sequential::forall_impl;
    forall_impl(fp.exec_policy, tiled_iterable, TileWrapper<Index, WrappedBody>{wrap});

    // Set range back to original values
    camp::get<Index>(wrap.data.segment_tuple) = tiled_iterable.it;
  }
};


}  // end namespace nested
}  // end namespace RAJA

#endif /* RAJA_pattern_nested_HPP */
