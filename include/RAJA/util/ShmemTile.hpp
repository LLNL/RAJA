#ifndef RAJA_util_ShmemTile_HPP
#define RAJA_util_ShmemTile_HPP


#include "RAJA/config.hpp"
#include "RAJA/util/StaticLayout.hpp"

#include <iostream>
#include <type_traits>

namespace RAJA
{

/*!
 * Provides a multi-dimensional tiled View of shared memory data.
 *
 * IndexPolicies provide mappings of each dimension into shmem indicies.
 * This is especially useful for mapping global loop indices into cuda block-
 * local indices.
 *
 * The dimension sizes specified are the block-local sizes, and define the
 * amount of shared memory to be requested.
 */
template <typename ShmemPol,
          typename T,
          typename Args,
          typename Sizes,
          typename Segments>
struct ShmemTile;

template <typename ShmemPol,
          typename T,
          camp::idx_t... Args,
          RAJA::Index_type... Sizes,
          typename... Segments>
struct ShmemTile<ShmemPol,
                 T,
                 RAJA::nested::ArgList<Args...>,
                 SizeList<Sizes...>,
                 camp::tuple<Segments...>> {
  static_assert(sizeof...(Args) == sizeof...(Sizes),
                "ArgList and SizeList must be same length");

  using self_t = ShmemTile<ShmemPol,
                           T,
                           RAJA::nested::ArgList<Args...>,
                           SizeList<Sizes...>,
                           camp::tuple<Segments...>>;
  // compute the index tuple that nested::forall is going to use
  using segment_tuple_t = camp::tuple<Segments...>;
  using index_tuple_t = RAJA::nested::internal::index_tuple_from_segments<
      typename segment_tuple_t::TList>;

  // compute the indices that we are going to use
  using arg_tuple_t =
      camp::tuple<camp::at_v<typename index_tuple_t::TList, Args>...>;

  // typed layout to map indices to shmem space
  using layout_t =
      RAJA::TypedStaticLayout<typename arg_tuple_t::TList, Sizes...>;

  // shared memory object type
  using shmem_t = SharedMemory<ShmemPol, T, layout_t::s_size>;
  using element_t = T;
  shmem_t shmem;


  RAJA_SUPPRESS_HD_WARN
  RAJA_INLINE
  RAJA_HOST_DEVICE
  constexpr ShmemTile() : shmem() {}

  RAJA_SUPPRESS_HD_WARN
  RAJA_INLINE
  RAJA_HOST_DEVICE
  ShmemTile(ShmemTile const &c) : shmem(c.shmem) {}


  RAJA_SUPPRESS_HD_WARN
  RAJA_INLINE
  RAJA_HOST_DEVICE
  element_t &operator()(
      camp::at_v<typename index_tuple_t::TList, Args>... idx) const
  {
#ifdef __CUDA_ARCH__

// Get the shared memory window
// (stored at beginning of CUDA dynamic shared memory region)

#if 0
      // BROKEN w/ cuda 9.0.176
      extern __shared__ char my_ptr[];
      index_tuple_t *shmem_window_ptr = reinterpret_cast<index_tuple_t *>(&my_ptr[0]);
      auto lin = layout_t::s_oper((idx - camp::get<Args>(*shmem_window_ptr))...);
#endif

#if 0
      // WORKS w/ cuda 9.0.176
      extern __shared__ char my_ptr[];
      index_tuple_t *shmem_window_ptr = reinterpret_cast<index_tuple_t *>(&my_ptr[0]);
			index_tuple_t shmem_window = *shmem_window_ptr;
      auto lin = layout_t::s_oper((idx - camp::get<Args>(shmem_window))...);
#endif

#if 0
      // BROKEN w/ cuda 9.0.176
      extern __shared__ index_tuple_t my_ptr[];
      auto lin = layout_t::s_oper((idx - camp::get<Args>(my_ptr[0]))...);
#endif

#if 0
      // BROKEN w/ cuda 9.0.176
      __syncthreads();
      extern __shared__ index_tuple_t my_ptr[];
      index_tuple_t const &shmem_window = my_ptr[0];
      auto lin = layout_t::s_oper((idx - camp::get<Args>(shmem_window))...);
#endif

#if 0
      // BROKEN w/ cuda 9.0.176
      extern __shared__ index_tuple_t my_ptr[];
      index_tuple_t shmem_window = my_ptr[0];
      auto lin = layout_t::s_oper((idx - camp::get<Args>(shmem_window))...);
#endif


#if 1
    // WORKS w/ cuda 9.0.176
    int *my_ptr = internal::cuda_get_shmem_ptr<int>();
    auto lin = layout_t::s_oper((idx - my_ptr[Args])...);
#endif


    return shmem[lin];

#else
    index_tuple_t const *shmem_window_ptr =
        static_cast<index_tuple_t *>(RAJA::detail::getSharedMemoryWindow());
    return shmem[layout_t::s_oper(
        (idx - camp::get<Args>(*shmem_window_ptr))...)];

#endif
  }
};


}  // end namespace RAJA


#endif
