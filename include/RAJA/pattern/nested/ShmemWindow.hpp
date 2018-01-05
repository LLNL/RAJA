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
struct SetShmemWindow : public internal::Statement<camp::nil, EnclosedStmts...>
{
};



/*!
 * Provides a multi-dimensional View of shared memory data.
 *
 * IndexPolicies provide mappings of each dimension into shmem indicies.
 * This is especially useful for mapping global loop indices into cuda block-
 * local indices.
 *
 * The dimension sizes specified are the block-local sizes, and define the
 * amount of shared memory to be requested.
 */
template<typename ShmemT, typename Args, typename Sizes, typename Segments>
struct ShmemWindowView {
};

template<typename ShmemT, camp::idx_t ... Args, RAJA::Index_type ... Sizes, typename ... Segments>
struct ShmemWindowView<ShmemT, ArgList<Args...>, SizeList<Sizes...>, camp::tuple<Segments...>>
{
    static_assert(sizeof...(Args) == sizeof...(Sizes), "ArgList and SizeList must be same length");

    using self_t = ShmemWindowView<ShmemT, ArgList<Args...>, SizeList<Sizes...>, camp::tuple<Segments...>>;
    // compute the index tuple that nested::forall is going to use
    using segment_tuple_t = camp::tuple<Segments...>;
    using index_tuple_t = RAJA::nested::internal::index_tuple_from_segments<typename segment_tuple_t::TList>;

    // compute the indices that we are going to use
    using arg_tuple_t = camp::tuple<camp::at_v<typename index_tuple_t::TList, Args>...>;

    // shared memory object type
    using shmem_t = ShmemT;
    using element_t = typename ShmemT::element_t;
    shmem_t shmem;

    // typed layout to map indices to shmem space
    using layout_t = RAJA::TypedStaticLayout<typename arg_tuple_t::TList, Sizes...>;
    static_assert(layout_t::s_size <= ShmemT::size, "Layout cannot span a larger size than the shared memory");

    index_tuple_t window;

    RAJA_SUPPRESS_HD_WARN
    RAJA_INLINE
    RAJA_HOST_DEVICE
    constexpr
    ShmemWindowView() : shmem(), window() {}

    RAJA_SUPPRESS_HD_WARN
    RAJA_INLINE
    RAJA_HOST_DEVICE
    ShmemWindowView(ShmemWindowView const &c) : shmem(c.shmem), window(c.window) {
#ifdef __CUDA_ARCH__
      // Grab a pointer to the shmem window tuple.  We are assuming that this
      // is the first thing in the dynamic shared memory
      extern __shared__ char my_ptr[];
      index_tuple_t *shmem_window_ptr = reinterpret_cast<index_tuple_t *>(&my_ptr[0]);
#else
      auto shmem_window_ptr = static_cast<index_tuple_t*>(RAJA::detail::getSharedMemoryWindow());
#endif
      if(shmem_window_ptr != nullptr){
        index_tuple_t &shmem_window = *shmem_window_ptr;
        window = shmem_window;
      }
    }


    RAJA_SUPPRESS_HD_WARN
    RAJA_INLINE
    RAJA_HOST_DEVICE
    constexpr
    element_t &operator()(camp::at_v<typename index_tuple_t::TList, Args> ... idx) const {
      return shmem[layout_t::s_oper((idx - camp::get<Args>(window))...)];
    }
};




namespace internal{


template<camp::idx_t ... Seq, typename ... IdxTypes, typename ... Segments>
RAJA_INLINE
RAJA_HOST_DEVICE
void set_shmem_window_tuple_expanded(camp::idx_seq<Seq...>, camp::tuple<IdxTypes...> &window, camp::tuple<Segments...> const &segment_tuple){
//  VarOps::ignore_args(
//        (printf("set_shmem_window_tuple: window[%d]=%d\n", (int)Seq, (int)**camp::get<Seq>(segment_tuple).begin()))...
//        );
  VarOps::ignore_args(
      (camp::get<Seq>(window) = *camp::get<Seq>(segment_tuple).begin())...
      );
}

template<typename ... IdxTypes, typename ... Segments>
RAJA_INLINE
RAJA_HOST_DEVICE
void set_shmem_window_tuple(camp::tuple<IdxTypes...> &window, camp::tuple<Segments...> const &segment_tuple){
  using loop_idx = typename camp::make_idx_seq<sizeof...(IdxTypes)>::type;

  set_shmem_window_tuple_expanded(loop_idx{}, window, segment_tuple);
}


template <typename... EnclosedStmts>
struct StatementExecutor<SetShmemWindow<EnclosedStmts...>> {


  template <typename WrappedBody>
  RAJA_INLINE
  void operator()(WrappedBody const &wrap)
  {
    // Grab pointer to shared shmem window
    using loop_data_t = camp::decay<decltype(wrap.data)>;
    using index_tuple_t = typename loop_data_t::index_tuple_t;
    index_tuple_t *shmem_window = static_cast<index_tuple_t*>(detail::getSharedMemoryWindow());

    if(shmem_window != nullptr){

      // Set the window by copying the current index_tuple to the shared location
      set_shmem_window_tuple(*shmem_window, wrap.data.segment_tuple);

      // Privatize to invoke copy ctors
      auto privatizer = thread_privatize(wrap);
      auto &private_wrap = privatizer.get_priv();

      // Invoke the enclosed statements
      private_wrap();
    }
    else{
      // No shared memory setup, so this becomes a NOP
      wrap();
    }
  }
};





} // namespace internal
}  // end namespace nested
}  // end namespace RAJA



#endif /* RAJA_pattern_nested_HPP */
