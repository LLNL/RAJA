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

//        printf("Window (%p): ", shmem_window_ptr);
//        VarOps::ignore_args(
//            printf("%d ", convertIndex<int>(camp::get<Args>(window)))...
//        );
//        printf("\n");
      }
//      else{
//        printf("No shmem window ptr\n");
//      }
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




template <typename... EnclosedStmts>
struct StatementExecutor<SetShmemWindow<EnclosedStmts...>> {


  template <typename Data>
  static
  RAJA_INLINE
  void exec(Data && data)
  {
    // Grab pointer to shared shmem window
    using loop_data_t = camp::decay<Data>;
    using index_tuple_t = typename loop_data_t::index_tuple_t;
    index_tuple_t *shmem_window = static_cast<index_tuple_t*>(detail::getSharedMemoryWindow());

    if(shmem_window != nullptr){
//      printf("Setting shmem window %p\n", shmem_window);

      // Set the window by copying the current index_tuple to the shared location
      *shmem_window = data.index_tuple;

      // Privatize to invoke copy ctors
      loop_data_t private_data = data;

      // Invoke the enclosed statements
      execute_statement_list<camp::list<EnclosedStmts...>>(private_data);
    }
    else{
      // No shared memory setup, so this becomes a NOP
//      printf("SetShmemWindow, but no window configured\n");
      execute_statement_list<camp::list<EnclosedStmts...>>(std::forward<Data>(data));
    }
  }
};





} // namespace internal
}  // end namespace nested
}  // end namespace RAJA



#endif /* RAJA_pattern_nested_HPP */
