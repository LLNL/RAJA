#ifndef RAJA_policy_cuda_nested_ShmemWindow_HPP
#define RAJA_policy_cuda_nested_ShmemWindow_HPP


#include "RAJA/config.hpp"
#include "RAJA/util/defines.hpp"
#include "RAJA/util/types.hpp"

#include "RAJA/policy/cuda/nested/internal.hpp"
#include "RAJA/pattern/nested/ShmemWindow.hpp"

#include <iostream>
#include <type_traits>

namespace RAJA
{
namespace nested
{
namespace internal
{




template <typename... EnclosedStmts>
struct CudaStatementExecutor<SetShmemWindow<EnclosedStmts...>> {

  template <typename WrappedBody, typename Data>
  static
  RAJA_DEVICE
  void exec(WrappedBody const &wrap, Data &data, CudaExecInfo &exec_info)
  {
    // Divine the type of the index tuple in wrap.data
    using loop_data_t = camp::decay<Data>;
    using index_tuple_t = typename loop_data_t::index_tuple_t;

    // Grab a pointer to the shmem window tuple.  We are assuming that this
    // is the first thing in the dynamic shared memory
    extern __shared__ char my_ptr[];
    index_tuple_t *shmem_window = reinterpret_cast<index_tuple_t *>(&my_ptr[0]);

    // Set the shared memory tuple with the beginning of our segments
    set_shmem_window_tuple(*shmem_window, data.segment_tuple);

    // make sure we're all synchronized
    __syncthreads();

    // Thread privatize, triggering Shmem objects to grab updated window info
    using RAJA::internal::thread_privatize;
    auto privatizer = thread_privatize(data);
    auto &private_data = privatizer.get_priv();

    // Execute enclosed statements
    wrap(private_data, exec_info);
  }
};





}  // namespace internal
}  // end namespace nested
}  // end namespace RAJA



#endif
