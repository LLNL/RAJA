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



template <typename... EnclosedStmts, typename IndexCalc>
struct CudaStatementExecutor<SetShmemWindow<EnclosedStmts...>, IndexCalc> {

  using stmt_list_t = StatementList<EnclosedStmts...>;

  template <typename Data>
  static
  RAJA_DEVICE
  inline
  void exec(Data &data, int logical_block)
  {
    // Divine the type of the index tuple in wrap.data
    using loop_data_t = camp::decay<Data>;
    using index_tuple_t = camp::decay<typename loop_data_t::index_tuple_t>;

    // Grab a pointer to the shmem window tuple.  We are assuming that this
    // is the first thing in the dynamic shared memory
    extern __shared__ char my_ptr[];
    index_tuple_t *shmem_window = reinterpret_cast<index_tuple_t *>(&my_ptr[0]);

    // Set the shared memory tuple with the beginning of our segments
    *shmem_window = data.index_tuple;

    // make sure we're all synchronized
    __syncthreads();

    // Thread privatize, triggering Shmem objects to grab updated window info
    //loop_data_t private_data = data;

    auto private_data = privatize_bodies(data);

    // execute enclosed statements
    cuda_execute_statement_list<stmt_list_t, IndexCalc>(private_data, logical_block);
  }


  template<typename Data>
  static
  RAJA_INLINE
  LaunchDim calculateDimensions(Data const &data, LaunchDim const &max_physical){

    // Return launch dimensions of enclosed statements
    return cuda_calcdims_statement_list<stmt_list_t, IndexCalc>(data, max_physical);
  }
};





}  // namespace internal
}  // end namespace nested
}  // end namespace RAJA



#endif
