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



template <typename Data, typename... EnclosedStmts, typename IndexCalc>
struct CudaStatementExecutor<Data, SetShmemWindow<EnclosedStmts...>, IndexCalc> {

  using stmt_list_t = StatementList<EnclosedStmts...>;

  using enclosed_stmts_t = CudaStatementListExecutor<Data, stmt_list_t, IndexCalc>;
  enclosed_stmts_t enclosed_stmts;

  IndexCalc index_calc;

  template <camp::idx_t... RangeInts>
  RAJA_DEVICE
  inline
  void setWindow(Data &data, camp::idx_seq<RangeInts...> const &){
    // get shmem window
    int *shmem_window = RAJA::internal::cuda_get_shmem_ptr<int>();

    // get the index value tuple
    auto index_tuple = data.get_begin_index_tuple();

    // Assign each index to the window
    VarOps::ignore_args((shmem_window[RangeInts] =
        RAJA::convertIndex<int>(camp::get<RangeInts>(index_tuple)))...);
  }

  inline
  __device__
  void exec(Data &data, int num_logical_blocks, int block_carry)
  {

    // make sure all threads are done with current window
    __syncthreads();
      
    // Grab a pointer to the shmem window tuple.  We are assuming that this
    // is the first thing in the dynamic shared memory
    if(threadIdx.x == 0){

      data.assign_begin_all();

      // Compute logical dimensions
      index_calc.assignBegin(data, 0, 0);


      // Set the shared memory tuple with the beginning of our segments
      using IndexRange = camp::make_idx_seq_t<Data::index_tuple_t::TList::size>;
      setWindow(data, IndexRange{});

    }

    // make sure we're all synchronized, so they all see the same window
		__syncthreads();

		// execute enclosed statements
    enclosed_stmts.exec(data, num_logical_blocks, block_carry);
	
	}


  inline
  RAJA_DEVICE
  void initBlocks(Data &data, int num_logical_blocks, int block_stride)
  {
    enclosed_stmts.initBlocks(data, num_logical_blocks, block_stride);
  }

  RAJA_INLINE
  LaunchDim calculateDimensions(Data const &data, LaunchDim const &max_physical){

    return enclosed_stmts.calculateDimensions(data, max_physical);

  }
};





}  // namespace internal
}  // end namespace nested
}  // end namespace RAJA



#endif
