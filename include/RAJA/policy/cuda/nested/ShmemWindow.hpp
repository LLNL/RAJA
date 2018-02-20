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


  template <typename Data, camp::idx_t... RangeInts>
  static
  RAJA_DEVICE
  inline
  void setWindow(Data &data, camp::idx_seq<RangeInts...> const &){
    // get shmem window
    extern __shared__ int shmem_window[];

    // Assign each index to the window
    VarOps::ignore_args((shmem_window[RangeInts] =
        RAJA::convertIndex<int>(camp::get<RangeInts>(data.index_tuple)))...);
  }

  template <typename Data>
  static
  RAJA_DEVICE
  inline
  void exec(Data &data, long num_logical_blocks, long logical_block)
  {
    // Get physical parameters
    LaunchDim max_physical(gridDim.x, blockDim.x);

    // Compute logical dimensions
    IndexCalc index_calc(data.segment_tuple, max_physical);

    // Divine the type of the index tuple in wrap.data
    using loop_data_t = camp::decay<Data>;
    using index_tuple_t = camp::decay<typename loop_data_t::index_tuple_t>;

    // make sure all threads are done with current window
    __syncthreads();
      
		data.assign_begin_all();

		//index_calc.assignBegin(data, threadIdx.x);
		index_calc.assignBegin(data, 0);

    // Grab a pointer to the shmem window tuple.  We are assuming that this
    // is the first thing in the dynamic shared memory
    if(threadIdx.x == 0){

      // Grab shmem window pointer
      //extern __shared__ int my_ptr[];
      //index_tuple_t *shmem_window = reinterpret_cast<index_tuple_t *>(&my_ptr[0]);

      // Set the shared memory tuple with the beginning of our segments
      using IndexRange = camp::make_idx_seq_t<Data::index_tuple_t::TList::size>;
      setWindow(data, IndexRange{});

      //*shmem_window = data.index_tuple;
			
    }

    // make sure we're all synchronized, so they all see the same window
		__syncthreads();

		// privatize data to invoke copy ctors that will capture new window
		//loop_data_t private_data = data;
    
		// execute enclosed statements
    //cuda_execute_statement_list<stmt_list_t, IndexCalc>(private_data, num_logical_blocks, logical_block);
    cuda_execute_statement_list<stmt_list_t, IndexCalc>(data, num_logical_blocks, logical_block);
	
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
