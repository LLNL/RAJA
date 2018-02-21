#ifndef RAJA_policy_cuda_nested_Thread_HPP
#define RAJA_policy_cuda_nested_Thread_HPP


#include "RAJA/config.hpp"
#include "RAJA/util/defines.hpp"
#include "RAJA/util/types.hpp"

#include "RAJA/policy/cuda/nested/internal.hpp"

#include <iostream>
#include <type_traits>

namespace RAJA
{
namespace nested
{

template <typename... EnclosedStmts>
struct Thread : public internal::Statement<EnclosedStmts...>{
};

namespace internal
{



template <typename... EnclosedStmts, typename IndexCalc>
struct CudaStatementExecutor<Thread<EnclosedStmts...>, IndexCalc> {

  using stmt_list_t = StatementList<EnclosedStmts...>;




  template <typename Data>
  static
  RAJA_DEVICE
  inline
  void exec(Data &data, int num_logical_blocks, int logical_block)
  {

    if(logical_block <= 0){
      // Get physical parameters
      LaunchDim max_physical(gridDim.x, blockDim.x);

      // Compute logical dimensions
      IndexCalc index_calc(data.segment_tuple, max_physical);

      // set indices to beginning of each segment, and increment
      // to this threads first iteration
      bool done = index_calc.assignBegin(data, threadIdx.x);

      while(!done) {

        // Since we are consuming everything in IndexCalc, start over
        using index_calc_t = CudaIndexCalc_Terminator<typename Data::segment_tuple_t>;

        // execute enclosed statements
        cuda_execute_statement_list<stmt_list_t, index_calc_t>(data, num_logical_blocks, logical_block);


        // increment to next thread
        done = index_calc.increment(data, blockDim.x);

      }
    }
	
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
