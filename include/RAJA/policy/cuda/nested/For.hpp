#ifndef RAJA_policy_cuda_nested_For_HPP
#define RAJA_policy_cuda_nested_For_HPP


#include "RAJA/config.hpp"
#include "RAJA/policy/cuda/nested.hpp"


namespace RAJA
{

namespace nested
{


struct cuda_thread_exec {};

namespace internal{





template <camp::idx_t ArgumentId, typename... EnclosedStmts>
struct CudaStatementExecutor<For<ArgumentId, seq_exec, EnclosedStmts...>> {

  using ForType = For<ArgumentId, seq_exec, EnclosedStmts...>;

  template <typename WrappedBody>
  RAJA_INLINE
  RAJA_DEVICE
  void operator()(ForType const &fp, WrappedBody const &wrap, CudaExecInfo &exec_info)
  {
    // Get the segment referenced by this For statement
    auto const &iter = camp::get<ArgumentId>(wrap.data.segment_tuple);

    // Pull out iterators
    auto begin = iter.begin();  // std::begin(iter);
    auto end = iter.end();      // std::end(iter);

    // compute trip count
    auto len = end - begin;  // std::distance(begin, end);


    for (decltype(len) i = 0; i < len; ++i) {
      wrap.data.template assign_index<ArgumentId>(i);
      wrap(exec_info);
    }
  }
};



template <camp::idx_t ArgumentId, typename... EnclosedStmts>
struct CudaStatementExecutor<For<ArgumentId, cuda_thread_exec, EnclosedStmts...>> {

  using ForType = For<ArgumentId, cuda_thread_exec, EnclosedStmts...>;

  template <typename WrappedBody>
  RAJA_INLINE
  RAJA_DEVICE
  void operator()(ForType const &fp, WrappedBody const &wrap, CudaExecInfo &exec_info)
  {
    // Get the segment referenced by this For statement
    auto const &iter = camp::get<ArgumentId>(wrap.data.segment_tuple);

    // Pull out iterators
    auto begin = iter.begin();  // std::begin(iter);
    auto end = iter.end();      // std::end(iter);

    // compute trip count
    auto len = end - begin;  // std::distance(begin, end);


    // How many batches of threads do we need?
    int num_batches = len / exec_info.threads_left;
    if(num_batches*exec_info.threads_left < len){
      num_batches++;
    }

    // compute our starting index
    int i = exec_info.thread_id;

    for(int batch = 0;batch < num_batches;++ batch){

      if(i < len){
        wrap.data.template assign_index<ArgumentId>(*(begin+i));
        wrap(exec_info);
      }

      i += exec_info.threads_left;
    }
  }
};



} // namespace internal
}  // end namespace nested
}  // end namespace RAJA



#endif /* RAJA_pattern_nested_HPP */
