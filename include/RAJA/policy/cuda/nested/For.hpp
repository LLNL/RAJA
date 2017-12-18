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
  void operator()(ForType const &fp, WrappedBody const &wrap)
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
      wrap();
    }
  }
};



template <camp::idx_t ArgumentId, typename... EnclosedStmts>
struct CudaStatementExecutor<For<ArgumentId, cuda_thread_exec, EnclosedStmts...>> {

  using ForType = For<ArgumentId, cuda_thread_exec, EnclosedStmts...>;

  template <typename WrappedBody>
  RAJA_INLINE
  RAJA_DEVICE
  void operator()(ForType const &fp, WrappedBody const &wrap)
  {
    // Get the segment referenced by this For statement
    auto const &iter = camp::get<ArgumentId>(wrap.data.segment_tuple);

    // Pull out iterators
    auto begin = iter.begin();  // std::begin(iter);
    auto end = iter.end();      // std::end(iter);

    // compute trip count
    auto len = end - begin;  // std::distance(begin, end);


    // compute our index
    int i = threadIdx.x;

    wrap.data.template assign_index<ArgumentId>(*(begin+i));
    wrap();
  }
};



} // namespace internal
}  // end namespace nested
}  // end namespace RAJA



#endif /* RAJA_pattern_nested_HPP */
