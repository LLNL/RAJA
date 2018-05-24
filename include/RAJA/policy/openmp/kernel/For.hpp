#ifndef 
template <camp::idx_t ArgumentId,
          int N,
          typename... EnclosedStmts>
struct StatementExecutor<statement::For<ArgumentId, omp_parallel_for_exec<N>, EnclosedStmts...>> {
  template <typename Data>
  static RAJA_INLINE void exec(Data &&data)
  {
    // Create a wrapper, just in case forall_impl needs to thread_privatize
    ForWrapper<ArgumentId, Data, EnclosedStmts...> for_wrapper(data);

    auto len = segment_length<ArgumentId>(data);
    using len_t = decltype(len);

    forall_impl(ExecPolicy{}, TypedRangeSegment<len_t>(0, len), for_wrapper);
  }
};
