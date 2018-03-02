#ifndef RAJA_pattern_nested_Collapse_HPP
#define RAJA_pattern_nested_Collapse_HPP

namespace RAJA
{

namespace nested
{


template <typename ExecPolicy, typename ForList, typename... EnclosedStmts>
struct Collapse : public internal::ForList, public internal::CollapseBase,
                  public internal::Statement<ExecPolicy, EnclosedStmts...> {

};







namespace internal{


//
// This is for demonstration only... can be removed eventually
//


template <camp::idx_t Arg0, typename... EnclosedStmts>
struct StatementExecutor<Collapse<seq_exec, ArgList<Arg0>, EnclosedStmts...>> {

  template <typename Data>
  static
  RAJA_INLINE
  void exec(Data &data)
  {
    auto len0 = segment_length<Arg0>(data);

    for (auto i0 = 0; i0 < len0; ++i0) {
      data.template assign_offset<Arg0>(i0);
      execute_statement_list<camp::list<EnclosedStmts...>>(data);
    }
  }
};


template <camp::idx_t Arg0, camp::idx_t Arg1, typename... EnclosedStmts>
struct StatementExecutor<Collapse<seq_exec, ArgList<Arg0, Arg1>, EnclosedStmts...>> {

  template <typename Data>
  static
  RAJA_INLINE
  void exec(Data &data)
  {
    auto len0 = segment_length<Arg0>(data);
    auto len1 = segment_length<Arg1>(data);

    // Skip a level
    for (auto i0 = 0; i0 < len0; ++i0) {
      data.template assign_offset<Arg0>(i0);
      for (auto i1 = 0; i1 < len1; ++i1) {
        data.template assign_offset<Arg1>(i1);

        execute_statement_list<camp::list<EnclosedStmts...>>(data);
      }
    }
  }
};


} // namespace internal

}  // end namespace nested
}  // end namespace RAJA



#endif /* RAJA_pattern_nested_HPP */
