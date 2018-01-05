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
template <camp::idx_t Arg0, camp::idx_t Arg1, typename... EnclosedStmts>
struct StatementExecutor<Collapse<seq_exec, ArgList<Arg0, Arg1>, EnclosedStmts...>> {

  template <typename WrappedBody>
  void operator()(WrappedBody const &wrap)
  {
    auto b0 = std::begin(camp::get<Arg0>(wrap.data.segment_tuple));
    auto b1 = std::begin(camp::get<Arg1>(wrap.data.segment_tuple));

    auto e0 = std::end(camp::get<Arg0>(wrap.data.segment_tuple));
    auto e1 = std::end(camp::get<Arg1>(wrap.data.segment_tuple));

    // Skip a level
    for (auto i0 = b0; i0 < e0; ++i0) {
      wrap.data.template assign_index<Arg0>(*i0);
      for (auto i1 = b1; i1 < e1; ++i1) {
        wrap.data.template assign_index<Arg1>(*i1);
        wrap();
      }
    }
  }
};


} // namespace internal

}  // end namespace nested
}  // end namespace RAJA



#endif /* RAJA_pattern_nested_HPP */
