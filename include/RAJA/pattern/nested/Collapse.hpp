#ifndef RAJA_pattern_nested_Collapse_HPP
#define RAJA_pattern_nested_Collapse_HPP

namespace RAJA
{

namespace nested
{


template <typename ExecPolicy, typename ForList, typename... EnclosedStmts>
struct Collapse : public internal::ForList, public internal::CollapseBase,
                  public internal::Statement<EnclosedStmts...> {};


template <typename ... Fors>
using CollapseList = camp::tuple<Fors...>;


template <typename ExecPolicy, typename... Fors, typename... EnclosedStmts>
struct Collapse<ExecPolicy, CollapseList<Fors...>, EnclosedStmts...> :
                  public internal::ForList, public internal::CollapseBase,
                  public internal::Statement<EnclosedStmts...> {
  //using as_for_list = CollapseList<Fors...>;

  // used for execution space resolution
  using as_space_list = camp::list<For<-1, ExecPolicy>>;


  const ExecPolicy exec_policy;
  RAJA_HOST_DEVICE constexpr Collapse() : exec_policy{} {}
  RAJA_HOST_DEVICE constexpr Collapse(ExecPolicy const &ep) : exec_policy{ep} {}
};




namespace internal{



//
// This is for demonstration only... can be removed eventually
//
template <typename FT0, typename FT1, typename... EnclosedStmts>
struct StatementExecutor<Collapse<seq_exec, camp::tuple<FT0, FT1>, EnclosedStmts...>> {
  static_assert(std::is_base_of<internal::ForBase, FT0>::value,
                "Only For-based policies should get here");
  static_assert(std::is_base_of<internal::ForBase, FT1>::value,
                "Only For-based policies should get here");
  template <typename WrappedBody>
  void operator()(Collapse<seq_exec, CollapseList<FT0, FT1>, EnclosedStmts...> const &, WrappedBody const &wrap)
  {
    auto b0 = std::begin(camp::get<FT0::index_val>(wrap.data.segment_tuple));
    auto b1 = std::begin(camp::get<FT1::index_val>(wrap.data.segment_tuple));

    auto e0 = std::end(camp::get<FT0::index_val>(wrap.data.segment_tuple));
    auto e1 = std::end(camp::get<FT1::index_val>(wrap.data.segment_tuple));

    // Skip a level
    for (auto i0 = b0; i0 < e0; ++i0) {
      wrap.data.template assign_index<FT0::index_val>(*i0);
      for (auto i1 = b1; i1 < e1; ++i1) {
        wrap.data.template assign_index<FT1::index_val>(*i1);
        wrap();
      }
    }
  }
};


} // namespace internal

}  // end namespace nested
}  // end namespace RAJA



#endif /* RAJA_pattern_nested_HPP */
