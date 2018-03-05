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





}  // end namespace nested
}  // end namespace RAJA



#endif /* RAJA_pattern_nested_HPP */
