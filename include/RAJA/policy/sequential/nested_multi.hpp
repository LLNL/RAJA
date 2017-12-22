#ifndef RAJA_policy_sequential_nested_multi_HPP
#define RAJA_policy_sequential_nested_multi_HPP

#include <RAJA/pattern/nested_multi.hpp>

namespace RAJA
{

struct seq_multi_exec{};

namespace nested
{
namespace detail
{

template<size_t i, size_t N>
struct InvokeLoopsSequential {

  template<typename ... LoopList>
  void operator()(camp::tuple<LoopList...> const &loops) const {

    auto loop_data = camp::get<i>(loops);
    RAJA::nested::forall(loop_data.pt, loop_data.st, loop_data.f);

    InvokeLoopsSequential<i+1, N> next_invoke;
    next_invoke(loops);
  }

};


template<size_t N>
struct InvokeLoopsSequential<N, N> {

  template<typename ... LoopList>
  void operator()(camp::tuple<LoopList...> const &) const {
  }

};

} //namespace detail




template <typename ... LoopList>
RAJA_INLINE void forall_multi(
    seq_multi_exec,
    LoopList const & ... loops)
{

  // Invoke each loop, one after the other,
  detail::InvokeLoopsSequential<0, sizeof...(LoopList)> loop_invoke;
  loop_invoke(camp::tuple<LoopList const &...>(loops...));
}


}  // end namespace nested
}  // end namespace RAJA



#endif /* RAJA_pattern_nested_HPP */
