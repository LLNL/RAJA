#ifndef RAJA_policy_openmp_nested_multi_HPP
#define RAJA_policy_openmp_nested_multi_HPP



#ifdef RAJA_ENABLE_OPENMP

namespace RAJA
{

template<bool Async = false>
struct omp_multi_exec{};

namespace nested
{



namespace detail {

template<bool Async, size_t i, size_t N>
struct InvokeLoopsOpenMP {

  template<typename ... LoopList>
  void operator()(camp::tuple<LoopList...> const &loops) const {

    auto loop_data = camp::get<i>(loops);
    RAJA::nested::forall(loop_data.pt, loop_data.st, loop_data.f);

    if(!Async){
#pragma omp barrier
    }

    InvokeLoopsOpenMP<Async, i+1, N> next_invoke;
    next_invoke(loops);
  }

};


template<bool Async, size_t N>
struct InvokeLoopsOpenMP<Async, N, N> {

  template<typename ... LoopList>
  void operator()(camp::tuple<LoopList...> const &) const {
  }

};


}



template <bool Async,
          typename ... LoopList>
RAJA_INLINE void forall_multi(
    omp_multi_exec<Async>,
    LoopList const & ... loops)
{

  // Invoke each loop, one after the other,
  detail::InvokeLoopsOpenMP<Async, 0, sizeof...(LoopList)> loop_invoke;

#pragma omp parallel
  loop_invoke(camp::tuple<LoopList const &...>(loops...));

}



}  // end namespace nested
}  // end namespace RAJA


#endif // RAJA_ENABLE_OPENMP

#endif /* RAJA_pattern_nested_HPP */
