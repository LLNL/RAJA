#ifndef NEW_REDUCE_FORALL_TBB_HPP
#define NEW_REDUCE_FORALL_TBB_HPP

#if defined(RAJA_ENABLE_TBB)
#include <tbb/tbb.h>

namespace detail {

  template <typename EXEC_POL, typename B, typename... Params>
  std::enable_if_t< std::is_same< EXEC_POL, RAJA::tbb_for_dynamic>::value >
  forall_param(EXEC_POL&&, int N, B const &loop_body, Params... params)
  {
    FORALL_PARAMS_T<Params...> f_params(params...);

    EXEC_POL p;
    using brange = ::tbb::blocked_range<size_t>;
    ::tbb::parallel_for(brange(0, N, p.grain_size), [=, &f_params](const brange& r) {

      init<EXEC_POL>(f_params);

      using RAJA::internal::thread_privatize;
      auto privatizer = thread_privatize(loop_body);
      auto body = privatizer.get_priv();

      for (int i = 0; i < N; ++i) {
        invoke(f_params, loop_body, i);
      }

    });
    resolve<EXEC_POL>(f_params);
  }

} //  namespace detail

#endif  // closing endif for if defined(RAJA_ENABLE_TBB)

#endif //  NEW_REDUCE_SEQ_HPP
