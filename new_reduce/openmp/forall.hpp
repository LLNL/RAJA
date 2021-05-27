#ifndef NEW_REDUCE_FORALL_OMP_HPP
#define NEW_REDUCE_FORALL_OMP_HPP

#if defined(RAJA_ENABLE_OPENMP)
namespace detail {

  template <typename EXEC_POL, typename B, typename... Params>
  std::enable_if_t< std::is_same< EXEC_POL, RAJA::omp_parallel_for_exec>::value >
  forall_param(EXEC_POL&&, int N, B const &body, Params... params)
  {
    FORALL_PARAMS_T<Params...> f_params(params...);

    //init<EXEC_POL>(f_params);

    #pragma omp declare reduction(          \
      combine                               \
      : decltype(f_params)                  \
      : combine<EXEC_POL>(omp_out, omp_in) )\
      initializer(init<EXEC_POL>(omp_priv))

    #pragma omp parallel for reduction(combine : f_params)
    for (int i = 0; i < N; ++i) {
      invoke(f_params, body, i);
    }

    resolve<EXEC_POL>(f_params);
  }

} //  namespace detail
#endif

#endif //  NEW_REDUCE_FORALL_OMP_HPP
