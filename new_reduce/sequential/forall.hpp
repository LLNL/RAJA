#ifndef PROTO_NEW_REDUCE_FORALL_SEQ_HPP
#define PROTO_NEW_REDUCE_FORALL_SEQ_HPP

namespace detail {

  template <typename EXEC_POL, typename B, typename... Params>
  std::enable_if_t< std::is_same< EXEC_POL, RAJA::seq_exec>::value >
  forall_param(EXEC_POL&&, int N, B const &body, Params... params)
  {
    ForallParamPack<Params...> f_params(params...);

    init<EXEC_POL>(f_params);

    for (int i = 0; i < N; ++i) {
      invoke(f_params, body, i);
    }

    resolve<EXEC_POL>(f_params);
  }

} //  namespace detail

#endif //  PROTO_NEW_REDUCE_SEQ_HPP
