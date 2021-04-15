#ifndef FORALL_PARAM_HPP
#define FORALL_PARAM_HPP

int use_dev = 1;
namespace detail
{

  //
  //
  // Invoke Forall with Params.
  //
  //
  CAMP_SUPPRESS_HD_WARN
  template <typename Fn,
            camp::idx_t... Sequence,
            typename Params,
            typename... Ts>
  CAMP_HOST_DEVICE constexpr auto invoke_with_order(Params&& params,
                                                    Fn&& f,
                                                    camp::idx_seq<Sequence...>,
                                                    Ts&&... extra)
  {
    return f(extra..., ( params.template get_param_ref<Sequence>() )...);
  }

  CAMP_SUPPRESS_HD_WARN
  template <typename Params, typename Fn, typename... Ts>
  CAMP_HOST_DEVICE constexpr auto invoke(Params&& params, Fn&& f, Ts&&... extra)
  {
    return invoke_with_order(
        camp::forward<Params>(params),
        camp::forward<Fn>(f),
        typename camp::decay<Params>::params_seq(),
        camp::forward<Ts>(extra)...);
  }

  //
  //
  // Forall param type thing..
  //
  //
  template<typename... Params>
  struct FORALL_PARAMS_T {
    using Base = camp::tuple<Params...>;
    using params_seq = camp::make_idx_seq_t< camp::tuple_size<Base>::value >;
    Base param_tup;

  private:
    template<camp::idx_t... Seq>
    constexpr auto m_param_refs(camp::idx_seq<Seq...>) -> decltype( camp::make_tuple( (&camp::get<Seq>(param_tup).val)...) ) {
      return camp::make_tuple( (&camp::get<Seq>(param_tup).val)...) ;
    }

    // Init
    template<typename EXEC_POL, camp::idx_t... Seq>
    friend void constexpr detail_init(EXEC_POL, FORALL_PARAMS_T& f_params, camp::idx_seq<Seq...>) {
      camp::make_tuple( (init<EXEC_POL>( camp::get<Seq>(f_params.param_tup) ))...  );
    }
    // Combine
    template<typename EXEC_POL, camp::idx_t... Seq>
    friend void constexpr detail_combine(EXEC_POL, FORALL_PARAMS_T& out, const FORALL_PARAMS_T& in, camp::idx_seq<Seq...>) {
      camp::make_tuple( (combine<EXEC_POL>( camp::get<Seq>(out.param_tup), camp::get<Seq>(in.param_tup)))...  );
    }
    // Resolve
    template<typename EXEC_POL, camp::idx_t... Seq>
    friend void constexpr detail_resove(EXEC_POL, FORALL_PARAMS_T& f_params, camp::idx_seq<Seq...>) {
      camp::make_tuple( (resolve<EXEC_POL>( camp::get<Seq>(f_params.param_tup) ))...  );
    }

  public:
    FORALL_PARAMS_T (){}
    FORALL_PARAMS_T(Params... params) {
      param_tup = camp::make_tuple(params...);
    };

    template<camp::idx_t Idx>
    constexpr auto get_param_ref() -> decltype(*camp::get<Idx>(m_param_refs(params_seq{}))) {
      return (*camp::get<Idx>(m_param_refs(params_seq{})));
    }

    // Init
    template<typename EXEC_POL>
    friend void constexpr init( FORALL_PARAMS_T& f_params ) {
      detail_init(EXEC_POL(), f_params, params_seq{} );
    }
    // Combine
    template<typename EXEC_POL>
    friend void constexpr combine(FORALL_PARAMS_T& out, const FORALL_PARAMS_T& in) {
      detail_combine(EXEC_POL(), out, in, params_seq{} );
    }
    // Resolve
    template<typename EXEC_POL>
    friend void constexpr resolve( FORALL_PARAMS_T& f_params ) {
      detail_resove(EXEC_POL(), f_params, params_seq{} );
    }
  };

} //  namespace detail

#include "sequential/forall.hpp"
#include "openmp/forall.hpp"
#include "omp-target/forall.hpp"

template<typename ExecPol, typename B, typename... Params>
void forall_param(int N, const B& body, Params... params) {
  detail::forall_param(ExecPol(), N, body, params...);
}

#endif //  FORALL_PARAM_HPP
