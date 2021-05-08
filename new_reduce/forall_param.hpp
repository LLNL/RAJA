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
#if defined(RAJA_ENABLE_CUDA)
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
  CAMP_HOST_DEVICE constexpr auto cuda_invoke_with_order(Params&& params,
                                                    Fn&& f,
                                                    camp::idx_seq<Sequence...>,
                                                    Ts&&... extra)
  {
    return f(extra..., ( params.template cuda_get_param_ref<Sequence>() )...);
  }

  CAMP_SUPPRESS_HD_WARN
  template <typename Params, typename Fn, typename... Ts>
  CAMP_HOST_DEVICE constexpr auto cuda_invoke(Params&& params, Fn&& f, Ts&&... extra)
  {
    return cuda_invoke_with_order(
        camp::forward<Params>(params),
        camp::forward<Fn>(f),
        typename camp::decay<Params>::params_seq(),
        camp::forward<Ts>(extra)...);
  }
#endif

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
#if defined(RAJA_ENABLE_CUDA)
    template<camp::idx_t... Seq>
    constexpr auto cuda_m_param_refs(camp::idx_seq<Seq...>) -> decltype( camp::make_tuple( (camp::get<Seq>(param_tup).cudaval)...) ) {
      return camp::make_tuple( (camp::get<Seq>(param_tup).cudaval)...) ;
    }
#endif

    // Init
    template<typename EXEC_POL, camp::idx_t... Seq, typename ...Args>
    friend void constexpr detail_init(EXEC_POL, FORALL_PARAMS_T& f_params, camp::idx_seq<Seq...>, Args&& ...args) {
      CAMP_EXPAND(init<EXEC_POL>( camp::get<Seq>(f_params.param_tup), std::forward<Args>(args)... ));
    }
    template<typename EXEC_POL, camp::idx_t... Seq>
    RAJA_HOST_DEVICE
    friend void constexpr detail_combine(EXEC_POL, FORALL_PARAMS_T& out, const FORALL_PARAMS_T& in, camp::idx_seq<Seq...>) {
      CAMP_EXPAND(combine<EXEC_POL>( camp::get<Seq>(out.param_tup), camp::get<Seq>(in.param_tup)));
    }
    template<typename EXEC_POL, camp::idx_t... Seq>
    RAJA_HOST_DEVICE
    friend void constexpr detail_combine(EXEC_POL, FORALL_PARAMS_T& f_params, camp::idx_seq<Seq...>) {
      camp::make_tuple( (combine<EXEC_POL>( camp::get<Seq>(f_params.param_tup) ))... );
      //CAMP_EXPAND(combine<EXEC_POL>( camp::get<Seq>(f_params.param_tup)));
      //CAMP_EXPAND(printf("Seq : %d\n", Seq));
    }
    // Resolve
    template<typename EXEC_POL, camp::idx_t... Seq>
    RAJA_HOST_DEVICE
    friend void constexpr detail_resolve(EXEC_POL, FORALL_PARAMS_T& f_params, camp::idx_seq<Seq...>) {
      CAMP_EXPAND(resolve<EXEC_POL>( camp::get<Seq>(f_params.param_tup) ));
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
#if defined(RAJA_ENABLE_CUDA)
    template<camp::idx_t Idx>
    constexpr auto cuda_get_param_ref() -> decltype(*camp::get<Idx>(cuda_m_param_refs(params_seq{}))) {
      return (*camp::get<Idx>(cuda_m_param_refs(params_seq{})));
    }
#endif

    // Init
    template<typename EXEC_POL, typename ...Args>
    friend void constexpr init( FORALL_PARAMS_T& f_params, Args&& ...args) {
      detail_init(EXEC_POL(), f_params, params_seq{}, std::forward<Args>(args)... );
    }
    template<typename EXEC_POL>
    RAJA_HOST_DEVICE
    friend void constexpr combine(FORALL_PARAMS_T& out, const FORALL_PARAMS_T& in) {
      detail_combine(EXEC_POL(), out, in, params_seq{} );
    }
    template<typename EXEC_POL>
    RAJA_HOST_DEVICE
    friend void constexpr combine(FORALL_PARAMS_T& f_params) {
      detail_combine(EXEC_POL(), f_params, params_seq{} );
    }
    // Resolve
    template<typename EXEC_POL>
    RAJA_HOST_DEVICE
    friend void constexpr resolve( FORALL_PARAMS_T& f_params ) {
      detail_resolve(EXEC_POL(), f_params, params_seq{} );
    }
  };

} //  namespace detail

#include "sequential/forall.hpp"
#include "openmp/forall.hpp"
#include "omp-target/forall.hpp"
#include "cuda/forall.hpp"

template<typename ExecPol, typename B, typename... Params>
void forall_param(int N, const B& body, Params... params) {
  detail::forall_param(ExecPol(), N, body, params...);
}

#endif //  FORALL_PARAM_HPP
