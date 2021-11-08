#ifndef PROTO_FORALL_PARAM_HPP
#define PROTO_FORALL_PARAM_HPP

// Used in omp reduction
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
  RAJA_HOST_DEVICE constexpr auto invoke_with_order(Params&& params,
                                                    Fn&& f,
                                                    camp::idx_seq<Sequence...>,
                                                    Ts&&... extra)
  {
    return f(extra..., ( params.template get_lambda_args<Sequence>() )...);
  }

  CAMP_SUPPRESS_HD_WARN
  template <typename Params, typename Fn, typename... Ts>
  RAJA_HOST_DEVICE constexpr auto invoke(Params&& params, Fn&& f, Ts&&... extra)
  {
    return invoke_with_order(
        camp::forward<Params>(params),
        camp::forward<Fn>(f),
        typename camp::decay<Params>::lambda_params_seq(),
        camp::forward<Ts>(extra)...);
  }


  //
  //
  // Forall Parameter Packing type
  //
  //
  template<typename... Params>
  struct ForallParamPack {
    using Base = camp::tuple<Params...>;
    using params_seq = camp::make_idx_seq_t< camp::tuple_size<Base>::value >;
    Base param_tup;

  private:

    template<camp::idx_t Seq>
    RAJA_HOST_DEVICE
    constexpr auto lambda_args(camp::idx_seq<Seq> )
    {
      return camp::get<Seq>(param_tup).get_lambda_arg_tup();
    }

    template<camp::idx_t First, camp::idx_t Second, camp::idx_t... Seq>
    RAJA_HOST_DEVICE
    constexpr auto lambda_args(camp::idx_seq<First, Second, Seq...> )
    {
      return camp::tuple_cat_pair(
               camp::get<First>(param_tup).get_lambda_arg_tup(),
               lambda_args(camp::idx_seq<Second, Seq...>())
             );
    }

    // Init
    template<typename EXEC_POL, camp::idx_t... Seq, typename ...Args>
    static void constexpr detail_init(EXEC_POL, camp::idx_seq<Seq...>, ForallParamPack& f_params, Args&& ...args) {
      CAMP_EXPAND(init<EXEC_POL>( camp::get<Seq>(f_params.param_tup), std::forward<Args>(args)... ));
    }

    // Combine
    template<typename EXEC_POL, camp::idx_t... Seq>
    RAJA_HOST_DEVICE
    static void constexpr detail_combine(EXEC_POL, camp::idx_seq<Seq...>, ForallParamPack& out, const ForallParamPack& in ) {
      CAMP_EXPAND(combine<EXEC_POL>( camp::get<Seq>(out.param_tup), camp::get<Seq>(in.param_tup)));
    }

    template<typename EXEC_POL, camp::idx_t... Seq>
    RAJA_HOST_DEVICE
    static void constexpr detail_combine(EXEC_POL, camp::idx_seq<Seq...>, ForallParamPack& f_params ) {
      CAMP_EXPAND(combine<EXEC_POL>( camp::get<Seq>(f_params.param_tup) ));
    }
    
    // Resolve
    template<typename EXEC_POL, camp::idx_t... Seq>
    static void constexpr detail_resolve(EXEC_POL, camp::idx_seq<Seq...>, ForallParamPack& f_params ) {
      CAMP_EXPAND(resolve<EXEC_POL>( camp::get<Seq>(f_params.param_tup) ));
    }

    template<typename Last>
    static size_t constexpr count_lambda_args() { return Last::num_lambda_args; }
    template<typename First, typename Second, typename... Rest>
    static size_t constexpr count_lambda_args() { return First::num_lambda_args + count_lambda_args<Second, Rest...>(); }

  public:
    ForallParamPack (){}
    ForallParamPack(Params... params) {
      param_tup = camp::make_tuple(params...);
    };
    ForallParamPack(camp::tuple<Params...> tuple) : param_tup(tuple) {};

    using lambda_params_seq = camp::make_idx_seq_t<count_lambda_args<Params...>()>;

    template<camp::idx_t Idx>
    RAJA_HOST_DEVICE
    constexpr auto get_lambda_args()
        -> decltype(  *camp::get<Idx>( lambda_args(params_seq{}) )  ) {
      return (  *camp::get<Idx>( lambda_args(params_seq{}) )  );
    }

    // Init
    template<typename EXEC_POL, typename ...Args>
    friend void constexpr init( ForallParamPack& f_params, Args&& ...args) {
      detail_init(EXEC_POL(), params_seq{}, f_params, std::forward<Args>(args)... );
    }

    // Combine
    template<typename EXEC_POL, typename ...Args>
    RAJA_HOST_DEVICE
    friend void constexpr combine(ForallParamPack& f_params, Args&& ...args) {
      detail_combine(EXEC_POL(), params_seq{}, f_params, std::forward<Args>(args)... );
    }

    // Resolve
    template<typename EXEC_POL, typename ...Args>
    friend void constexpr resolve( ForallParamPack& f_params, Args&& ...args) {
      detail_resolve(EXEC_POL(), params_seq{}, f_params , std::forward<Args>(args)... );
    }
  };


  //TODO :: Figure out where this tuple malarky should go ...
  //===========================================================================
  // Should this go in camp?
  template<camp::idx_t... Seq, typename... Ts>
  constexpr auto tuple_from_seq (const camp::idx_seq<Seq...>&, const camp::tuple<Ts...>& tuple){
    return camp::make_tuple( camp::get< Seq >(tuple)... );
  };

  // Should this go in camp?
  template<typename... Ts>
  constexpr auto strip_last_elem(const camp::tuple<Ts...>& tuple){
    return tuple_from_seq(camp::make_idx_seq_t<sizeof...(Ts)-1>{},tuple);
  };

    template<typename... Args>
    constexpr auto get_param_tuple(Args&&... args){
      return strip_last_elem(camp::make_tuple(args...));
    }

    template<typename... Ts>
    constexpr auto make_forall_param_pack_from_tuple(const camp::tuple<Ts...>& tuple) {
      return ForallParamPack<Ts...>(tuple);
    }

  //===========================================================================

  // Make a tuple of the param pack except the final element...
  template<typename... Args>
  constexpr auto make_forall_param_pack(Args&&... args){
    return make_forall_param_pack_from_tuple( get_param_tuple(args...) );
  }

  // Lambda should be the last argument in the param pack, just extract it...
  template<typename... Args>
  constexpr auto get_lambda(Args&&... args){
    return camp::get<sizeof...(Args)-1>( camp::make_tuple(args...) ); 
  } 


} //  namespace detail

#include "sequential/forall.hpp"
#include "openmp/forall.hpp"
#include "omp-target/forall.hpp"
#include "cuda/forall.hpp"
#include "hip/forall.hpp"
#include "sycl/forall.hpp"

template<typename ExecPol, typename... Params>
void forall_param(int N, Params... params) {
  auto f_params = detail::make_forall_param_pack(params...);
  auto body = detail::get_lambda(params...);

  detail::forall_param(ExecPol(), N, body, f_params);
}

#endif //  PROTO_FORALL_PARAM_HPP
