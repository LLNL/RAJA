#ifndef FORALL_PARAM_HPP
#define FORALL_PARAM_HPP

#include "RAJA/policy/sequential/new_reduce.hpp"
#include "RAJA/policy/openmp/new_reduce.hpp"

#if defined(RAJA_EXPT_FORALL)
#define RAJA_EXPT_FORALL_WARN(Msg)
#else
#define RAJA_EXPT_FORALL_WARN(Msg) RAJA_DEPRECATE(Msg)
#endif

namespace RAJA
{
namespace expt
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
    return f(std::forward<Ts...>(extra...), ( params.template get_lambda_args<Sequence>() )...);
  }

  //CAMP_SUPPRESS_HD_WARN
  template <typename Params, typename Fn, typename... Ts>
  RAJA_HOST_DEVICE constexpr auto invoke_body(Params&& params, Fn&& f, Ts&&... extra)
  {
    return expt::invoke_with_order(
        camp::forward<Params>(params),
        camp::forward<Fn>(f),
        typename camp::decay<Params>::lambda_params_seq(),
        camp::forward<Ts...>(extra)...);
  }


  //
  //
  // Forall Parameter Packing type
  //
  //
  struct ParamMultiplexer;

  template<typename... Params>
  struct ForallParamPack {

    friend struct ParamMultiplexer;

    using Base = camp::tuple<Params...>;
    using params_seq = camp::make_idx_seq_t< camp::tuple_size<Base>::value >;
    Base param_tup;

  private:

    RAJA_HOST_DEVICE
    constexpr auto lambda_args(camp::idx_seq<> )
    {
      return camp::make_tuple();
    }

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
      CAMP_EXPAND(expt::detail::init<EXEC_POL>( camp::get<Seq>(f_params.param_tup), std::forward<Args>(args)... ));
    }

    // Combine
    template<typename EXEC_POL, camp::idx_t... Seq>
    RAJA_HOST_DEVICE
    static void constexpr detail_combine(EXEC_POL, camp::idx_seq<Seq...>, ForallParamPack& out, const ForallParamPack& in ) {
      CAMP_EXPAND(detail::combine<EXEC_POL>( camp::get<Seq>(out.param_tup), camp::get<Seq>(in.param_tup)));
    }

    template<typename EXEC_POL, camp::idx_t... Seq>
    RAJA_HOST_DEVICE
    static void constexpr detail_combine(EXEC_POL, camp::idx_seq<Seq...>, ForallParamPack& f_params ) {
      CAMP_EXPAND(detail::combine<EXEC_POL>( camp::get<Seq>(f_params.param_tup) ));
    }
    
    // Resolve
    template<typename EXEC_POL, camp::idx_t... Seq>
    static void constexpr detail_resolve(EXEC_POL, camp::idx_seq<Seq...>, ForallParamPack& f_params ) {
      CAMP_EXPAND(detail::resolve<EXEC_POL>( camp::get<Seq>(f_params.param_tup) ));
    }

    template<typename null_t = camp::nil>
    static size_t constexpr count_lambda_args() { return 0; }
    template<typename null_t = camp::nil, typename Last>
    static size_t constexpr count_lambda_args() { return Last::num_lambda_args; }
    template<typename null_t = camp::nil, typename First, typename Second, typename... Rest>
    static size_t constexpr count_lambda_args() { return First::num_lambda_args + count_lambda_args<camp::nil, Second, Rest...>(); }

  public:
    ForallParamPack(){}

    //ForallParamPack(Params... params) {
    //  param_tup = camp::make_tuple(params...);
    //};

    ForallParamPack(camp::tuple<Params...> t) : param_tup(t) {};

    using lambda_params_seq = camp::make_idx_seq_t<count_lambda_args<camp::nil, Params...>()>;

    template<camp::idx_t Idx>
    RAJA_HOST_DEVICE
    constexpr auto get_lambda_args()
        -> decltype(  *camp::get<Idx>( lambda_args(params_seq{}) )  ) {
      return (  *camp::get<Idx>( lambda_args(params_seq{}) )  );
    }
  }; // struct ForallParamPack 
  
  
  struct ParamMultiplexer {
    template<typename EXEC_POL, typename... Params, typename ...Args, typename FP = ForallParamPack<Params...>>
    static void constexpr init( ForallParamPack<Params...>& f_params, Args&& ...args) {
      FP::detail_init(EXEC_POL(),typename FP::params_seq(), f_params, std::forward<Args>(args)... );
    }
    template<typename EXEC_POL, typename... Params, typename ...Args, typename FP = ForallParamPack<Params...>>
    static void constexpr combine(ForallParamPack<Params...>& f_params, Args&& ...args){
      FP::detail_combine(EXEC_POL(), typename FP::params_seq(), f_params, std::forward<Args>(args)... );
    }
    template<typename EXEC_POL, typename... Params, typename ...Args, typename FP = ForallParamPack<Params...>>
    static void constexpr resolve( ForallParamPack<Params...>& f_params, Args&& ...args){
      FP::detail_resolve(EXEC_POL(), typename FP::params_seq(), f_params, std::forward<Args>(args)... );
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

  //TODO :: static asserts here?
  // Make a tuple of the param pack except the final element...
  template<typename... Args>
  constexpr auto make_forall_param_pack(Args&&... args){
    return make_forall_param_pack_from_tuple( get_param_tuple(args...) );
  }

  //TODO :: static asserts here?
  // Lambda should be the last argument in the param pack, just extract it...
  template<typename... Args>
  constexpr auto get_lambda(Args&&... args){
    return camp::get<sizeof...(Args)-1>( camp::make_tuple(args...) ); 
  } 

  RAJA_INLINE static auto get_empty_forall_param_pack(){
    static ForallParamPack<> p;
    return p;
  }

  namespace type_traits
  {
    template <typename T> struct is_ForallParamPack : std::false_type {};
    template <typename... Args> struct is_ForallParamPack<ForallParamPack<Args...>> : std::true_type {};

    template <typename T> struct is_ForallParamPack_empty : std::true_type {};
    template <typename First, typename... Rest> struct is_ForallParamPack_empty<ForallParamPack<First, Rest...>> : std::false_type {};
    template <> struct is_ForallParamPack_empty<ForallParamPack<>> : std::true_type {};
  }

} //  namespace expt
} //  namespace RAJA

#endif //  FORALL_PARAM_HPP
