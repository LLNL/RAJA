#ifndef FORALL_PARAM_HPP
#define FORALL_PARAM_HPP

#include "RAJA/policy/sequential/new_reduce.hpp"
#include "RAJA/policy/openmp/new_reduce.hpp"
#include "RAJA/policy/cuda/params/new_reduce.hpp"
#include "RAJA/policy/cuda/params/kernel_name.hpp"
#include "RAJA/policy/hip/params/new_reduce.hpp"

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

    RAJA_HOST_DEVICE
    constexpr
    auto lambda_args(camp::idx_seq<> )
    {
      return camp::make_tuple();
    }

    template<camp::idx_t Seq>
    RAJA_HOST_DEVICE
    constexpr
    auto lambda_args(camp::idx_seq<Seq> )
    {
      return camp::get<Seq>(param_tup).get_lambda_arg_tup();
    }

    template<camp::idx_t First, camp::idx_t Second, camp::idx_t... Seq>
    RAJA_HOST_DEVICE
    constexpr
    auto lambda_args(camp::idx_seq<First, Second, Seq...> )
    {
      return camp::tuple_cat_pair(
               camp::get<First>(param_tup).get_lambda_arg_tup(),
               lambda_args(camp::idx_seq<Second, Seq...>())
             );
    }

    ForallParamPack(camp::tuple<Params...> t) : param_tup(t) {};

    using lambda_params_seq = camp::make_idx_seq_t<count_lambda_args<camp::nil, Params...>()>;
  }; // struct ForallParamPack 
  
  //===========================================================================
  RAJA_INLINE static auto get_empty_forall_param_pack(){
    static ForallParamPack<> p;
    return p;
  }

  template<camp::idx_t Idx, typename FP>
  RAJA_HOST_DEVICE
  constexpr
  auto get_lambda_args(FP& fpp)
      -> decltype(  *camp::get<Idx>( fpp.lambda_args(typename FP::params_seq()) )  ) {
    return (  *camp::get<Idx>( fpp.lambda_args(typename FP::params_seq()) )  );
  }
  //===========================================================================
  
  //===========================================================================
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
  //===========================================================================

  //===========================================================================
  // DoesThisGoInCamp
  namespace dtgic {

    template<camp::idx_t... Seq, typename... Ts>
    constexpr auto tuple_from_seq (const camp::idx_seq<Seq...>&, const camp::tuple<Ts...>& tuple){
      return camp::make_tuple( camp::get< Seq >(tuple)... );
    };

    template<typename... Ts>
    constexpr auto strip_last_elem(const camp::tuple<Ts...>& tuple){
      return tuple_from_seq(camp::make_idx_seq_t<sizeof...(Ts)-1>{},tuple);
    };

    template<typename First, typename... Ts>
    constexpr auto strip_first_elem(const camp::list<First, Ts...>&){
      return camp::list<Ts...>{};
    }

    template<typename First, typename... Ts>
    constexpr auto strip_first_elem(const camp::tuple<First, Ts...>&){
      return camp::tuple<Ts...>{};
    }

    template<typename... Ts>
    constexpr auto decay_remove_pointer_list(const camp::list<Ts...>&){
      return camp::list<camp::decay<typename std::remove_pointer<Ts>::type>...>{};
    }

    template<typename... Ts>
    constexpr auto tuple_to_list(const camp::tuple<Ts...>&) {
      return camp::list<Ts...>{};
    }

    template<typename... Ts>
    using get_last_t = camp::at<camp::list<Ts...>, camp::num<sizeof...(Ts)>>; 
    
    // all_true trick to perform variadic expansion in static asserts.
    // https://stackoverflow.com/questions/36933176/how-do-you-static-assert-the-values-in-a-parameter-pack-of-a-variadic-template
    template<bool...> struct bool_pack;
    template<bool... bs>
    using all_true = std::is_same<bool_pack<bs..., true>, bool_pack<true, bs...>>;

    template<typename Base, typename... Ts>
    using check_types_derive_base = dtgic::all_true<std::is_convertible<Ts, Base>::value...>;

    template <typename F, typename... Args>
    struct is_invocable :
      std::is_constructible<
        std::function<void(Args ...)>,
        std::reference_wrapper<typename std::remove_reference<F>::type>
      >{};

  } //  namespace dtgic

  //===========================================================================


  //===========================================================================
  // ForallParamPAck generators.
  template<typename... Ts>
  constexpr auto make_forall_param_pack_from_tuple(const camp::tuple<Ts...>& tuple) {
    static_assert(dtgic::check_types_derive_base<detail::ForallParamBase, Ts...>::value,
        "Forall optional arguments do not derive ForallParamBase. Please see Reducer, ReducerLoc and KernelName for examples.") ;
    return ForallParamPack<Ts...>(tuple);
  }

  // Make a tuple of the param pack except the final element...
  template<typename... Args>
  constexpr auto make_forall_param_pack(Args&&... args){
    // We assume the last element of the pack is the lambda so we need to strip it from the list.
    auto stripped_arg_tuple = dtgic::strip_last_elem(camp::make_tuple(args...)); 
    return make_forall_param_pack_from_tuple(stripped_arg_tuple);
  }
  //===========================================================================


  //===========================================================================
  

  template<typename T>
  struct lambda_arg_list : public lambda_arg_list<decltype(&T::operator())> {};


  template<typename ReturnT, typename ClassT, typename... Args>
  struct lambda_arg_list<ReturnT(ClassT::*)(Args...) const> {
    using type = camp::list<Args...>;
  
  };

  //===========================================================================


  // Lambda should be the last argument in the param pack, just extract it...
  template<typename... Args>
  constexpr auto get_lambda(Args&&... args){
    auto lambda = camp::get<sizeof...(Args)-1>( camp::make_tuple(args...) );
    return lambda; 
  } 

  template<typename Lambda, typename ForallParams>
  constexpr void check_forall_optional_args(const Lambda&, ForallParams& fpp) {
    using l_args = typename lambda_arg_list<Lambda>::type;

    using user_type_list = decltype( dtgic::decay_remove_pointer_list( dtgic::strip_first_elem( l_args{} ) ) );
    using expected_type_list = decltype( dtgic::decay_remove_pointer_list( dtgic::tuple_to_list( fpp.lambda_args(typename ForallParams::params_seq())) ) );

    static_assert(std::is_same<user_type_list, expected_type_list>::value, "Incorrect lambda argument types for optional Forall parameters.");
  }

  namespace type_traits
  {
    template <typename T> struct is_ForallParamPack : std::false_type {};
    template <typename... Args> struct is_ForallParamPack<ForallParamPack<Args...>> : std::true_type {};

    template <typename T> struct is_ForallParamPack_empty : std::true_type {};
    template <typename First, typename... Rest> struct is_ForallParamPack_empty<ForallParamPack<First, Rest...>> : std::false_type {};
    template <> struct is_ForallParamPack_empty<ForallParamPack<>> : std::true_type {};
  }

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
    //return f(std::forward<Ts...>(extra...), ( params.template get_lambda_args<Sequence>() )...);
    return f(std::forward<Ts...>(extra...), ( get_lambda_args<Sequence>(params) )...);
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

} //  namespace expt
} //  namespace RAJA

#endif //  FORALL_PARAM_HPP
