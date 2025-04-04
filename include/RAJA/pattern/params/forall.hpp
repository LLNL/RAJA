#ifndef FORALL_PARAM_HPP
#define FORALL_PARAM_HPP


#include "RAJA/pattern/params/reducer.hpp"
#include "RAJA/util/CombiningAdapter.hpp"

#include "RAJA/pattern/params/params_base.hpp"

namespace RAJA
{
namespace expt
{

template<typename... T>
struct TupleConcatType
{};

template<typename... LParams, typename... RParams>
struct TupleConcatType<camp::tuple<LParams...>, camp::tuple<RParams...>>
{
  using type = camp::tuple<LParams..., RParams...>;
};

template<typename T>
struct FilterOutReducers
{};

template<>
struct FilterOutReducers<camp::tuple<>>
{
  using type = camp::tuple<>;
};

template<typename T>
struct FilterOutReducers<camp::tuple<T>>
{
  using type = typename std::conditional<is_instance_of_reducer<T>::value,
                                         camp::tuple<T>,
                                         camp::tuple<>>::type;
};

template<typename FirstParam, typename... RestofParams>
struct FilterOutReducers<camp::tuple<FirstParam, RestofParams...>>
{
private:
  using rest_of_params_type =
      typename FilterOutReducers<camp::tuple<RestofParams...>>::type;

public:
  using type = typename std::conditional<
      is_instance_of_reducer<FirstParam>::value,
      typename TupleConcatType<camp::tuple<FirstParam>,
                               rest_of_params_type>::type,
      rest_of_params_type>::type;
};

template<typename VOp, typename T, typename Op, typename... Params>
RAJA_HOST_DEVICE constexpr camp::tuple<detail::Reducer<VOp, T, Op>&, Params...>
filter_reducers_impl_helper(detail::Reducer<VOp, T, Op>& red,
                            camp::tuple<Params...> RHS)
{
  return camp::tuple_cat_pair(camp::tuple<detail::Reducer<VOp, T, Op>&>(red),
                              RHS);
}

template<typename LHS, typename... Params>
RAJA_HOST_DEVICE constexpr camp::tuple<Params...> filter_reducers_impl_helper(
    LHS&,
    camp::tuple<Params...> RHS)
{
  return RHS;
}

template<camp::idx_t param_size, typename TupleType>
RAJA_HOST_DEVICE constexpr camp::tuple<> filter_reducers_impl(TupleType&,
                                                              camp::num<0>)
{
  return camp::tuple<> {};
}

template<camp::idx_t param_size, typename TupleType, camp::idx_t idx>
RAJA_HOST_DEVICE constexpr auto filter_reducers_impl(TupleType& param,
                                                     camp::num<idx>)
{
  return filter_reducers_impl_helper(
      camp::get<param_size - idx>(param),
      filter_reducers_impl<param_size>(param, camp::num<idx - 1> {}));
}

template<typename... Params>
RAJA_HOST_DEVICE constexpr auto filter_reducers(camp::tuple<Params...>& params)
{
  return filter_reducers_impl<sizeof...(Params)>(
      params, camp::num<sizeof...(Params)> {});
}

template<typename... Params>
RAJA_HOST_DEVICE constexpr auto filter_reducers_const(
    const camp::tuple<Params...>& params)
{
  return filter_reducers_impl<sizeof...(Params)>(
      params, camp::num<sizeof...(Params)> {});
}

template<typename ExecPol,
         typename ParamTuple,
         camp::idx_t... Seq,
         typename... Args>
void resolve_params_helper(ParamTuple& params_tuple,
                           const camp::idx_seq<Seq...>&,
                           Args&&... args)
{
  CAMP_EXPAND(param_resolve(ExecPol {}, camp::get<Seq>(params_tuple),
                            std::forward<Args>(args)...));
}

template<typename ExecPol, typename... Params, typename... Args>
void resolve_params(camp::tuple<Params...>& params_tuple, Args&&... args)
{
  auto params          = filter_reducers(params_tuple);
  using ParamTupleType = decltype(params);
  resolve_params_helper<ExecPol>(
      params, camp::make_idx_seq_t<camp::tuple_size<ParamTupleType>::value>(),
      std::forward<Args>(args)...);
}

template<typename ExecPol,
         typename ParamTuple,
         camp::idx_t... Seq,
         typename... Args>
void init_params_helper(ParamTuple& params_tuple,
                        const camp::idx_seq<Seq...>&,
                        Args&&... args)
{
  CAMP_EXPAND(param_init(ExecPol {}, camp::get<Seq>(params_tuple),
                         std::forward<Args>(args)...));
}

template<typename ExecPol, typename... Params, typename... Args>
void init_params(camp::tuple<Params...>& params_tuple, Args&&... args)
{
  auto params          = filter_reducers(params_tuple);
  using ParamTupleType = decltype(params);
  init_params_helper<ExecPol>(
      params, camp::make_idx_seq_t<camp::tuple_size<ParamTupleType>::value>(),
      std::forward<Args>(args)...);
}

template<typename ExecPol,
         typename ParamTuple,
         camp::idx_t... Seq,
         typename... Args>
RAJA_HOST_DEVICE void combine_params_helper(const camp::idx_seq<Seq...>&,
                                            ParamTuple& params_tuple)
{
  CAMP_EXPAND(param_combine(ExecPol {}, camp::get<Seq>(params_tuple)));
}

template<typename ExecPol,
         typename ParamTuple,
         camp::idx_t... Seq,
         typename... Args>
RAJA_HOST_DEVICE void combine_params_helper(const camp::idx_seq<Seq...>&,
                                            ParamTuple& params_tuple,
                                            const ParamTuple& params_tuple_in)
{
  CAMP_EXPAND(param_combine(ExecPol {}, camp::get<Seq>(params_tuple),
                            camp::get<Seq>(params_tuple_in)));
}

template<typename ExecPol, typename... Params, typename... Args>
RAJA_HOST_DEVICE void combine_params(camp::tuple<Params...>& params_tuple,
                                     Args&&... args)
{
  using ParamTupleType = camp::decay<decltype(params_tuple)>;
  combine_params_helper<ExecPol>(
      camp::make_idx_seq_t<camp::tuple_size<ParamTupleType>::value>(),
      params_tuple, std::forward<Args>(args)...);
}

//template<typename ExecPol, typename... Params, typename... Args>
//RAJA_HOST_DEVICE void combine_params(camp::tuple<Params...>& params_tuple,
//                                  camp::tuple<Params...>& params_tuple
//                                     Args&&... args)
//{
//  using ParamTupleType = camp::decay<decltype(params_tuple)>;
//  combine_params_helper<ExecPol>(
//      camp::make_idx_seq_t<camp::tuple_size<ParamTupleType>::value>(),
//      params_tuple, std::forward<Args>(args)...);
//}
//
//
//
// Forall Parameter Packing type
//
//
struct ParamMultiplexer;

template<typename... Params>
struct ForallParamPack
{

  friend struct ParamMultiplexer;

  using Base = camp::tuple<Params...>;
  Base param_tup;

  static constexpr size_t param_tup_sz = camp::tuple_size<Base>::value;
  using params_seq                     = camp::make_idx_seq_t<param_tup_sz>;

private:
  // Init
  template<typename EXEC_POL, camp::idx_t... Seq, typename... Args>
  static constexpr void parampack_init(EXEC_POL const& pol,
                                       camp::idx_seq<Seq...>,
                                       ForallParamPack& f_params,
                                       Args&&... args)
  {
    CAMP_EXPAND(param_init(pol, camp::get<Seq>(f_params.param_tup),
                           std::forward<Args>(args)...));
  }

  // Combine
  template<typename EXEC_POL, camp::idx_t... Seq>
  RAJA_HOST_DEVICE static constexpr void parampack_combine(
      EXEC_POL const& pol,
      camp::idx_seq<Seq...>,
      ForallParamPack& out,
      const ForallParamPack& in)
  {
    CAMP_EXPAND(param_combine(pol, camp::get<Seq>(out.param_tup),
                              camp::get<Seq>(in.param_tup)));
  }

  template<typename EXEC_POL, camp::idx_t... Seq>
  RAJA_HOST_DEVICE static constexpr void parampack_combine(
      EXEC_POL const& pol,
      camp::idx_seq<Seq...>,
      ForallParamPack& f_params)
  {
    CAMP_EXPAND(param_combine(pol, camp::get<Seq>(f_params.param_tup)));
  }

  // Resolve
  template<typename EXEC_POL, camp::idx_t... Seq, typename... Args>
  static constexpr void parampack_resolve(EXEC_POL const& pol,
                                          camp::idx_seq<Seq...>,
                                          ForallParamPack& f_params,
                                          Args&&... args)
  {
    CAMP_EXPAND(param_resolve(pol, camp::get<Seq>(f_params.param_tup),
                              std::forward<Args>(args)...));
  }

  // Used to construct the argument TYPES that will be invoked with the lambda.
  template<typename null_t = camp::nil>
  static constexpr auto LAMBDA_ARG_TUP_T()
  {
    return camp::tuple<> {};
  };

  template<typename null_t = camp::nil, typename First>
  static constexpr auto LAMBDA_ARG_TUP_T()
  {
    return typename First::ARG_TUP_T();
  };

  template<typename null_t = camp::nil,
           typename First,
           typename Second,
           typename... Rest>
  static constexpr auto LAMBDA_ARG_TUP_T()
  {
    return camp::tuple_cat_pair(typename First::ARG_TUP_T(),
                                LAMBDA_ARG_TUP_T<camp::nil, Second, Rest...>());
  };

  using lambda_arg_tuple_t = decltype(LAMBDA_ARG_TUP_T<camp::nil, Params...>());

  // Use the size of param_tup to generate the argument list.
  RAJA_HOST_DEVICE constexpr auto LAMBDA_ARG_TUP_V(camp::num<0>)
  {
    return camp::make_tuple();
  }

  RAJA_HOST_DEVICE constexpr auto LAMBDA_ARG_TUP_V(camp::num<1>)
  {
    return camp::get<param_tup_sz - 1>(param_tup).get_lambda_arg_tup();
  }

  template<camp::idx_t N>
  RAJA_HOST_DEVICE constexpr auto LAMBDA_ARG_TUP_V(camp::num<N>)
  {
    return camp::tuple_cat_pair(
        camp::get<param_tup_sz - N>(param_tup).get_lambda_arg_tup(),
        LAMBDA_ARG_TUP_V(camp::num<N - 1>()));
  }

public:
  ForallParamPack() {}

  RAJA_HOST_DEVICE constexpr lambda_arg_tuple_t lambda_args()
  {
    return LAMBDA_ARG_TUP_V(camp::num<sizeof...(Params)>());
  }

  using lambda_arg_seq =
      camp::make_idx_seq_t<camp::tuple_size<lambda_arg_tuple_t>::value>;

  template<typename... Ts>
  ForallParamPack(camp::tuple<Ts...>&& t) : param_tup(std::move(t)) {};

};  // struct ForallParamPack

//===========================================================================
//
//
// ParamMultiplexer is how we hook into the individual calls within forall_impl.
//
//
struct ParamMultiplexer
{
  template<typename EXEC_POL,
           typename... Params,
           typename... Args,
           typename FP = ForallParamPack<Params...>>
  static void constexpr parampack_init(EXEC_POL const& pol,
                                       ForallParamPack<Params...>& f_params,
                                       Args&&... args)
  {
    FP::parampack_init(pol, typename FP::params_seq(), f_params,
                       std::forward<Args>(args)...);
  }

  template<typename EXEC_POL,
           typename... Params,
           typename... Args,
           typename FP = ForallParamPack<Params...>>
  static void constexpr parampack_combine(EXEC_POL const& pol,
                                          ForallParamPack<Params...>& f_params,
                                          Args&&... args)
  {
    FP::parampack_combine(pol, typename FP::params_seq(), f_params,
                          std::forward<Args>(args)...);
  }

  template<typename EXEC_POL,
           typename... Params,
           typename... Args,
           typename FP = ForallParamPack<Params...>>
  static void constexpr parampack_resolve(EXEC_POL const& pol,
                                          ForallParamPack<Params...>& f_params,
                                          Args&&... args)
  {
    FP::parampack_resolve(pol, typename FP::params_seq(), f_params,
                          std::forward<Args>(args)...);
  }
};

//===========================================================================


//===========================================================================
//
//
// ForallParamPack generators.
//
//
RAJA_INLINE static auto get_empty_forall_param_pack()
{
  static ForallParamPack<> p;
  return p;
}

namespace detail
{
// all_true trick to perform variadic expansion in static asserts.
// https://stackoverflow.com/questions/36933176/how-do-you-static-assert-the-values-in-a-parameter-pack-of-a-variadic-template
template<bool...>
struct bool_pack;
template<bool... bs>
using all_true = std::is_same<bool_pack<bs..., true>, bool_pack<true, bs...>>;

template<typename Base, typename... Ts>
using check_types_derive_base =
    all_true<std::is_convertible<Ts, Base>::value...>;
}  // namespace detail

template<typename... Ts>
constexpr auto make_forall_param_pack_from_tuple(camp::tuple<Ts...>&& tuple)
{
  static_assert(detail::check_types_derive_base<detail::ForallParamBase,
                                                camp::decay<Ts>...>::value,
                "Forall optional arguments do not derive ForallParamBase. "
                "Please see Reducer, ReducerLoc and KernelName for examples.");
  return ForallParamPack<camp::decay<Ts>...>(std::move(tuple));
}

// pattern/param/forall.hpp contains a very similar function, but it requires
// passing an rvalue and also strips out the final element from the tuple
template<typename... Params>
constexpr auto make_forall_param_pack_from_tuple(camp::tuple<Params...>& tuple)
{
  return RAJA::expt::ForallParamPack<camp::decay<Params>...>(tuple);
}

namespace detail
{
// Maybe we should do a lot of these with structs...
template<camp::idx_t... Seq, typename TupleType>
constexpr auto tuple_from_seq(const camp::idx_seq<Seq...>&, TupleType&& tuple)
{
  return camp::forward_as_tuple(
      camp::get<Seq>(std::forward<TupleType>(tuple))...);
};

template<typename... Ts>
constexpr auto strip_last_elem(camp::tuple<Ts...>&& tuple)
{
  return tuple_from_seq(camp::make_idx_seq_t<sizeof...(Ts) - 1> {},
                        std::move(tuple));
};
}  // namespace detail

// Make a tuple of the param pack except the final element...
template<typename... Args>
constexpr auto make_forall_param_pack(Args&&... args)
{
  // We assume the last element of the pack is the lambda so we need to strip it
  // from the list.
  auto stripped_arg_tuple = detail::strip_last_elem(
      camp::forward_as_tuple(std::forward<Args>(args)...));
  return make_forall_param_pack_from_tuple(std::move(stripped_arg_tuple));
}

//===========================================================================


//===========================================================================
//
//
// Callable should be the last argument in the param pack, just extract it...
//
//
template<typename... Args>
constexpr auto&& get_lambda(Args&&... args)
{
  return camp::get<sizeof...(Args) - 1>(
      camp::forward_as_tuple(std::forward<Args>(args)...));
}

//===========================================================================


//===========================================================================
//
//
// Checking expected argument list against the assumed lambda.
//
//
namespace detail
{

//
//
// Lambda traits Utilities
//
//
template<class F>
struct lambda_traits;

template<class R, class C, class First, class... Rest>
struct lambda_traits<R (C::*)(First, Rest...)>
{  // non-const specialization
  using arg_type = First;
};

template<class R, class C, class First, class... Rest>
struct lambda_traits<R (C::*)(First, Rest...) const>
{  // const specialization
  using arg_type = First;
};

template<class T>
typename lambda_traits<T>::arg_type* lambda_arg_helper(T);

//
//
// List manipulation Utilities
//
//
template<typename... Ts>
constexpr auto list_remove_pointer(const camp::list<Ts...>&)
{
  return camp::list<camp::decay<typename std::remove_pointer<Ts>::type>...> {};
}

template<typename... Ts>
constexpr auto list_add_lvalue_ref(const camp::list<Ts...>&)
{
  return camp::list<typename std::add_lvalue_reference<Ts>::type...> {};
}

template<typename... Ts>
constexpr auto tuple_to_list(const camp::tuple<Ts...>&)
{
  return camp::list<Ts...> {};
}

// TODO : Change to std::is_invocable at c++17
template<typename F, typename... Args>
struct is_invocable
    : std::is_constructible<
          std::function<void(Args...)>,
          std::reference_wrapper<typename std::remove_reference<F>::type>>
{};

template<class...>
using void_t = void;

template<class F, class = void>
struct has_empty_op : std::false_type
{};

template<class F>
struct has_empty_op<F, void_t<decltype(std::declval<F::operator()>)>>
    : std::true_type
{};

template<class F>
struct get_lambda_index_type
{
  typedef typename std::remove_pointer<decltype(lambda_arg_helper(
      &camp::decay<F>::operator()))>::type type;
};

// If LAMBDA::operator() is not available this probably isn't a generic lambda
// and we can't extract and check args.
template<typename LAMBDA, typename... EXPECTED_ARGS>
constexpr concepts::enable_if<concepts::negate<has_empty_op<LAMBDA>>>
check_invocable(LAMBDA&&, const camp::list<EXPECTED_ARGS...>&)
{}

template<typename LAMBDA, typename... EXPECTED_ARGS>
constexpr concepts::enable_if<has_empty_op<LAMBDA>> check_invocable(
    LAMBDA&&,
    const camp::list<EXPECTED_ARGS...>&)
{
#if !defined(RAJA_ENABLE_HIP)
  static_assert(
      is_invocable<LAMBDA, typename get_lambda_index_type<LAMBDA>::type,
                   EXPECTED_ARGS...>::value,
      "LAMBDA Not invocable w/ EXPECTED_ARGS. Ordering and types must match "
      "between RAJA::expt::Reduce() and ValOp arguments.");
#endif
}

}  // namespace detail

template<typename Lambda, typename ForallParams>
constexpr void check_forall_optional_args(Lambda&& l, ForallParams& fpp)
{

  using expected_arg_type_list = decltype(detail::list_add_lvalue_ref(
      detail::list_remove_pointer(detail::tuple_to_list(fpp.lambda_args()))));

  detail::check_invocable(std::forward<Lambda>(l), expected_arg_type_list {});
}

//===========================================================================


//===========================================================================
//
//
// Type trailts for SFINAE work.
//
//
namespace type_traits
{
template<typename T>
struct is_ForallParamPack : std::false_type
{};

template<typename... Args>
struct is_ForallParamPack<ForallParamPack<Args...>> : std::true_type
{};

template<typename T>
struct is_ForallParamPack_empty : std::true_type
{};

template<typename First, typename... Rest>
struct is_ForallParamPack_empty<ForallParamPack<First, Rest...>>
    : std::false_type
{};

template<>
struct is_ForallParamPack_empty<ForallParamPack<>> : std::true_type
{};
}  // namespace type_traits

//===========================================================================


//===========================================================================
//
//
// Invoke Forall with Params.
//
//
namespace detail
{
template<camp::idx_t Idx, typename FP>
RAJA_HOST_DEVICE constexpr auto get_lambda_args(FP& fpp)
    -> decltype(*camp::get<Idx>(fpp.lambda_args()))
{
  return (*camp::get<Idx>(fpp.lambda_args()));
}

CAMP_SUPPRESS_HD_WARN
template<typename Fn, camp::idx_t... Sequence, typename Params, typename... Ts>
RAJA_HOST_DEVICE constexpr auto invoke_with_order(Params&& params,
                                                  Fn&& f,
                                                  camp::idx_seq<Sequence...>,
                                                  Ts&&... extra)
{
  return f(std::forward<Ts...>(extra...),
           (get_lambda_args<Sequence>(params))...);
}
}  // namespace detail

// CAMP_SUPPRESS_HD_WARN
template<typename Params, typename Fn, typename... Ts>
RAJA_HOST_DEVICE constexpr auto invoke_body(Params&& params,
                                            Fn&& f,
                                            Ts&&... extra)
{
  return detail::invoke_with_order(
      camp::forward<Params>(params), camp::forward<Fn>(f),
      typename camp::decay<Params>::lambda_arg_seq(),
      camp::forward<Ts...>(extra)...);
}

template<typename Fn, typename... Ts>
RAJA_HOST_DEVICE constexpr auto invoke_body_for_wrapper(Fn&& f, Ts&&... extra)
{
  using Params = decltype(f.data.param_tuple);
  return detail::invoke_with_order(
      camp::forward<Params>(f.data.param_tuple), camp::forward<Fn>(f),
      camp::make_idx_seq_t<camp::tuple_size<Params>::value>(),
      camp::forward<Ts...>(extra)...);
}

//===========================================================================

}  //  namespace expt
}  //  namespace RAJA

#endif  //  FORALL_PARAM_HPP
