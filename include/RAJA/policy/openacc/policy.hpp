#ifndef policy_openacc_HXX_
#define policy_openacc_HXX_

#include "RAJA/policy/PolicyBase.hpp"

#include <utility>

#if defined(RAJA_ENABLE_VERBOSE)
#if !defined(RAJA_VERBOSE)
#define RAJA_VERBOSE(A) [[deprecated(A)]]
#else
#define RAJA_VERBOSE(A)
#endif
#endif

namespace RAJA
{

//
//////////////////////////////////////////////////////////////////////
//
// Execution policies
//
//////////////////////////////////////////////////////////////////////
//
namespace acc
{

namespace num
{
template <unsigned int N>
struct gangs {
  static constexpr unsigned int num_gangs = N;
};
template <unsigned int N>
struct workers {
  static constexpr unsigned int num_workers = N;
};
template <unsigned int N>
struct vectors {
  static constexpr unsigned int num_vectors = N;
};
}

struct independent {
  static constexpr bool is_independent = true;
};
struct gang {
  static constexpr bool is_gang = true;
};
struct worker {
  static constexpr bool is_worker = true;
};
struct vector {
  static constexpr bool is_vector = true;
};

template <typename... T>
struct config : public T... {
};

namespace has
{
template <class T>
class num_gangs
{
  template <typename U,
            typename = typename std::enable_if<!std::is_member_pointer<decltype(
                &U::num_gangs)>::value>::type>
  static std::true_type check(int);
  template <typename>
  static std::false_type check(...);

public:
  static constexpr bool value = decltype(check<T>(0))::value;
};

template <class T>
class num_workers
{
  template <typename U,
            typename = typename std::enable_if<!std::is_member_pointer<decltype(
                &U::num_workers)>::value>::type>
  static std::true_type check(int);
  template <typename>
  static std::false_type check(...);

public:
  static constexpr bool value = decltype(check<T>(0))::value;
};

template <class T>
class num_vectors
{
  template <typename U,
            typename = typename std::enable_if<!std::is_member_pointer<decltype(
                &U::num_vectors)>::value>::type>
  static std::true_type check(int);
  template <typename>
  static std::false_type check(...);

public:
  static constexpr bool value = decltype(check<T>(0))::value;
};

template <class T>
class is_independent
{
  template <typename U,
            typename = typename std::enable_if<!std::is_member_pointer<decltype(
                &U::is_independent)>::value>::type>
  static std::true_type check(int);
  template <typename>
  static std::false_type check(...);

public:
  static constexpr bool value = decltype(check<T>(0))::value;
};

template <class T>
class is_gang
{
  template <typename U,
            typename = typename std::enable_if<!std::is_member_pointer<decltype(
                &U::is_gang)>::value>::type>
  static std::true_type check(int);
  template <typename>
  static std::false_type check(...);

public:
  static constexpr bool value = decltype(check<T>(0))::value;
};

template <class T>
class is_worker
{
  template <typename U,
            typename = typename std::enable_if<!std::is_member_pointer<decltype(
                &U::is_worker)>::value>::type>
  static std::true_type check(int);
  template <typename>
  static std::false_type check(...);

public:
  static constexpr bool value = decltype(check<T>(0))::value;
};

template <class T>
class is_vector
{
  template <typename U,
            typename = typename std::enable_if<!std::is_member_pointer<decltype(
                &U::is_vector)>::value>::type>
  static std::true_type check(int);
  template <typename>
  static std::false_type check(...);

public:
  static constexpr bool value = decltype(check<T>(0))::value;
};
}
}


namespace detail
{

template <bool...>
struct blist;

template <bool... B>
struct all_of : std::is_same<blist<true, B...>, blist<B..., true>> {
};

template <typename T>
struct Not {
  static constexpr bool value = !T::value;
};

template <typename Config, template <typename> class... Type>
struct Check {
  static constexpr bool value = all_of<Type<Config>::value...>::value;
};
}

namespace no
{
template <typename T>
using n_gang = detail::Not<acc::has::num_gangs<T>>;
template <typename T>
using n_worker = detail::Not<acc::has::num_workers<T>>;
template <typename T>
using n_vector = detail::Not<acc::has::num_vectors<T>>;
template <typename T>
using gang = detail::Not<acc::has::is_gang<T>>;
template <typename T>
using worker = detail::Not<acc::has::is_worker<T>>;
template <typename T>
using vector = detail::Not<acc::has::is_vector<T>>;
template <typename T>
using ind = detail::Not<acc::has::is_independent<T>>;
}

namespace yes
{
template <typename T>
using n_gang = acc::has::num_gangs<T>;
template <typename T>
using n_worker = acc::has::num_workers<T>;
template <typename T>
using n_vector = acc::has::num_vectors<T>;
template <typename T>
using gang = acc::has::is_gang<T>;
template <typename T>
using worker = acc::has::is_worker<T>;
template <typename T>
using vector = acc::has::is_vector<T>;
template <typename T>
using ind = acc::has::is_independent<T>;
}

template <typename Conf, template <typename> class... TArgs>
using When =
    typename std::enable_if<detail::Check<Conf, TArgs...>::value>::type;

///
/// Segment execution policies
///
template <typename InnerPolicy, typename Config = acc::config<>>
struct acc_parallel_exec : public RAJA::wrap<InnerPolicy> {
};
template <typename InnerPolicy, typename Config = acc::config<>>
struct acc_kernels_exec : public RAJA::wrap<InnerPolicy> {
};

template <typename Config = acc::config<>>
struct acc_loop_exec : public RAJA::make_policy_pattern<RAJA::Policy::openacc,
                                                        RAJA::Pattern::forall> {
};

template <typename Config = acc::config<>, typename InnerConfig = acc::config<>>
using acc_parallel_loop_exec =
    acc_parallel_exec<acc_loop_exec<InnerConfig>, Config>;

template <typename Config = acc::config<>, typename InnerConfig = acc::config<>>
using acc_kernels_loop_exec =
    acc_kernels_exec<acc_loop_exec<InnerConfig>, Config>;


}  // closing brace for RAJA namespace

#endif
