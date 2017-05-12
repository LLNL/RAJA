#ifndef RAJA_POLICYBASE_HXX
#define RAJA_POLICYBASE_HXX

#include <stddef.h>

namespace RAJA
{

enum class Policy { undefined, sequential, simd, openmp, cuda, cilk };

enum class Launch { undefined, sync, async };

enum class Pattern {
  undefined,
  forall,
  reduce,
  taskgraph,
};

struct PolicyBase {
};

template <Policy P = Policy::undefined,
          Launch L = Launch::undefined,
          Pattern Pat = Pattern::undefined>
struct PolicyBaseT : public PolicyBase {
  static constexpr Policy policy = P;
  static constexpr Launch launch = L;
  static constexpr Pattern pattern = Pat;
};

template <typename Inner, typename... T>
struct WrapperPolicy : public Inner {
  using inner = Inner;
};

// "makers"

template <typename Inner, typename... T>
struct wrap : public WrapperPolicy<Inner, T...> {
};

template <Policy Pol, Launch L, Pattern P>
struct make_policy_launch_pattern : public PolicyBaseT<Pol, L, P> {
};

template <Policy P>
struct make_policy
    : public PolicyBaseT<P, Launch::undefined, Pattern::undefined> {
};

template <Launch L>
struct make_launch
    : public PolicyBaseT<Policy::undefined, L, Pattern::undefined> {
};

template <Pattern P>
struct make_pattern
    : public PolicyBaseT<Policy::undefined, Launch::undefined, P> {
};

template <Policy Pol, Launch L>
struct make_policy_launch : public PolicyBaseT<Pol, L, Pattern::undefined> {
};

template <Policy Pol, Pattern P>
struct make_policy_pattern : public PolicyBaseT<Pol, Launch::undefined, P> {
};

template <Launch L, Pattern P>
struct make_launch_pattern : public PolicyBaseT<Policy::undefined, L, P> {
};

}  // end namespace RAJA

#endif /* RAJA_POLICYBASE_HXX */
