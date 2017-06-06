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

enum class Platform {
  undefined = 0,
  host = 1,
  cuda = 2,
  omp_target = 4
};

struct PolicyBase {
};

template <Policy P = Policy::undefined,
          Launch L = Launch::undefined,
          Pattern Pat = Pattern::undefined,
          Platform Plat = Platform::host
          >
struct PolicyBaseT : public PolicyBase {
  static constexpr Policy policy = P;
  static constexpr Launch launch = L;
  static constexpr Pattern pattern = Pat;
  static constexpr Platform platform = Plat;
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

template <Policy Pol, Launch L, Pattern P, Platform Plat>
struct make_policy_launch_pattern_platform : public PolicyBaseT<Pol, L, P, Plat> {
};

struct make_undefined
    : public PolicyBaseT<Policy::undefined, Launch::undefined, Pattern::undefined> {
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

template <Policy Pol, Pattern P, Platform Plat>
struct make_policy_pattern_platform : public PolicyBaseT<Pol, Launch::undefined, P, Plat>
{
};

}  // end namespace RAJA

#endif /* RAJA_POLICYBASE_HXX */
