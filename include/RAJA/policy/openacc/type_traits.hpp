#ifndef RAJA_openacc_type_traits_HPP_
#define RAJA_openacc_type_traits_HPP_

#include <utility>

namespace RAJA
{
namespace acc
{

template <typename T>
struct Not {
  static constexpr bool value = !T::value;
};

#define MEMBER_HAS(NAME)                                                     \
  template <class T>                                                         \
  class NAME                                                                 \
  {                                                                          \
    template <                                                               \
        typename U,                                                          \
        typename = typename std::enable_if<!std::is_member_pointer<decltype( \
            &U::NAME)>::value>::type>                                        \
    static std::true_type check(int);                                        \
    template <typename>                                                      \
    static std::false_type check(...);                                       \
                                                                             \
  public:                                                                    \
    static constexpr bool value = decltype(check<T>(0))::value;              \
  }

#define MEMBER_HAS_NO(NAME) \
  template <typename T>     \
  using no_##NAME = Not<NAME<T>>;

MEMBER_HAS(ngangs);
MEMBER_HAS(nworkers);
MEMBER_HAS(nvectors);
MEMBER_HAS(independent);
MEMBER_HAS(gang);
MEMBER_HAS(worker);
MEMBER_HAS(vector);
MEMBER_HAS_NO(ngangs);
MEMBER_HAS_NO(nworkers);
MEMBER_HAS_NO(nvectors);
MEMBER_HAS_NO(independent);
MEMBER_HAS_NO(gang);
MEMBER_HAS_NO(worker);
MEMBER_HAS_NO(vector);

#undef MEMBER_HAS
#undef MEMBER_HAS_NO

namespace ___hidden
{

template <bool...>
struct blist;

template <bool... B>
struct all_of : std::is_same<blist<true, B...>, blist<B..., true>> {
};

template <typename Config, template <typename> class... Type>
struct Check : all_of<Type<Config>::value...> {
};
}

template <typename Conf, template <typename> class... TArgs>
using When =
    typename std::enable_if<___hidden::Check<Conf, TArgs...>::value>::type;

}  // end namespace acc

}  // end namespace RAJA

#endif
