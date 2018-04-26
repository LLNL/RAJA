#ifndef RAJA_get_platform_HPP
#define RAJA_get_platform_HPP

#include "RAJA/internal/ForallNPolicy.hpp"
#include "RAJA/internal/LegacyCompatibility.hpp"

namespace RAJA
{

namespace detail 
{

struct max_platform {
  RAJA_HOST_DEVICE
  RAJA_INLINE
  constexpr RAJA::Platform operator()(const RAJA::Platform& l,
                                      const RAJA::Platform& r) const
  {
    return (l > r) ? l : r;
  }
};

/*!
 * Returns the platform for the specified execution policy.
 * This is a catch-all, so anything undefined gets Platform::undefined
 */
template <typename T, typename = void>
struct get_platform {
  // catch-all: undefined CHAI space
  static constexpr Platform value = Platform::undefined;
};


/*!
 * Takes a list of policies, extracts their platforms, and provides the
 * reduction of them all.
 */
template <typename... Policies>
struct get_platform_from_list {
  static constexpr Platform value =
      VarOps::foldl(max_platform(), get_platform<Policies>::value...);
};

/*!
 * Define an empty list as Platform::undefined;
 */
template <>
struct get_platform_from_list<> {
  static constexpr Platform value = Platform::undefined;
};


/*!
 * Specialization to define the platform for anything derived from PolicyBase,
 * which should catch all standard policies.
 *
 * (not for MultiPolicy or nested::Policy)
 */
template <typename T>
struct get_platform<T,
                    typename std::
                        enable_if<std::is_base_of<RAJA::PolicyBase, T>::value
                                  && !RAJA::type_traits::is_indexset_policy<T>::
                                         value>::type> {

  static constexpr Platform value = T::platform;
};


/*!
 * Specialization to define the platform for an IndexSet execution policy.
 *
 * Examines both segment iteration and segment execution policies.
 */
template <typename SEG, typename EXEC>
struct get_platform<RAJA::ExecPolicy<SEG, EXEC>>
    : public get_platform_from_list<SEG, EXEC> {
};


/*!
 * specialization for combining the execution polices for a forallN policy.
 *
 */
template <typename TAGS, typename... POLICIES>
struct get_platform<RAJA::NestedPolicy<RAJA::ExecList<POLICIES...>, TAGS>>
    : public get_platform_from_list<POLICIES...> {
};

} // closing brace for detail namespace
} // closing brace for RAJA namespace

#endif // RAJA_get_platform_HPP
