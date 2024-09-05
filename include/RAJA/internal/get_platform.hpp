#ifndef RAJA_get_platform_HPP
#define RAJA_get_platform_HPP

#include "RAJA/util/Operators.hpp"
#include "RAJA/internal/foldl.hpp"
#include "RAJA/pattern/kernel/internal.hpp"

namespace RAJA
{

namespace policy
{
namespace multi
{
template <typename Selector, typename... Policies>
class MultiPolicy;

}
} // namespace policy

namespace detail
{

struct max_platform
{
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
struct get_platform
{
  // catch-all: undefined platform
  static constexpr Platform value = Platform::undefined;
};


/*!
 * Takes a list of policies, extracts their platforms, and provides the
 * reduction of them all.
 */
template <typename... Policies>
struct get_platform_from_list
{
  static constexpr Platform value =
      foldl(max_platform(), get_platform<Policies>::value...);
};

/*!
 * Define an empty list as Platform::undefined;
 */
template <>
struct get_platform_from_list<>
{
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
                    typename std::enable_if<
                        std::is_base_of<RAJA::PolicyBase, T>::value &&
                        !RAJA::type_traits::is_indexset_policy<T>::value>::type>
{

  static constexpr Platform value = T::platform;
};


/*!
 * Specialization to define the platform for an IndexSet execution policy.
 *
 * Examines both segment iteration and segment execution policies.
 */
template <typename SEG, typename EXEC>
struct get_platform<RAJA::ExecPolicy<SEG, EXEC>>
    : public get_platform_from_list<SEG, EXEC>
{};


template <typename T>
struct get_statement_platform
{
  static constexpr Platform value =
      get_platform_from_list<typename T::execution_policy_t,
                             typename T::enclosed_statements_t>::value;
};

/*!
 * Specialization to define the platform for an kernel::StatementList, and
 * (by alias) a kernel::Policy
 *
 * This collects the Platform from each of it's statements, recursing into
 * each of them.
 */
template <typename... Stmts>
struct get_platform<RAJA::internal::StatementList<Stmts...>>
{
  static constexpr Platform value =
      foldl(max_platform(), get_statement_platform<Stmts>::value...);
};

/*!
 * Specialize for an empty statement list to be undefined
 */
template <>
struct get_platform<RAJA::internal::StatementList<>>
{
  static constexpr Platform value = Platform::undefined;
};


// Top level MultiPolicy shouldn't select a platform
// Once a specific policy is selected, that policy will select the correct
// platform... see policy_invoker in MultiPolicy.hpp
template <typename SELECTOR, typename... POLICIES>
struct get_platform<RAJA::policy::multi::MultiPolicy<SELECTOR, POLICIES...>>
{
  static constexpr Platform value = Platform::undefined;
};

} // namespace detail
} // namespace RAJA

#endif // RAJA_get_platform_HPP
