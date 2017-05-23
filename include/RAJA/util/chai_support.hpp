#ifndef RAJA_DETAIL_RAJA_CHAI_HPP
#define RAJA_DETAIL_RAJA_CHAI_HPP

#include "chai/ExecutionSpaces.hpp"

#include "RAJA/policy/PolicyBase.hpp"

#include "RAJA/index/IndexSet.hpp"
#include "RAJA/internal/LegacyCompatibility.hpp"
#include "RAJA/internal/ForallNPolicy.hpp"

#include "RAJA/internal/type_traits.hpp"

namespace RAJA {

template <bool cuda, typename POL, typename... REST>
struct has_cuda : has_cuda< is_cuda_policy<POL>::type, POL, REST...> { };

template <typename POL, typename... REST>
struct has_cuda<true, POL, REST...>
{
    static bool const value = true;
};

template <typename POL, typename... REST>
struct has_cuda<false, POL, REST...>
{
    static bool const value = has_cuda<false, REST...>::value;
};

template <typename POL>
struct has_cuda<true, POL>
{
    static bool const value = true;
};

template <typename POL>
struct has_cuda<false, POL>
{
    static bool const value = false;
};



template <typename Selector, typename... Policies>
class MultiPolicy;

namespace detail {

template<bool gpu>
struct get_space_impl {};

template<>
struct get_space_impl<false> {
  static constexpr chai::ExecutionSpace value = chai::CPU;
};

template<>
struct get_space_impl<true> {
  static constexpr chai::ExecutionSpace value = chai::GPU;
};


template <typename T, typename=void> struct get_space {};

template <typename T>
struct get_space<T, typename std::enable_if<std::is_base_of<PolicyBase, T>::value>::type>
    : public get_space_impl<is_cuda_policy<T>::value > {};

template <typename SEG, typename EXEC>
struct get_space<RAJA::IndexSet::ExecPolicy<SEG, EXEC> > : public get_space<EXEC> {};

template <typename Selector, typename... Policies>
struct get_space<RAJA::MultiPolicy<Selector, Policies...> > : public get_space_impl<false> {};

template <typename... POLICIES>
struct get_space<RAJA::NestedPolicy< RAJA::ExecList<POLICIES...> > > : 
  public get_space_impl< has_cuda<false, POLICIES...>::value > {};

// constexpr chai::ExecutionSpace getSpace(RAJA::PolicyBase&& policy) {
//   // return is_cuda_policy<decltype(policy)>::value ? chai::GPU : chai::CPU;
// }

// template <typename SEG, typename EXEC>
// constexpr chai::ExecutionSpace getSpace(RAJA::IndexSet::ExecPolicy<SEG, EXEC>&& policy) {
//   return getSpace(EXEC());
// }
// 
// template <typename Selector, typename... Policies>
// constexpr chai::ExecutionSpace getSpace(RAJA::MultiPolicy<Selector, Policies...> && policy) {
//   return chai::NONE;
// }
// 
// template <typename... Policies>
// constexpr chai::ExecutionSpace getSpace(RAJA::NestedPolicy<Policies...>&& policy) {
//     return chai::CPU;
// }


}
}

#endif
