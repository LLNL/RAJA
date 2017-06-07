#ifndef RAJA_DETAIL_RAJA_CHAI_HPP
#define RAJA_DETAIL_RAJA_CHAI_HPP

#include "chai/ExecutionSpaces.hpp"

#include "RAJA/policy/PolicyBase.hpp"

#include "RAJA/index/IndexSet.hpp"
#include "RAJA/internal/LegacyCompatibility.hpp"
#include "RAJA/internal/ForallNPolicy.hpp"

#include "RAJA/internal/type_traits.hpp"

namespace RAJA {
namespace detail {

struct max_platform {
    constexpr
    RAJA::Platform
    operator()(const RAJA::Platform& l, const RAJA::Platform& r) const
    {
      return (l == RAJA::Platform::cuda) ? l : r;
    }
};

template<Platform p>
struct get_space_impl {};

template<>
struct get_space_impl<Platform::host> {
  static constexpr chai::ExecutionSpace value = chai::CPU;
};

#if defined(RAJA_ENABLE_CUDA)
template<>
struct get_space_impl<Platform::cuda> {
  static constexpr chai::ExecutionSpace value = chai::GPU;
};
#endif

template<>
struct get_space_impl<Platform::undefined> {
  static constexpr chai::ExecutionSpace value = chai::NONE;
};


template<typename... Ts>
struct get_space_from_list {
  static constexpr chai::ExecutionSpace value = 
    get_space_impl<VarOps::foldl(max_platform(), Ts::platform...)>::value;
};

template <typename T, typename=void> struct get_space {};

template <typename T>
struct get_space<T, typename std::enable_if<std::is_base_of<PolicyBase, T>::value>::type>
    : public get_space_impl<T::platform> {};

template <typename SEG, typename EXEC>
struct get_space<RAJA::IndexSet::ExecPolicy<SEG, EXEC> > : public get_space<EXEC> {};

template<typename SEG, typename EXEC>
struct get_space_from_list<RAJA::IndexSet::ExecPolicy<SEG, EXEC>> {
  static constexpr chai::ExecutionSpace value = get_space<EXEC>::value;
};

template <typename TAGS, typename... POLICIES>
struct get_space<RAJA::NestedPolicy< RAJA::ExecList<POLICIES...>, TAGS > > : 
  public get_space_from_list<POLICIES...> {};

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
