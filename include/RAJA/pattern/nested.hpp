#ifndef RAJA_pattern_nested_HPP
#define RAJA_pattern_nested_HPP

#include "RAJA/RAJA.hpp"
#include "RAJA/config.hpp"
#include "RAJA/internal/tuple.hpp"
#include "RAJA/util/defines.hpp"
#include "RAJA/util/types.hpp"

#include "RAJA/external/metal.hpp"

#include <iostream>
#include <type_traits>

namespace RAJA
{
namespace nested
{

namespace internal
{


template <size_t depth, typename Pol, typename SegmentTuple, typename Body>
RAJA_INLINE void forall(const Pol& p, const SegmentTuple& st, const Body& b)
{
}

// Universal base of all For wrappers for type traits
struct ForBase {
};

using is_for_policy = metal::
    bind<metal::trait<std::is_base_of>, metal::always<ForBase>, metal::_1>;
}
template <typename Seq>
using get_for_policies = metal::remove_if<
    Seq,
    metal::bind<metal::lambda<metal::not_>, internal::is_for_policy>>;

template <typename Pol,
          metal::int_ ArgumentId,
          typename IndexType = RAJA::Index_type>
struct For : public internal::ForBase {
  const Pol& pol;
  For() : pol{} {}
  For(const Pol& p) : pol{p} {}
};

template <typename... Policies>
using Policy = RAJA::util::tuple<Policies...>;

template <typename Pol, typename SegmentTuple, typename Body>
RAJA_INLINE void forall(const Pol& p, const SegmentTuple& st, const Body& b)
{
#ifdef RAJA_ENABLE_CUDA
  // this call should be moved into a cuda file
  // but must be made before loop_body is copied
  beforeCudaKernelLaunch();
#endif
  using fors = get_for_policies<typename Pol::TList>;
  static_assert(RAJA::util::tuple_size<SegmentTuple>::value
                    == metal::size<fors>::value,
                "policy and segment index counts do not match");
  std::cout << typeid(fors).name() << std::endl;


// internal::forall<0>(p, st, b);

#ifdef RAJA_ENABLE_CUDA
  afterCudaKernelLaunch();
#endif
}

}  // end namespace nested
}  // end namespace RAJA

#endif /* RAJA_pattern_nested_HPP */
