#ifndef RAJA_pattern_nested_For_HPP
#define RAJA_pattern_nested_For_HPP


#include "RAJA/config.hpp"

#include <iostream>
#include <type_traits>

namespace RAJA
{

namespace nested
{


/*!
 * A nested::forall statement that implements a single loop.
 *
 *
 */
template <camp::idx_t ArgumentId, typename ExecPolicy = camp::nil, typename... EnclosedStmts>
struct For : public internal::ForList,
             public internal::ForTraitBase<ArgumentId, ExecPolicy>,
             public internal::Statement<EnclosedStmts...>{
  using as_for_list = camp::list<For>;

  // used for execution space resolution
  using as_space_list = camp::list<For>;


  // TODO: add static_assert for valid policy in Pol
  const ExecPolicy exec_policy;

  RAJA_HOST_DEVICE constexpr For() : exec_policy{} {}
  RAJA_HOST_DEVICE constexpr For(const ExecPolicy &p) : exec_policy{p} {}
};





namespace internal{


template <camp::idx_t ArgumentId, typename BaseWrapper>
struct ForWrapper : GenericWrapper<ArgumentId, BaseWrapper> {
  using Base = GenericWrapper<ArgumentId, BaseWrapper>;
  using Base::Base;

  template <typename InIndexType>
  void operator()(InIndexType i)
  {
    Base::wrapper.data.template assign_index<ArgumentId>(i);
    Base::wrapper();
  }
};



template <typename ForType>
struct StatementExecutor {
  static_assert(std::is_base_of<internal::ForBase, ForType>::value,
                "Only For-based policies should get here");
  template <typename WrappedBody>
  RAJA_INLINE
  void operator()(ForType const &fp, WrappedBody const &wrap)
  {
    using ::RAJA::policy::sequential::forall_impl;
    forall_impl(fp.exec_policy,
                camp::get<ForType::index_val>(wrap.data.segment_tuple),
                ForWrapper<ForType::index_val, WrappedBody>{wrap});
  }
};




/**
 * @brief specialization of internal::thread_privatize for nested
 */
template <camp::idx_t Index, typename BW>
auto thread_privatize(const nested::internal::ForWrapper<Index, BW> &item)
    -> NestedPrivatizer<nested::internal::ForWrapper<Index, BW>>
{
  return NestedPrivatizer<nested::internal::ForWrapper<Index, BW>>{item};
}

} // namespace internal
}  // end namespace nested
}  // end namespace RAJA



#endif /* RAJA_pattern_nested_HPP */
