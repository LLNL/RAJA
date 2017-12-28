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
             public internal::Statement<ExecPolicy, EnclosedStmts...>{

  // TODO: add static_assert for valid policy in Pol
  const ExecPolicy execution_policy;

  RAJA_HOST_DEVICE constexpr For() : execution_policy{} {}
  RAJA_HOST_DEVICE constexpr For(const ExecPolicy &p) : execution_policy{p} {}
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



template <camp::idx_t ArgumentId, typename ExecPolicy, typename... EnclosedStmts>
struct StatementExecutor<For<ArgumentId, ExecPolicy, EnclosedStmts...>> {

  using ForType = For<ArgumentId, ExecPolicy, EnclosedStmts...>;

  template <typename WrappedBody>
  RAJA_INLINE
  void operator()(ForType const &fp, WrappedBody const &wrap)
  {
    using ::RAJA::policy::sequential::forall_impl;
    forall_impl(fp.execution_policy,
                camp::get<ArgumentId>(wrap.data.segment_tuple),
                ForWrapper<ArgumentId, WrappedBody>{wrap});
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

}  // namespace internal
}  // end namespace nested
}  // end namespace RAJA



#endif /* RAJA_pattern_nested_HPP */
