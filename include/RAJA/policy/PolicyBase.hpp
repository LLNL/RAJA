#ifndef RAJA_POLICYBASE_HXX
#define RAJA_POLICYBASE_HXX

namespace RAJA
{

struct PolicyBase {
};

struct reduce_policy : public PolicyBase {
};

struct forall_policy : public PolicyBase {
};

struct taskgraph_policy : public PolicyBase {
};

namespace detail
{
  struct wrap_policy : public PolicyBase {
};
} // end namespace detail

template <typename Inner>
struct wrap_policy : public detail::wrap_policy {
  using inner_policy = Inner;
};

} // end namespace RAJA

#endif /* RAJA_POLICYBASE_HXX */
