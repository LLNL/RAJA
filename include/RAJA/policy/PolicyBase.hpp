#ifndef RAJA_POLICYBASE_HXX
#define RAJA_POLICYBASE_HXX

namespace RAJA
{

namespace detail
{
struct wrap_policy {
};
}

struct PolicyBase {
};

struct reduce_policy : public PolicyBase {
};

struct forall_policy : public PolicyBase {
};

template <typename Inner>
struct wrap_policy : public PolicyBase, public detail::wrap_policy {
  using inner_policy = Inner;
};

struct taskgraph_policy : public PolicyBase {
};
}

#endif /* RAJA_POLICYBASE_HXX */
