#ifndef RAJA_internal_ForallNPolicy_HXX_
#define RAJA_internal_ForallNPolicy_HXX_

namespace RAJA
{

/******************************************************************
 *  ForallN generic policies
 ******************************************************************/

template <typename P, typename I>
struct ForallN_PolicyPair : public I {
  typedef P POLICY;
  typedef I ISET;

  RAJA_INLINE
  explicit constexpr ForallN_PolicyPair(ISET const &i) : ISET(i) {}
};

template <typename... PLIST>
struct ExecList {
  constexpr const static size_t num_loops = sizeof...(PLIST);
  typedef std::tuple<PLIST...> tuple;
};

// Execute (Termination default)
struct ForallN_Execute_Tag {
};

struct Execute {
  typedef ForallN_Execute_Tag PolicyTag;
};

template <typename EXEC, typename NEXT = Execute>
struct NestedPolicy {
  typedef NEXT NextPolicy;
  typedef EXEC ExecPolicies;
};


template <typename... POLICY_REST>
struct ForallN_Executor {
};

/*!
 * \brief Functor that binds the first argument of a callable.
 *
 * This version has host-only constructor and host-device operator.
 */
template <typename BODY, typename INDEX_TYPE = Index_type>
struct ForallN_BindFirstArg_HostDevice {
  BODY const body;
  INDEX_TYPE const i;

  RAJA_INLINE
  constexpr ForallN_BindFirstArg_HostDevice(BODY b, INDEX_TYPE i0)
      : body(b), i(i0)
  {
  }

  RAJA_SUPPRESS_HD_WARN
  template <typename... ARGS>
  RAJA_INLINE RAJA_HOST_DEVICE void operator()(ARGS... args) const
  {
    body(i, args...);
  }
};

/*!
 * \brief Functor that binds the first argument of a callable.
 *
 * This version has host-only constructor and host-only operator.
 */
template <typename BODY, typename INDEX_TYPE = Index_type>
struct ForallN_BindFirstArg_Host {
  BODY const body;
  INDEX_TYPE const i;

  RAJA_INLINE
  constexpr ForallN_BindFirstArg_Host(BODY const &b, INDEX_TYPE i0)
      : body(b), i(i0)
  {
  }

  template <typename... ARGS>
  RAJA_INLINE void operator()(ARGS... args) const
  {
    body(i, args...);
  }
};

template <typename NextExec, typename BODY_in>
struct ForallN_PeelOuter {
  NextExec const next_exec;
  using BODY = typename std::remove_reference<BODY_in>::type;
  BODY const body;

  RAJA_INLINE
  constexpr ForallN_PeelOuter(NextExec const &ne, BODY const &b)
      : next_exec(ne), body(b)
  {
  }

  RAJA_INLINE
  RAJA_HOST_DEVICE
  void operator()(Index_type i) const
  {
    ForallN_BindFirstArg_HostDevice<BODY> inner(body, i);
    next_exec(inner);
  }

  RAJA_INLINE
  RAJA_HOST_DEVICE
  void operator()(Index_type i, Index_type j) const
  {
    ForallN_BindFirstArg_HostDevice<BODY> inner_i(body, i);
    ForallN_BindFirstArg_HostDevice<decltype(inner_i)> inner_j(inner_i, j);
    next_exec(inner_j);
  }

  RAJA_INLINE
  RAJA_HOST_DEVICE
  void operator()(Index_type i, Index_type j, Index_type k) const
  {
    ForallN_BindFirstArg_HostDevice<BODY> inner_i(body, i);
    ForallN_BindFirstArg_HostDevice<decltype(inner_i)> inner_j(inner_i, j);
    ForallN_BindFirstArg_HostDevice<decltype(inner_j)> inner_k(inner_j, k);
    next_exec(inner_k);
  }
};

}  // end of RAJA namespace

#endif
