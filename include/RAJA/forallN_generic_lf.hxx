/*
 * Copyright (c) 2016, Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 *
 * All rights reserved.
 *
 * This source code cannot be distributed without permission and
 * further review from Lawrence Livermore National Laboratory.
 */


/**
 *
 * This is the non-code-generated for Lambda-Free (lf) forallN implementation.
 *
 * We bent over backwards to avoid using lambdas anywhere in this code,
 * instead using functors and other struct objects, in order to avoid the
 * CUDA host/device labmda issue.
 *
 */


#ifndef RAJA_forallN_generic_lf_HXX__
#define RAJA_forallN_generic_lf_HXX__

#include"config.hxx"
#include"int_datatypes.hxx"
#include<tuple>

namespace RAJA {



/******************************************************************
 *  ForallN generic policies
 ******************************************************************/

template<typename P, typename I>
struct ForallN_PolicyPair : public I {
  
  typedef P POLICY;
  typedef I ISET;

  RAJA_INLINE
  ForallN_PolicyPair(ISET const &i) : ISET(i) {}  
  
  RAJA_INLINE
  ISET const &operator*() const { return static_cast<ISET const &>(*this); }
};


template<typename ... PLIST>
struct ExecList{
  constexpr const static size_t num_loops = sizeof...(PLIST);
  typedef std::tuple<PLIST...> tuple;
};



// Execute (Termination default)
struct ForallN_Execute_Tag {};


struct ForallN_Execute {
  typedef ForallN_Execute_Tag PolicyTag;
};

template<typename EXEC, typename NEXT=ForallN_Execute>
struct ForallN_Policy {
  typedef NEXT NextPolicy;
  typedef EXEC ExecPolicies;
};



/******************************************************************
 *  ForallN_Executor(): Default Executor for loops
 ******************************************************************/


template<typename BODY, typename INDEX_TYPE=Index_type>
struct ForallN_BindFirstArg {

  BODY const body;
  INDEX_TYPE const i;

  RAJA_INLINE
  RAJA_HOST_DEVICE
  ForallN_BindFirstArg(BODY const &b, INDEX_TYPE i0) : body(b), i(i0) {}

  template<typename ... ARGS>
  RAJA_INLINE
  RAJA_HOST_DEVICE
  void operator()(ARGS ... args) const {
    body(i, args...);
  }
};

template<typename NextExec, typename BODY>
struct ForallN_PeelOuter {

  NextExec const next_exec;
  BODY const body;

  RAJA_INLINE
  RAJA_HOST_DEVICE
  ForallN_PeelOuter(NextExec const &ne, BODY const &b) : next_exec(ne), body(b) {}

  RAJA_INLINE
  RAJA_HOST_DEVICE
  void operator()(Index_type i) const {
    ForallN_BindFirstArg<BODY> inner(body, i);
    next_exec(inner);
  }
};




template<typename PI>
struct Forall1Executor {
  typedef typename PI::ISET TI;
  typedef typename PI::POLICY POLICY_I;

  TI const is_i;

  explicit Forall1Executor(TI const &is_i0) : is_i(is_i0) {}

  template<typename BODY>
  inline void RAJA_HOST_DEVICE operator()(BODY body) const {
    RAJA::forall<POLICY_I>(is_i, body);
  }
};

template<typename PI, typename ... PREST>
struct ForallN_Executor {
  typedef typename PI::ISET TI;
  typedef typename PI::POLICY POLICY_I;

  typedef ForallN_Executor<PREST...> NextExec;

  TI const is_i;
  NextExec next_exec;

  template<typename ... TREST>
  ForallN_Executor(TI const &is_i0, TREST ... is_rest) : is_i(is_i0), next_exec(is_rest...) {}

  template<typename BODY>
  inline void operator()(BODY body) const {
    ForallN_PeelOuter<NextExec, BODY> outer(next_exec, body);
    RAJA::forall<POLICY_I>(is_i, outer);
  }
};


template<typename PI>
struct ForallN_Executor<PI> {
  typedef typename PI::ISET TI;
  typedef typename PI::POLICY POLICY_I;

  TI const &is_i;

  explicit ForallN_Executor(TI const &is_i0) : is_i(is_i0) {}

  template<typename BODY>
  inline void RAJA_HOST_DEVICE operator()(BODY body) const {
    RAJA::forall<POLICY_I>(is_i, body);
  }
};



/******************************************************************
 *  forallN_policy(), base execution policies
 ******************************************************************/


/*!
 * \brief Execute inner loops policy function.
 *
 * This is the default termination case.
 */
 
template<typename POLICY, typename BODY, typename ... ARGS>
RAJA_INLINE
void forallN_policy(ForallN_Execute_Tag, BODY body, ARGS ... args){

  // Create executor object to launch loops
  ForallN_Executor<ARGS...> exec(args...);

  // Launch loop body
  exec(body);
}



/******************************************************************
 *  Index type conversion, wraps lambda given by user with an outer
 *  callable object where all variables are Index_type
 ******************************************************************/



template<typename BODY, typename IdxI, typename ... IdxRest>
struct ForallN_IndexTypeConverter {

  explicit ForallN_IndexTypeConverter(BODY const &b) : body(b) {}

  // call 'policy' layer with next policy
  template<typename ... ARGS>
  inline void RAJA_HOST_DEVICE operator()(Index_type i, ARGS ... args) const {
    // Bind the first argument
    ForallN_BindFirstArg<BODY, IdxI> bound(body, IdxI(i));

    // Peel a wrapper
    ForallN_IndexTypeConverter<ForallN_BindFirstArg<BODY, IdxI>, IdxRest...> inner(bound);
    inner(args...);
  }

  // Copy of loop body
  BODY const &body;
};

template<typename BODY, typename IdxI>
struct ForallN_IndexTypeConverter<BODY, IdxI> {

  explicit ForallN_IndexTypeConverter(BODY const &b) : body(b) {}

  // call 'policy' layer with next policy
  inline void RAJA_HOST_DEVICE operator()(Index_type i) const {
    body(IdxI(i));
  }

  // Copy of loop body
  BODY const &body;
};




} // namespace RAJA
  
#endif

