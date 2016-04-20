/*!
 *
 * \file
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

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016, Lawrence Livermore National Security, LLC.
//
// Produced at the Lawrence Livermore National Laboratory
//
// LLNL-CODE-689114
//
// All rights reserved.
//
// This file is part of RAJA.
//
// For additional details, please also read raja/README-license.txt.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// * Redistributions of source code must retain the above copyright notice,
//   this list of conditions and the disclaimer below.
//
// * Redistributions in binary form must reproduce the above copyright notice,
//   this list of conditions and the disclaimer (as noted below) in the
//   documentation and/or other materials provided with the distribution.
//
// * Neither the name of the LLNS/LLNL nor the names of its contributors may
//   be used to endorse or promote products derived from this software without
//   specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL LAWRENCE LIVERMORE NATIONAL SECURITY,
// LLC, THE U.S. DEPARTMENT OF ENERGY OR CONTRIBUTORS BE LIABLE FOR ANY
// DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
// DAMAGES  (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS
// OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
// HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
// STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING
// IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
//
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//


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

  RAJA_INLINE ~ForallN_PolicyPair() {}

  RAJA_INLINE
  ForallN_PolicyPair(ISET const &i) : ISET(i) {}
};


template<typename ... PLIST>
struct ExecList{
  constexpr const static size_t num_loops = sizeof...(PLIST);
  typedef std::tuple<PLIST...> tuple;
};



// Execute (Termination default)
struct ForallN_Execute_Tag {};


struct Execute {
  typedef ForallN_Execute_Tag PolicyTag;
};

template<typename EXEC, typename NEXT=Execute>
struct NestedPolicy {
  typedef NEXT NextPolicy;
  typedef EXEC ExecPolicies;
};



/******************************************************************
 *  ForallN_Executor(): Default Executor for loops
 ******************************************************************/

template<typename BODY, typename INDEX_TYPE=Index_type>
struct ForallN_BindFirstArg_HostDevice {
  BODY const body;  
  INDEX_TYPE i;

  RAJA_INLINE
  ForallN_BindFirstArg_HostDevice(BODY b, INDEX_TYPE i0) : body(b), i(i0) {}

RAJA_SUPPRESS_HD_WARN
  template<typename ... ARGS>
  RAJA_INLINE
  RAJA_HOST_DEVICE
  void  operator()(ARGS ... args) const {
    body(i, args...);
  }
};

template<typename BODY, typename INDEX_TYPE=Index_type>
struct ForallN_BindFirstArg_Host {

  BODY const body;  
  INDEX_TYPE i;

  RAJA_INLINE
  ForallN_BindFirstArg_Host(BODY b, INDEX_TYPE i0) : body(b), i(i0) {}

  template<typename ... ARGS>
  RAJA_INLINE
  void  operator()(ARGS ... args) const {
    body(i, args...);
  }
};


template<typename NextExec, typename BODY>
struct ForallN_PeelOuter {

  NextExec const next_exec;
  BODY const body;

  RAJA_INLINE
  constexpr
  ForallN_PeelOuter(NextExec const &ne, BODY const &b) : next_exec(ne), body(b) {}

  RAJA_INLINE
  void operator()(Index_type i) const {
    ForallN_BindFirstArg_HostDevice<BODY> inner(body, i);
    next_exec(inner);
  }
  
  RAJA_INLINE
  void operator()(Index_type i, Index_type j) const {
    ForallN_BindFirstArg_HostDevice<BODY> inner_i(body, i);
    ForallN_BindFirstArg_HostDevice<decltype(inner_i)> inner_j(inner_i, j);
    next_exec(inner_j);
  }
  
  RAJA_INLINE
  void operator()(Index_type i, Index_type j, Index_type k) const {
    ForallN_BindFirstArg_HostDevice<BODY> inner_i(body, i);
    ForallN_BindFirstArg_HostDevice<decltype(inner_i)> inner_j(inner_i, j);
    ForallN_BindFirstArg_HostDevice<decltype(inner_j)> inner_k(inner_j, k);
    next_exec(inner_k);
  }
};

template<typename ... PREST>
struct ForallN_Executor {};

template<typename PI, typename ... PREST>
struct ForallN_Executor<PI, PREST...> {
  typedef typename PI::ISET TI;
  typedef typename PI::POLICY POLICY_I;

  typedef ForallN_Executor<PREST...> NextExec;

  PI const is_i;
  NextExec next_exec;

  template<typename ... TREST>
  constexpr
  ForallN_Executor(PI const &is_i0, TREST ... is_rest) : is_i(is_i0), next_exec(is_rest...) {}

  template<typename BODY>
  RAJA_INLINE
  void operator()(BODY body) const {
    ForallN_PeelOuter<NextExec, BODY> outer(next_exec, body);
    RAJA::forall<POLICY_I>(is_i, outer);
  }
};

template<>
struct ForallN_Executor<> {
  constexpr
  ForallN_Executor()  {}

  template<typename BODY>
  RAJA_INLINE void operator()(BODY body) const {
    body();
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

template<typename BODY, typename INDEX_TYPE=Index_type>
struct ForallN_BindFirstArg_Idx {
  BODY const body;  
  INDEX_TYPE i;

  RAJA_INLINE
  RAJA_HOST_DEVICE
  ForallN_BindFirstArg_Idx(BODY b, INDEX_TYPE i0) : body(b), i(i0) {}

  RAJA_SUPPRESS_HD_WARN
  template<typename ... ARGS>
  RAJA_INLINE
  RAJA_HOST_DEVICE
  void  operator()(ARGS ... args) const {
    body(i, args...);
  }
};


template<typename BODY, typename IdxI, typename ... IdxRest>
struct ForallN_IndexTypeConverter {


  RAJA_INLINE
  RAJA_HOST_DEVICE
  constexpr
  explicit ForallN_IndexTypeConverter(BODY const &b) : body(b) {}


  // call 'policy' layer with next policy
  RAJA_SUPPRESS_HD_WARN
  template<typename ... ARGS>
  RAJA_INLINE
  RAJA_HOST_DEVICE
  void operator()(Index_type i, ARGS ... args) const {
    // Bind the first argument
    ForallN_BindFirstArg_Idx<BODY, IdxI> bound(body, IdxI(i));

    // Peel a wrapper
    ForallN_IndexTypeConverter<decltype(bound), IdxRest...> inner(bound);
    inner(args...);
  }
  
  
  BODY body;

};

template<typename BODY, typename IdxI>
struct ForallN_IndexTypeConverter<BODY, IdxI> {

  RAJA_SUPPRESS_HD_WARN
  RAJA_INLINE
  RAJA_HOST_DEVICE
  explicit ForallN_IndexTypeConverter(BODY b) : body(b) {}

  // call 'policy' layer with next policy
  RAJA_SUPPRESS_HD_WARN
  RAJA_INLINE
  RAJA_HOST_DEVICE
  void operator()(Index_type i) const {
 
    body(IdxI(i));
  }

  // Copy of loop body
  BODY const body;
};




} // namespace RAJA
  
#endif

