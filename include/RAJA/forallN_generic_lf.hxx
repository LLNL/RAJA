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

template<typename P, typename I, typename IDX>
struct ForallN_PolicyPair : public I {
  
  typedef P POLICY;
  typedef I ISET;
  typedef IDX INDEX;

  RAJA_INLINE
  explicit
  constexpr
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

/*!
 * \brief Functor that binds the first argument of a callable.
 * 
 * This version has host-only constructor and host-device operator.
 */
template<typename BODY, typename INDEX_TYPE=Index_type>
struct ForallN_BindFirstArg_HostDevice {
  BODY const body;  
  INDEX_TYPE const i;

  RAJA_INLINE
  constexpr
  ForallN_BindFirstArg_HostDevice(BODY b, INDEX_TYPE i0) : body(b), i(i0) {}

  RAJA_SUPPRESS_HD_WARN
  template<typename ... ARGS>
  RAJA_INLINE
  RAJA_HOST_DEVICE
  void  operator()(ARGS ... args) const {
    body(i, args...);
  }
};


/*!
 * \brief Functor that binds the first argument of a callable.
 * 
 * This version has host-only constructor and host-only operator.
 */
template<typename BODY, typename INDEX_TYPE=Index_type>
struct ForallN_BindFirstArg_Host {

  BODY const body;  
  INDEX_TYPE const i;

  RAJA_INLINE
  constexpr
  ForallN_BindFirstArg_Host(BODY const &b, INDEX_TYPE i0) : body(b), i(i0) {}

  template<typename ... ARGS>
  RAJA_INLINE
  void  operator()(ARGS ... args) const {
    body(i, args...);
  }
};


template<typename ... PREST>
struct ForallN_Executor {};


/*!
 * \brief Primary policy execution that peels off loop nests.
 *
 *  The default action is to call RAJA::forall to peel off outer loop nest.
 */

template<typename PI, typename ... PREST>
struct ForallN_Executor<PI, PREST...> {
  

  template<typename BODY>
  RAJA_INLINE
  void operator()(BODY const &body, PI const &pi, PREST const &... prest) const {
  
    using POLICY_I = typename PI::POLICY;
    using INDEX_I  = typename PI::INDEX;
    
    ForallN_Executor<PREST...> next_exec;
    
    RAJA::forall<POLICY_I>(pi,
      [=](int i){
        RAJA::ForallN_BindFirstArg_HostDevice<BODY, INDEX_I> bound(body, INDEX_I(i));
        next_exec(bound, prest...);
      }
    );
  }
};


/*!
 * \brief Execution termination case
 */
template<>
struct ForallN_Executor<> {

  template<typename BODY>
  RAJA_INLINE void operator()(BODY const &body) const {
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
void forallN_policy(ForallN_Execute_Tag, BODY const &body, ARGS const &... args){

  // Create executor object to launch loops
  ForallN_Executor<ARGS...> exec;

  // Launch loop body
  exec(body, args...);
}





} // namespace RAJA
  
#endif

