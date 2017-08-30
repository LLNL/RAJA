/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   RAJA header file defining automatic and builtin atomic operations.
 *
 ******************************************************************************
 */

#ifndef RAJA_policy_atomic_builtin_HPP
#define RAJA_policy_atomic_builtin_HPP

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
// For additional details, please also read RAJA/LICENSE.
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

#include "RAJA/config.hpp"
#include "RAJA/util/defines.hpp"
#include "RAJA/util/TypeConvert.hpp"


namespace RAJA
{

//! Atomic policy that uses the compilers builtin __sync_XXX or __atomic_XXX routines
struct builtin_atomic{};



#ifdef RAJA_COMPILER_MSVC

template<typename T>
RAJA_INLINE
T atomicCAS(RAJA::builtin_atomic, T volatile *acc, T compare, T value);

template<>
RAJA_INLINE
unsigned atomicCAS(RAJA::builtin_atomic, unsigned volatile *acc, unsigned compare, unsigned value){

  long long_value =  RAJA::util::reinterp_A_as_B<unsigned, long>(value);
  long long_compare =  RAJA::util::reinterp_A_as_B<unsigned, long>(compare);

  long old = _InterlockedCompareExchange((long *)acc, long_value, long_compare);

  return RAJA::util::reinterp_A_as_B<long, unsigned>(old);
}


template<>
RAJA_INLINE
unsigned long long atomicCAS(RAJA::builtin_atomic, unsigned long long volatile *acc, unsigned long long compare, unsigned long long value){

  long long long_value =  RAJA::util::reinterp_A_as_B<unsigned long long, long long>(value);
  long long long_compare =  RAJA::util::reinterp_A_as_B<unsigned long long, long long>(compare);

  long long old = _InterlockedCompareExchange((long long *)acc, long_value, long_compare);

  return RAJA::util::reinterp_A_as_B<long long, unsigned long long>(old);
}

#else // RAJA_COMPILER_MSVC
template<typename T>
RAJA_INLINE
T atomicCAS(RAJA::builtin_atomic, T volatile *acc, T compare, T value){  
  __atomic_compare_exchange_n(acc, &compare, value, false, __ATOMIC_RELAXED, __ATOMIC_RELAXED);
  return compare;
}
#endif // RAJA_COMPILER_MSVC

namespace detail {

template<size_t BYTES>
struct BuiltinAtomicCAS {};


template<>
struct BuiltinAtomicCAS<4> {
    
  /*!
   * Generic impementation of any atomic 32-bit operator.
   * Implementation uses the existing builtin unsigned 32-bit CAS operator.
   * Returns the OLD value that was replaced by the result of this operation.
   */
  template<typename T, typename OPER>
  RAJA_INLINE
  T operator()(T volatile *acc, OPER const &oper) const {
    unsigned oldval, newval, readback;
    
    oldval = RAJA::util::reinterp_A_as_B<T, unsigned>(*acc);
    newval = RAJA::util::reinterp_A_as_B<T, unsigned>( 
        oper( RAJA::util::reinterp_A_as_B<unsigned, T>(oldval) ) );
            
    while ((readback = RAJA::atomicCAS(RAJA::builtin_atomic{}, (unsigned *)acc, oldval, newval))
           != oldval) {
      oldval = readback;
      newval = RAJA::util::reinterp_A_as_B<T, unsigned>( 
        oper( RAJA::util::reinterp_A_as_B<unsigned, T>(oldval) ) );
    }
    return RAJA::util::reinterp_A_as_B<unsigned, T>(oldval);
  }
  
};

template<>
struct BuiltinAtomicCAS<8> {

  /*!
   * Generic impementation of any atomic 64-bit operator.
   * Implementation uses the existing builtin unsigned 64-bit CAS operator.
   * Returns the OLD value that was replaced by the result of this operation.
   */  
  template<typename T, typename OPER>
  RAJA_INLINE
  T operator()(T volatile *acc, OPER const &oper) const {
    unsigned long long oldval, newval, readback;
    
    oldval = RAJA::util::reinterp_A_as_B<T, unsigned long long>(*acc);
    newval = RAJA::util::reinterp_A_as_B<T, unsigned long long>(
      oper( RAJA::util::reinterp_A_as_B<unsigned long long, T>(oldval) ) );
      
    while ((readback = RAJA::atomicCAS(RAJA::builtin_atomic{}, (unsigned long long *)acc, oldval, newval))
           != oldval) {
      oldval = readback;
      newval = RAJA::util::reinterp_A_as_B<T, unsigned long long>(
      oper( RAJA::util::reinterp_A_as_B<unsigned long long, T>(oldval) ) );
    }
    return RAJA::util::reinterp_A_as_B<unsigned long long, T>(oldval);
  }
  
};


/*!
 * Generic impementation of any atomic 32-bit or 64-bit operator that can be
 * implemented using a compare and swap primitive.
 * Implementation uses the builtin unsigned 32-bit and 64-bit CAS operators.
 * Returns the OLD value that was replaced by the result of this operation.
 */
template<typename T, typename OPER>
RAJA_INLINE
T builtin_atomic_CAS_oper(T volatile *acc, OPER && oper){
  BuiltinAtomicCAS<sizeof(T)> cas;
  return cas(acc, std::forward<OPER>(oper));
}



} // namespace detail



template<typename T>
RAJA_INLINE
T atomicAdd(RAJA::builtin_atomic, T volatile *acc, T value){
  return detail::builtin_atomic_CAS_oper(acc, [=](T a){return a+value;});
}



template<typename T>
RAJA_INLINE
T atomicSub(RAJA::builtin_atomic, T volatile *acc, T value){
  return detail::builtin_atomic_CAS_oper(acc, [=](T a){return a-value;});
}

template<typename T>
RAJA_INLINE
T atomicMin(RAJA::builtin_atomic, T volatile *acc, T value){
  return detail::builtin_atomic_CAS_oper(acc, [=](T a){return a<value ? a : value;});
}

template<typename T>
RAJA_INLINE
T atomicMax(RAJA::builtin_atomic, T volatile *acc, T value){
  return detail::builtin_atomic_CAS_oper(acc, [=](T a){return a>value ? a : value;});
}

template<typename T>
RAJA_INLINE
T atomicInc(RAJA::builtin_atomic, T volatile *acc){
  return detail::builtin_atomic_CAS_oper(acc, [=](T a){return a+1;});
}

template<typename T>
RAJA_INLINE
T atomicInc(RAJA::builtin_atomic, T volatile *acc, T val){
  return detail::builtin_atomic_CAS_oper(acc, [=](T old){return ((old >= val) ? 0 : (old+1));});
}

template<typename T>
RAJA_INLINE
T atomicDec(RAJA::builtin_atomic, T volatile *acc){
  return detail::builtin_atomic_CAS_oper(acc, [=](T a){return a-1;});
}

template<typename T>
RAJA_INLINE
T atomicDec(RAJA::builtin_atomic, T volatile *acc, T val){
  return detail::builtin_atomic_CAS_oper(acc, [=](T old){return (((old==0)|(old>val))?val:(old-1));});
}

template<typename T>
RAJA_INLINE
T atomicAnd(RAJA::builtin_atomic, T volatile *acc, T value){
  return detail::builtin_atomic_CAS_oper(acc, [=](T a){return a&value;});
}

template<typename T>
RAJA_INLINE
T atomicOr(RAJA::builtin_atomic, T volatile *acc, T value){
  return detail::builtin_atomic_CAS_oper(acc, [=](T a){return a|value;});
}

template<typename T>
RAJA_INLINE
T atomicXor(RAJA::builtin_atomic, T volatile *acc, T value){
  return detail::builtin_atomic_CAS_oper(acc, [=](T a){return a^value;});
}

template<typename T>
RAJA_INLINE
T atomicExchange(RAJA::builtin_atomic, T volatile *acc, T value){
  return detail::builtin_atomic_CAS_oper(acc, [=](T){return value;});
}









}  // namespace RAJA

// make sure this define doesn't bleed out of this header
#undef RAJA_AUTO_ATOMIC

#endif
