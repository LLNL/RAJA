/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   Header file providing RAJA reduction declarations.
 * 
 * \author  Rich Hornung, Center for Applied Scientific Computing, LLNL
 * \author  Jeff Keasler, Applications, Simulations And Quality, LLNL 
 *
 ******************************************************************************
 */

#ifndef RAJA_reducers_HXX
#define RAJA_reducers_HXX

#include "int_datatypes.hxx"

#define RAJA_MAX_REDUCE_VARS (8)

namespace RAJA {

#if 0 // RDH Will we ever need something like this?
///
/// Enum defining valid reduction type.
///
enum ReductionType { 
                     _SUM_,
                     _MIN_,
                     _MINLOC_,
                     _MAX_,
                     _MAXLOC_,
                     _INACTIVE_
                   };
#endif

///
/// Macros for type agnostic reduction operations.
///
#define RAJA_MIN(a, b) (((b) < (a)) ? (b) : (a))
#define RAJA_MAX(a, b) (((b) > (a)) ? (b) : (a))


///
/// Forward declarations for reduction templates.
///

template<typename REDUCE_POLICY_T,
         typename T>
class ReduceMin;

template<typename REDUCE_POLICY_T,
         typename T>
class ReduceMinLoc;


template<typename REDUCE_POLICY_T,
         typename T>
class ReduceMax;

template<typename REDUCE_POLICY_T,
         typename T>
class ReduceMaxLoc;


template<typename REDUCE_POLICY_T,
         typename T>
class ReduceSum;


}  // closing brace for RAJA namespace


#endif  // closing endif for header file include guard
