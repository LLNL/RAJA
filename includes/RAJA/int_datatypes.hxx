/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   Header file for RAJA integer type definitions.
 * 
 *          Definitions in this file will propagate to all RAJA header files.
 *
 * \author  Rich Hornung, Center for Applied Scientific Computing, LLNL
 * \author  Jeff Keasler, Applications, Simulations And Quality, LLNL 
 *
 ******************************************************************************
 */

#ifndef RAJA_int_datatypes_HXX
#define RAJA_int_datatypes_HXX

#include "config.hxx"

namespace RAJA {

///
/// Enum describing index set types.
///
enum SegmentType { _Range_, 
                   _RangeStride_, 
                   _Unstructured_, 
                   _Unknown_    // Keep last; used for default in case stmts
                 };

///
/// Enumeration used to indicate whether IndexSet objects own data
/// representing their indices.
///
enum IndexOwnership {
   Owned,
   Unowned
};

///
/// Type use for all loop indexing in RAJA constructs.
///
typedef int     Index_type;

}  // closing brace for RAJA namespace


#endif  // closing endif for header file include guard
