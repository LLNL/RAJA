/*
 * Copyright (c) 2016, Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 *
 * All rights reserved.
 *
 * This source code cannot be distributed without permission and
 * further review from Lawrence Livermore National Laboratory.
 */

/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   Header file for RAJA integer type definitions.
 * 
 *          Definitions in this file will propagate to all RAJA header files.
 *
 ******************************************************************************
 */

#ifndef RAJA_int_datatypes_HXX
#define RAJA_int_datatypes_HXX

#include "../config.hxx"

namespace RAJA {

///
/// Enum describing index set types.
///
enum SegmentType { _RangeSeg_, 
                   _RangeStrideSeg_, 
                   _ListSeg_, 
                   _UnknownSeg_   // Keep last; used for default in case stmts
                 };

///
/// Enumeration used to indicate whether IndexSet objects own data
/// representing their indices.
///
enum IndexOwnership {
   Unowned,
   Owned
};

///
/// Type use for all loop indexing in RAJA constructs.
///
typedef int     Index_type;

///
/// Integer value for undefined indices and other integer values. 
/// Although this is a magic value, it avoids sprinkling them throughout code.
///
const int UndefinedValue = -9999999;

}  // closing brace for RAJA namespace


#endif  // closing endif for header file include guard
