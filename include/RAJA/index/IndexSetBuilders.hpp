/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   Header file for various index set builder methods.
 *
 ******************************************************************************
 */

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-18, Lawrence Livermore National Security, LLC.
//
// Produced at the Lawrence Livermore National Laboratory
//
// LLNL-CODE-689114
//
// All rights reserved.
//
// This file is part of RAJA.
//
// For details about use and distribution, please read RAJA/LICENSE.
//
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef RAJA_IndexSetBuilders_HPP
#define RAJA_IndexSetBuilders_HPP

#include "RAJA/config.hpp"

#include "RAJA/index/IndexSet.hpp"
#include "RAJA/util/types.hpp"

namespace RAJA
{

/*!
 ******************************************************************************
 *
 * \brief Initialize index set with aligned Ranges and List segments from
 *        array of indices with given length.
 *
 *        Specifically, Range segments will be greater than RANGE_MIN_LENGTH
 *        and starting index and length of each range segment will be
 *        multiples of RANGE_ALIGN. These constants are defined in the
 *        RAJA config.hpp header file.
 *
 *        Routine does no error-checking on argements and assumes Index_type
 *        array contains valid indices.
 *
 * Note: Method assumes TypedIndexSet reference refers to an empty index set.
 *
 ******************************************************************************
 */
void buildTypedIndexSetAligned(IndexSet& hiset,
                               const Index_type* const indices_in,
                               Index_type length);

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
//
// The following methods build "lock-free" index sets.
//
// Lock-free indexsets are designed to be used with coarse-grained OpenMP
// iteration policies.  The "lock-free" part here assumes interactions among
// the cell-complex associated with the space being partitioned are "tightly
// bound".
//
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

/*
 ******************************************************************************
 *
 * Initialize lock-free "block" index set (planar division).
 *
 * The method chunks a fastDim x midDim x slowDim mesh into blocks that can
 * be dependency-scheduled, removing need for lock constructs.
 *
 * Note: Method assumes TypedIndexSet reference refers to an empty index set.
 *
 ******************************************************************************
 */
void buildLockFreeBlockIndexset(
    RAJA::TypedIndexSet<RAJA::RangeSegment,
                        RAJA::ListSegment,
                        RAJA::RangeStrideSegment>& iset,
    int fastDim,
    int midDim,
    int slowDim);

/*
 ******************************************************************************
 *
 * Build Lock-free "color" index set. The domain-set is colored based on
 * connectivity to the range-set. All elements in each segment are
 * independent, and no two segments can be executed in parallel.
 *
 * Note: Method assumes TypedIndexSet reference refers to an empty index set.
 *
 ******************************************************************************
 */
void buildLockFreeColorIndexset(
    RAJA::TypedIndexSet<RAJA::RangeSegment,
                        RAJA::ListSegment,
                        RAJA::RangeStrideSegment>& iset,
    Index_type const* domainToRange,
    int numEntity,
    int numRangePerDomain,
    int numEntityRange,
    Index_type* elemPermutation = 0l,
    Index_type* ielemPermutation = 0l);

}  // closing brace for RAJA namespace

#endif  // closing endif for header file include guard
