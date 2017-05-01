/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   Header file for various index set builder methods.
 *
 ******************************************************************************
 */

#ifndef RAJA_IndexSetBuilders_HXX
#define RAJA_IndexSetBuilders_HXX

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

#include "RAJA/util/types.hpp"

namespace RAJA
{

class IndexSet;

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
 * Note: Method assumes IndexSet reference refers to an empty index set.
 *
 ******************************************************************************
 */
void buildIndexSetAligned(IndexSet& hiset,
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
 * Note: Method assumes IndexSet reference refers to an empty index set.
 *
 ******************************************************************************
 */
void buildLockFreeBlockIndexset(IndexSet& iset,
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
 * Note: Method assumes IndexSet reference refers to an empty index set.
 *
 ******************************************************************************
 */
void buildLockFreeColorIndexset(IndexSet& iset,
                                int const* domainToRange,
                                int numEntity,
                                int numRangePerDomain,
                                int numEntityRange,
                                int* elemPermutation = 0l,
                                int* ielemPermutation = 0l);

}  // closing brace for RAJA namespace

#endif  // closing endif for header file include guard
