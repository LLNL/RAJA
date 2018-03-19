/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   Main RAJA header file.
 *
 *          This is the main header file to include in code that uses RAJA.
 *          It includes other RAJA headers files that define types, index
 *          sets, ieration methods, etc.
 *
 *          IMPORTANT: If changes are made to this file, note that contents
 *                     of some header files require that they are included
 *                     in the order found here.
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

#ifndef RAJA_HPP
#define RAJA_HPP

#include "RAJA/config.hpp"

#include "RAJA/util/defines.hpp"

#include "RAJA/util/types.hpp"


#include "RAJA/util/Operators.hpp"

#include "RAJA/util/basic_mempool.hpp"

#include "RAJA/util/camp_aliases.hpp"


//
// Generic iteration templates require specializations defined
// in the files included below.
//
#include "RAJA/pattern/forall.hpp"
#include "RAJA/pattern/kernel.hpp"


//
// Shared memory abstractions
//
#include "RAJA/pattern/shared_memory.hpp"

//
// All platforms must support sequential execution.
//
#include "RAJA/policy/sequential.hpp"

//
// All platforms must support loop execution.
//
#include "RAJA/policy/loop.hpp"

//
// All platforms should support simd execution.
//
#include "RAJA/policy/simd.hpp"

#if defined(RAJA_ENABLE_TBB)
#include "RAJA/policy/tbb.hpp"
#endif

#if defined(RAJA_ENABLE_CUDA)
#include "RAJA/policy/cuda.hpp"
#endif

#if defined(RAJA_ENABLE_OPENMP)
#include "RAJA/policy/openmp.hpp"
#endif

#include "RAJA/index/IndexSet.hpp"

//
// Strongly typed index class
//
#include "RAJA/index/IndexValue.hpp"


#include "RAJA/policy/MultiPolicy.hpp"


//
// Multidimensional layouts and views
//
#include "RAJA/util/Layout.hpp"
#include "RAJA/util/OffsetLayout.hpp"
#include "RAJA/util/PermutedLayout.hpp"
#include "RAJA/util/StaticLayout.hpp"
#include "RAJA/util/View.hpp"

//
// Shared memory view patterns
//
#include "RAJA/util/ShmemTile.hpp"

//
// Atomic operations support
//
#include "RAJA/pattern/atomic.hpp"


//
// Generic iteration templates for perfectly nested loops
//
#include "RAJA/pattern/forallN.hpp"


//
// Reduction objects
//
#include "RAJA/pattern/reduce.hpp"


//
// Synchronization
//
#include "RAJA/pattern/synchronize.hpp"

//
//////////////////////////////////////////////////////////////////////
//
// These contents of the header files included here define index set
// and segment execution methods whose implementations depend on
// programming model choice.
//
// The ordering of these file inclusions must be preserved since there
// are dependencies among them.
//
//////////////////////////////////////////////////////////////////////
//

#include "RAJA/index/IndexSetUtils.hpp"

// Tiling policies
#include "RAJA/pattern/tile.hpp"

// Loop interchange policies
#include "RAJA/pattern/permute.hpp"

#include "RAJA/pattern/scan.hpp"

#endif  // closing endif for header file include guard
