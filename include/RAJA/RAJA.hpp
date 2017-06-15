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

#ifndef RAJA_HPP
#define RAJA_HPP

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

#include "RAJA/util/types.hpp"

#include "RAJA/util/Operators.hpp"

//
// All platforms must support sequential execution.
//
#include "RAJA/policy/sequential.hpp"

//
// All platforms should support simd execution.
//
#include "RAJA/policy/simd.hpp"

#if defined(RAJA_ENABLE_CUDA)
#include "RAJA/policy/cuda.hpp"
#endif

#if defined(RAJA_ENABLE_OPENMP)
#include "RAJA/policy/openmp.hpp"
#endif

#if defined(RAJA_ENABLE_CILK)
#include "RAJA/policy/cilk.hpp"
#endif

//
// Strongly typed index class.
//
#include "RAJA/index/IndexValue.hpp"

#include "RAJA/policy/MultiPolicy.hpp"

//
// Generic iteration templates require specializations defined
// in the files included below.
//
#include "RAJA/pattern/forall.hpp"

//
// Multidimensional layouts and views.
//
#include "RAJA/util/Layout.hpp"
#include "RAJA/util/OffsetLayout.hpp"
#include "RAJA/util/PermutedLayout.hpp"
#include "RAJA/util/View.hpp"

//
// Generic iteration templates for perfectly nested loops
//
#include "RAJA/pattern/forallN.hpp"


#include "RAJA/pattern/reduce.hpp"

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
