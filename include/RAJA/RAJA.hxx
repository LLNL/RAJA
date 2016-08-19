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

#ifndef RAJA_HXX
#define RAJA_HXX

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

#include "RAJA/config.hxx"

#include "RAJA/internal/defines.hxx"

#include "RAJA/int_datatypes.hxx"
#include "RAJA/real_datatypes.hxx"

#include "RAJA/reducers.hxx"

#include "RAJA/IndexSet.hxx"
#include "RAJA/ListSegment.hxx"
#include "RAJA/RangeSegment.hxx"

//
// Strongly typed index class.
//
#include "RAJA/IndexValue.hxx"

//
// Generic iteration templates require specializations defined
// in the files included below.
//
#include "RAJA/forall_generic.hxx"

#if defined(RAJA_ENABLE_NESTED)

//
// Multidimensional layouts and views.
//
#include "RAJA/foralln/Layout.hxx"
#include "RAJA/foralln/View.hxx"

//
// Generic iteration templates for perfectly nested loops
//
#include "RAJA/foralln/Generic.hxx"

#endif  // defined(RAJA_ENABLE_NESTED)

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

//
// All platforms must support sequential execution.
//
#include "RAJA/exec-sequential/raja_sequential.hxx"

//
// All platforms should support simd execution.
//
#include "RAJA/exec-simd/raja_simd.hxx"

#if defined(RAJA_ENABLE_CUDA)
#include "RAJA/exec-cuda/raja_cuda.hxx"
#endif

#if defined(RAJA_ENABLE_OPENMP)
#include "RAJA/exec-openmp/raja_openmp.hxx"
#endif

#if defined(RAJA_ENABLE_CILK)
#include "RAJA/exec-cilk/raja_cilk.hxx"
#endif

#include "RAJA/IndexSetUtils.hxx"

#if defined(RAJA_ENABLE_NESTED)

//
// Perfectly nested loop transformations
//

// Tiling policies
#include "RAJA/foralln/Tile.hxx"

// Loop interchange policies
#include "RAJA/foralln/Permute.hxx"

#endif  // defined(RAJA_ENABLE_NESTED)

#endif  // closing endif for header file include guard
