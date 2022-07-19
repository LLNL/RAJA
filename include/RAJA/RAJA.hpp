/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   Main RAJA header file.
 *
 *          This is the main header file to include in code that uses RAJA.
 *          It provides a single access point to all RAJA features by
 *          including other RAJA headers.
 *
 *          IMPORTANT: If changes are made to this file, note that contents
 *                     of some header files require that they are included
 *                     in the order found here.
 *
 ******************************************************************************
 */

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-22, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef RAJA_HPP
#define RAJA_HPP

#include "RAJA/config.hpp"

#include "RAJA/util/Operators.hpp"
#include "RAJA/util/basic_mempool.hpp"
#include "RAJA/util/camp_aliases.hpp"
#include "RAJA/util/macros.hpp"
#include "RAJA/util/types.hpp"
#include "RAJA/util/plugins.hpp"
#include "RAJA/util/Registry.hpp"


//
// Generic iteration templates require specializations defined
// in the files included below.
//
#include "RAJA/pattern/forall.hpp"
#include "RAJA/pattern/kernel.hpp"
#include "RAJA/pattern/teams.hpp"

//
// Generic templates to describe SIMD/SIMT registers and vectors
//
#include "RAJA/pattern/tensor.hpp"

//
// All platforms must support sequential execution.
//
#include "RAJA/policy/sequential.hpp"

//
// All platforms must support loop execution.
//
#include "RAJA/policy/loop.hpp"

//
// All platforms should support simd and vector execution.
//
#include "RAJA/policy/simd.hpp"
#include "RAJA/policy/tensor.hpp"

#if defined(RAJA_ENABLE_TBB)
#include "RAJA/policy/tbb.hpp"
#endif

#if defined(RAJA_ENABLE_CUDA)
#include "RAJA/policy/cuda.hpp"
#endif

#if defined(RAJA_ENABLE_HIP)
#include "RAJA/policy/hip.hpp"
#endif

#if defined(RAJA_ENABLE_SYCL)
#include "RAJA/policy/sycl.hpp"
#endif

#if defined(RAJA_ENABLE_OPENMP)
#include "RAJA/policy/openmp.hpp"

#if defined(RAJA_ENABLE_TARGET_OPENMP)
#include "RAJA/policy/openmp_target.hpp"
#endif
#endif

#if defined(RAJA_ENABLE_DESUL_ATOMICS)
    #include "RAJA/policy/desul.hpp"
#endif

#include "RAJA/index/IndexSet.hpp"

//
// Strongly typed index class
//
#include "RAJA/index/IndexValue.hpp"


//
// Generic iteration templates require specializations defined
// in the files included below.
//
#include "RAJA/pattern/forall.hpp"
#include "RAJA/pattern/region.hpp"

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
// View for sequences of objects
//
#include "RAJA/util/Span.hpp"

//
// zip iterator to iterator over sequences simultaneously
//
#include "RAJA/util/zip.hpp"

//
// Atomic operations support
//
#include "RAJA/pattern/atomic.hpp"

//
// Shared memory view patterns
//
#include "RAJA/util/LocalArray.hpp"

//
// Bit masking operators
//
#include "RAJA/util/BitMask.hpp"

//
// sort algorithms
//
#include "RAJA/util/sort.hpp"

//
// WorkPool, WorkGroup, WorkSite objects
//
#include "RAJA/policy/WorkGroup.hpp"
#include "RAJA/pattern/WorkGroup.hpp"

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
#include "RAJA/index/IndexSetBuilders.hpp"

#include "RAJA/pattern/scan.hpp"

#if defined(RAJA_ENABLE_RUNTIME_PLUGINS)
#include "RAJA/util/PluginLinker.hpp"
#endif

#include "RAJA/pattern/sort.hpp"

namespace RAJA {
namespace expt{}
  // provide a RAJA::expt namespace for experimental work, but bring alias
  // it into RAJA so it doesn't affect user code
  using namespace expt;
}


void check1(double* A, double* x)
{

    using mat_t = RAJA::expt::RectMatrixRegister<double, RAJA::expt::RowMajorLayout, 16, 4, RAJA::avx2_register>;
    using row_t = RAJA::expt::RowIndex<int, mat_t>;
    using col_t = RAJA::expt::ColIndex<int, mat_t>;

    using vec_t = RAJA::expt::VectorRegister<double>;
    using idx_t = RAJA::expt::VectorIndex<int, vec_t>;

    double y[16];

    auto aa   = RAJA::View<double, RAJA::StaticLayout<RAJA::PERM_IJ,16,4>>( A );
    auto xV   = RAJA::View<double, RAJA::StaticLayout<RAJA::PERM_I ,16>>  ( x );
    auto yV   = RAJA::View<double, RAJA::StaticLayout<RAJA::PERM_I ,16>>  ( y );
    //auto cc   = RAJA::View<double, RAJA::Layout<2>>( C, 16, 4 );


    auto rall = row_t::static_all();
    auto call = col_t::static_all();
    auto vall = idx_t::static_all();

    yV(vall) = aa(rall,call) * xV(vall);
    xV(vall) = yV(vall);
   
}


void check2(double* A, double* x)
{


    using mat_t = RAJA::expt::RectMatrixRegister<double, RAJA::expt::RowMajorLayout, 4, 4, RAJA::avx2_register>;
    using row_t = RAJA::expt::RowIndex<int, mat_t>;
    using col_t = RAJA::expt::ColIndex<int, mat_t>;

    double y[16];

    #if 0
    auto aa   = RAJA::View<double, RAJA::StaticLayout<RAJA::PERM_IJ, 4, 4>>  ( A );
    auto xx   = RAJA::View<double, RAJA::StaticLayout<RAJA::PERM_IJ, 4, 4>>  ( x );
    auto yy   = RAJA::View<double, RAJA::StaticLayout<RAJA::PERM_IJ, 4, 4>>  ( y );
    #else
    auto aa   = RAJA::View<double, RAJA::Layout<2>>  ( A,  4,4 );
    auto xx   = RAJA::View<double, RAJA::Layout<2>>  ( x,  4,4 );
    auto yy   = RAJA::View<double, RAJA::Layout<2>>  ( y,  4,4 );
    #endif

    auto rall = row_t::static_all();
    auto call = col_t::static_all();

    for(int i=0; i <4; i++){
        yy(row_t::static_range<0,1>(),call) += aa(i,0) * xx(row_t::static_range<0,1>(),call);
        yy(row_t::static_range<1,2>(),call) += aa(i,1) * xx(row_t::static_range<1,2>(),call);
        yy(row_t::static_range<2,3>(),call) += aa(i,2) * xx(row_t::static_range<2,3>(),call);
        yy(row_t::static_range<3,4>(),call) += aa(i,3) * xx(row_t::static_range<3,4>(),call);
    }

    xx(rall,call) = yy(rall,call);
    
}


#endif  // closing endif for header file include guard
