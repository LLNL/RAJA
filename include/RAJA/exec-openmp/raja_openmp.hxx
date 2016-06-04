/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   Header file containing RAJA headers for OpenMP execution.
 *
 *          These methods work only on platforms that support OpenMP. 
 *
 ******************************************************************************
 */

#ifndef RAJA_openmp_HXX
#define RAJA_openmp_HXX

#include "RAJA/config.hxx"

#if defined(RAJA_USE_OPENMP)

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

#include "RAJA/PolicyBase.hxx"

namespace RAJA {

//
//////////////////////////////////////////////////////////////////////
//
// Execution policies
//
//////////////////////////////////////////////////////////////////////
//

///
/// Segment execution policies
///
struct omp_parallel_for_exec : public PolicyBase {
    template<typename IndexT = Index_type,
             typename Func>
    void range(IndexT begin, IndexT end, Func &&f) const {
        // printf("yup omp pfor1...\n");
#pragma omp parallel for schedule(static)
        for ( auto ii = begin ; ii < end ; ++ii ) {
            loop_body( ii );
        }
    }

    template<typename Iterator,
             typename Func>
    void iterator(Iterator &&begin, Iterator &&end, Func &&loop_body) const {
        // printf("yup omp pfor2...\n");
#pragma omp parallel for schedule(static)
        for ( auto &ii = begin ; ii < end ; ++ii ) {
            loop_body( *ii );
        }
    }
};
//struct omp_parallel_for_nowait_exec {};
struct omp_for_nowait_exec : public PolicyBase {
    template<typename IndexT = Index_type,
             typename Func>
    void range(IndexT begin, IndexT end, Func &&f) const {
#pragma omp for schedule(static) nowait
        for ( auto ii = begin ; ii < end ; ++ii ) {
            loop_body( ii );
        }
    }

    template<typename Iterator,
             typename Func>
    void iterator(Iterator &&begin, Iterator &&end, Func &&loop_body) const {
#pragma omp for schedule(static) nowait
        for ( auto &ii = begin ; ii < end ; ++ii ) {
            loop_body( *ii );
        }
    }
};

///
/// Index set segment iteration policies
///
struct omp_parallel_for_segit : public PolicyBase {};
struct omp_parallel_segit : public PolicyBase {};
struct omp_taskgraph_segit : public PolicyBase {};
struct omp_taskgraph_interval_segit : public PolicyBase {};

///
///////////////////////////////////////////////////////////////////////
///
/// Reduction execution policies
///
///////////////////////////////////////////////////////////////////////
///
struct omp_reduce {};

}  // closing brace for RAJA namespace


#include "RAJA/exec-openmp/reduce_openmp.hxx"
#include "RAJA/exec-openmp/forall_openmp.hxx"

#if defined(RAJA_USE_NESTED)
#include "RAJA/exec-openmp/forallN_openmp.hxx"
#endif

#endif  // closing endif for if defined(RAJA_USE_OPENMP)

#endif  // closing endif for header file include guard

