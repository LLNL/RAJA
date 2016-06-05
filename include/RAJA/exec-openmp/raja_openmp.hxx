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

#include <thread>
#include <iostream>
#include <omp.h>

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
struct omp_parallel_for_segit : public SegmentPolicyBase {
    template<typename Iterator,
             typename Func>
    void iterator(Iterator &&begin, Iterator &&end, Func &&loop_body) const {
#pragma omp parallel for schedule(static, 1)
        for ( auto &ii = begin ; ii < end ; ++ii ) {
            loop_body( *ii );
        }
    }
};
struct omp_parallel_segit : public SegmentPolicyBase {
};
struct omp_taskgraph_segit : public SegmentPolicyBase {
    template<typename Func>
    void indexset(IndexSet & iset, Func &&loop_body) const {
        if ( !iset.dependencyGraphSet() ) {
            std::cerr << "\n RAJA IndexSet dependency graph not set , "
                << "FILE: "<< __FILE__ << " line: "<< __LINE__ << std::endl;
            exit(1);
        }


        IndexSet& ncis = (*const_cast<IndexSet *>(&iset)) ;

        int num_seg = ncis.getNumSegments();

#pragma omp parallel for schedule(static, 1)
        for ( int isi = 0; isi < num_seg; ++isi ) {

            IndexSetSegInfo* seg_info = ncis.getSegmentInfo(isi);
            DepGraphNode* task  = seg_info->getDepGraphNode();

#pragma omp critical
            std::cerr << omp_get_thread_num()
                      << " seg_info:"
                      << seg_info
                      << " task: "
                      << task
                      << std::endl;
            //
            // This is declared volatile to prevent compiler from
            // optimizing the while loop (into an if-statement, for example).
            // It may not be able to see that the value accessed through
            // the method call will be changed at the end of the for-loop
            // from another executing thread.
            //
            volatile int* __restrict__ semVal = &(task->semaphoreValue());

            while(*semVal != 0) {
                /* spin or (better) sleep here */ ;
                // printf("%d ", *semVal) ;
                // sleep(1) ;
                // for (volatile int spin = 0; spin<1000; ++spin) {
                //    spin = spin ;
                // }
                sched_yield() ;
            }

            loop_body(*seg_info);

            if (task->semaphoreReloadValue() != 0) {
                task->semaphoreValue() = task->semaphoreReloadValue() ;
            }

            if (task->numDepTasks() != 0) {
                for (int ii = 0; ii < task->numDepTasks(); ++ii) {
                    // Alternateively, we could get the return value of this call
                    // and actively launch the task if we are the last depedent
                    // task. In that case, we would not need the semaphore spin
                    // loop above.
                    int seg = task->depTaskNum(ii) ;
                    DepGraphNode* dep  = ncis.getSegmentInfo(seg)->getDepGraphNode();
                    __sync_fetch_and_sub(&(dep->semaphoreValue()), 1) ;
                }
            }

        } // iterate over segments of index set
    }
};
struct omp_taskgraph_interval_segit : public SegmentPolicyBase {};

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

