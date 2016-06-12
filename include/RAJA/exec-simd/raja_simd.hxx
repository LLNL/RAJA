/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   Header file containing RAJA headers for SIMD segment execution.
 *
 *          These methods work on all platforms.
 *
 ******************************************************************************
 */

#ifndef RAJA_simd_HXX
#define RAJA_simd_HXX

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

//
//////////////////////////////////////////////////////////////////////
//
// Execution policies
//
//////////////////////////////////////////////////////////////////////
//

#include "RAJA/PolicyBase.hxx"

///
/// Segment execution policies
///
namespace RAJA {

struct simd_exec : public PolicyBase {
    template<typename IndexT = Index_type,
             typename Func,
             typename std::enable_if<!std::is_base_of<
                 std::random_access_iterator_tag,
                 typename std::iterator_traits<IndexT>::iterator_category>::value>::type * = nullptr>
    inline void operator()(IndexT begin, IndexT end, Func &&f) const {
        RAJA_SIMD
        for ( auto ii = begin ; ii < end ; ++ii ) {
            f( ii );
        }
    }

    template<typename Iterator,
             typename Func>
    inline void operator()(Iterator &&begin, Iterator &&end, Func &&loop_body) const {
        RAJA_SIMD
        for ( auto &ii = begin ; ii < end ; ++ii ) {
            loop_body( *ii );
        }
    }
};

}

//
// NOTE: There is no Index set segment iteration policy for SIMD
//


///
///////////////////////////////////////////////////////////////////////
///
/// Reduction execution policies
///
///////////////////////////////////////////////////////////////////////
///

//
// NOTE: RAJA reductions in SIMD loops use seg_reduce policy
//

#include "RAJA/exec-simd/forall_simd.hxx"

#endif  // closing endif for header file include guard

