/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   Header file containing RAJA index set and segment iteration 
 *          template methods for Intel Cilk Plus execution.
 *
 *          These methods work only on platforms that support Cilk Plus. 
 *
 ******************************************************************************
 */

#ifndef RAJA_forall_cilk_HXX
#define RAJA_forall_cilk_HXX

#if defined(RAJA_USE_CILK)

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

#include "RAJA/int_datatypes.hxx"

#include "RAJA/fault_tolerance.hxx"

#include "RAJA/segment_exec.hxx"

#include <cilk/cilk.h>
#include <cilk/cilk_api.h>


namespace RAJA {


//
//////////////////////////////////////////////////////////////////////
//
// Function templates that iterate over index ranges.
//
//////////////////////////////////////////////////////////////////////
//

/*!
 ******************************************************************************
 *
 * \brief  cilk_for iteration over index range.
 *
 ******************************************************************************
 */
template <typename LOOP_BODY>
RAJA_INLINE
void forall(cilk_for_exec,
            Index_type begin, Index_type end, 
            LOOP_BODY loop_body)
{
   RAJA_FT_BEGIN ;

   cilk_for ( Index_type ii = begin ; ii < end ; ++ii ) {
      loop_body( ii );
   }

   RAJA_FT_END ;
}

/*!
 ******************************************************************************
 *
 * \brief  cilk_for iteration over index range with index count.
 *
 *         NOTE: lambda loop body requires two args (icount, index).
 *
 ******************************************************************************
 */
template <typename LOOP_BODY>
RAJA_INLINE
void forall_Icount(cilk_for_exec,
                   Index_type begin, Index_type end,
                   Index_type icount,
                   LOOP_BODY loop_body)
{
   Index_type loop_end = end - begin;

   RAJA_FT_BEGIN ;

   cilk_for ( Index_type ii = 0 ; ii < loop_end ; ++ii ) {
      loop_body( ii+icount, ii+begin );
   }

   RAJA_FT_END ;
}


//
//////////////////////////////////////////////////////////////////////
//
// Function templates that iterate over range segments. 
//
//////////////////////////////////////////////////////////////////////
//

/*!
 ******************************************************************************
 *
 * \brief  cilk_for  iteration over range segment object.
 *
 ******************************************************************************
 */
template <typename LOOP_BODY>
RAJA_INLINE
void forall(cilk_for_exec,
            const RangeSegment& iseg,
            LOOP_BODY loop_body)
{
   Index_type begin = iseg.getBegin();
   Index_type end   = iseg.getEnd();

   RAJA_FT_BEGIN ;

   cilk_for ( Index_type ii = begin ; ii < end ; ++ii ) {
      loop_body( ii );
   }

   RAJA_FT_END ;
}

/*!
 ******************************************************************************
 *
 * \brief  cilk_for iteration over range segment object with index count.
 *
 *         NOTE: lambda loop body requires two args (icount, index).
 *
 ******************************************************************************
 i/
template <typename LOOP_BODY>
RAJA_INLINE
void forall_Icount(cilk_for_exec,
                   const RangeSegment& iseg,
                   Index_type icount,
                   LOOP_BODY loop_body)
{
   Index_type begin = iseg.getBegin();
   Index_type loop_end = iseg.getEnd() - begin;

   RAJA_FT_BEGIN ;

   cilk_for ( Index_type ii = 0 ; ii < loop_end ; ++ii ) {
      loop_body( ii+icount, ii+begin );
   }

   RAJA_FT_END ;
}


//
//////////////////////////////////////////////////////////////////////
//
// Function templates that iterate over raw Box.
//
//////////////////////////////////////////////////////////////////////
//

/*!
 ******************************************************************************
 *
 * \brief  cilk_for iteration over raw Box.
 *         
 ******************************************************************************
 */

/*!
 ******************************************************************************
 *
 * \brief  cilk_for iteration over raw Box with index count.
 *
 *         NOTE: lambda loop body requires two args (icount, index).
 *
 ******************************************************************************
 */

//
//////////////////////////////////////////////////////////////////////
//
// Function templates that iterate over Box segment objects.
//
//////////////////////////////////////////////////////////////////////
//

#if defined(RAJA_USE_BOXSEGMENT)
/*!
 ******************************************************************************
 *
 * \brief  cilk_for iteration over Box segment object.
 *
 ******************************************************************************
 */

template <typename LOOP_BODY>
RAJA_INLINE
void forall(cilk_for_exec,
            const BoxSegment& iseg,
            LOOP_BODY loop_body)
{
   Index_type        corner = iseg.getCorner();
   Index_type        dim    = iseg.getDim();
   Index_type const *extent = iseg.getExtent();
   Index_type const *stride = iseg.getStride();

   RAJA_FT_BEGIN ;

   if (dim == 2) {
      Index_type xExtent = extent[0] ; /* fast stride */
      Index_type xStride = stride[0] ;
      Index_type yExtent = extent[1] ; /* slow stride */
      Index_type yStride = stride[1] ;

      cilk_for ( Index_type y = 0 ; y < yExtent ; ++y ) {
         Index_type idx = corner + y*yStride ;
         for (Index_type x = 0 ; x < xExtent ; ++x) {
            loop_body( idx );
            idx += xStride ;
         }
      }
   }
   else if (dim == 1) {
      Index_type xExtent = extent[0] ; /* fast stride */
      Index_type xStride = stride[0] ;

      cilk_for (Index_type x = 0 ; x < xExtent ; ++x) {
         loop_body( corner + x*xStride );
      }
   }
   else /* dim == 3 */ {
      /* need to but some code here */
   }

   RAJA_FT_END ;
}


/*!
 ******************************************************************************
 *
 * \brief  cilk_for iteration over Box segment object with index count.
 *
 *         NOTE: lambda loop body requires two args (icount, index).
 *
 ******************************************************************************
 */

template <typename LOOP_BODY>
RAJA_INLINE
void forall_Icount(cilk_for_exec,
            const BoxSegment& iseg,
            Index_type icount,
            LOOP_BODY loop_body)
{
   Index_type        corner = iseg.getCorner();
   Index_type        dim    = iseg.getDim();
   Index_type const *extent = iseg.getExtent();
   Index_type const *stride = iseg.getStride();

   RAJA_FT_BEGIN ;

   if (dim == 2) {
      Index_type xExtent = extent[0] ; /* fast stride */
      Index_type xStride = stride[0] ;
      Index_type yExtent = extent[1] ; /* slow stride */
      Index_type yStride = stride[1] ;

      cilk_for ( Index_type y = 0 ; y < yExtent ; ++y ) {
         Index_type idx = corner + y*yStride ;
         Index_type myCount = icount + y*xExtent ;
         for (Index_type x = 0 ; x < xExtent ; ++x) {
            loop_body( myCount, idx );
            idx += xStride ;
            ++myCount ;
         }
      }
    }
    else if (dim == 1) {
      Index_type xExtent = extent[0] ; /* fast stride */
      Index_type xStride = stride[0] ;

      cilk_for (Index_type x = 0 ; x < xExtent ; ++x) {
         loop_body( icount + x, corner + x*xStride );
      }
    }
    else /* dim == 3 */ {
       /* need to but some code here */
    }

   RAJA_FT_END ;
}
#endif  // defined(RAJA_USE_BOXSEGMENT)


//
//////////////////////////////////////////////////////////////////////
//
// Function templates that iterate over indirection arrays.
//
//////////////////////////////////////////////////////////////////////
//

/*!
 ******************************************************************************
 *
 * \brief  cilk_for iteration over indirection array.
 *
 ******************************************************************************
 */
template <typename LOOP_BODY>
RAJA_INLINE
void forall(cilk_for_exec,
            const Index_type* __restrict__ idx, Index_type len,
            LOOP_BODY loop_body)
{
   RAJA_FT_BEGIN ;

   cilk_for ( Index_type k = 0 ; k < len ; ++k ) {
      loop_body( idx[k] );
   }

   RAJA_FT_END ;
}

/*!
 ******************************************************************************
 *
 * \brief  cilk_for iteration over indirection array with index count.
 *
 *         NOTE: lambda loop body requires two args (icount, index).
 *
 ******************************************************************************
 */
template <typename LOOP_BODY>
RAJA_INLINE
void forall_Icount(cilk_for_exec,
                   const Index_type* __restrict__ idx, Index_type len,
                   Index_type icount,
                   LOOP_BODY loop_body)
{
   RAJA_FT_BEGIN ;

   cilk_for ( Index_type k = 0 ; k < len ; ++k ) {
      loop_body( k+icount, idx[k] );
   }

   RAJA_FT_END ;
}


//
//////////////////////////////////////////////////////////////////////
//
// Function templates that iterate over list segment objects.
//
//////////////////////////////////////////////////////////////////////
//

/*!
 ******************************************************************************
 *
 * \brief  cilk_for iteration over list segment object.
 *
 ******************************************************************************
 */
template <typename LOOP_BODY>
RAJA_INLINE
void forall(cilk_for_exec,
            const ListSegment& iseg,
            LOOP_BODY loop_body)
{
   const Index_type* __restrict__ idx = iseg.getIndex();
   Index_type len = iseg.getLength();

   RAJA_FT_BEGIN ;

   cilk_for ( Index_type k = 0 ; k < len ; ++k ) {
      loop_body( idx[k] );
   }

   RAJA_FT_END ;
}

/*!
 ******************************************************************************
 *
 * \brief  cilk_for iteration over list segment object with index count.
 *
 *         NOTE: lambda loop body requires two args (icount, index).
 *
 ******************************************************************************
 */
template <typename LOOP_BODY>
RAJA_INLINE
void forall_Icount(cilk_for_exec,
                   const ListSegment& iseg,
                   Index_type icount,
                   LOOP_BODY loop_body)
{
   const Index_type* __restrict__ idx = iseg.getIndex();
   Index_type len = iseg.getLength();

   RAJA_FT_BEGIN ;

   cilk_for ( Index_type k = 0 ; k < len ; ++k ) {
      loop_body( k+icount, idx[k] );
   }

   RAJA_FT_END ;
}


//
//////////////////////////////////////////////////////////////////////
//
// The following function templates iterate over index set
// segments using cilk_for. Segment execution is defined by 
// segment execution policy template parameter.
//
//////////////////////////////////////////////////////////////////////
//

/*!
 ******************************************************************************
 *
 * \brief  cilk_for iteration over segments of index set and 
 *         use execution policy template parameter to execute segments.
 *
 ******************************************************************************
 */
template <typename SEG_EXEC_POLICY_T,
          typename LOOP_BODY>
RAJA_INLINE
void forall( IndexSet::ExecPolicy<cilk_for_segit, SEG_EXEC_POLICY_T>,
             const IndexSet& iset, LOOP_BODY loop_body )
{
   int num_seg = iset.getNumSegments();
   cilk_for ( int isi = 0; isi < num_seg; ++isi ) {

      const IndexSetSegInfo* seg_info = iset.getSegmentInfo(isi);
      executeRangeList_forall<SEG_EXEC_POLICY_T>(seg_info, loop_body);

   } // iterate over segments of index set
}

/*!
 ******************************************************************************
 *
 * \brief  cilk_for iteration over segments of index set and
 *         use execution policy template parameter to execute segments.
 *
 *         This method passes count segment iteration.
 *
 *         NOTE: lambda loop body requires two args (icount, index).
 *
 ******************************************************************************
 */
template <typename SEG_EXEC_POLICY_T,
          typename LOOP_BODY>
RAJA_INLINE
void forall_Icount( IndexSet::ExecPolicy<cilk_for_segit, SEG_EXEC_POLICY_T>,
                    const IndexSet& iset, LOOP_BODY loop_body )
{
   int num_seg = iset.getNumSegments();
   cilk_for ( int isi = 0; isi < num_seg; ++isi ) {

      const IndexSetSegInfo* seg_info = iset.getSegmentInfo(isi);
      executeRangeList_forall_Icount<SEG_EXEC_POLICY_T>(seg_info, loop_body);

   } // iterate over segments of index set
}


}  // closing brace for RAJA namespace

#endif  // closing endif for RAJA_USE_CILK guard

#endif  // closing endif for header file include guard

