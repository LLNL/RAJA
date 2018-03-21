/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   Implementation file for aligned-range index set builder methods.
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

#include "RAJA/index/IndexSet.hpp"
#include "RAJA/index/ListSegment.hpp"
#include "RAJA/index/RangeSegment.hpp"

namespace RAJA
{

/*
*************************************************************************
*
* Initialize index set with aligned Ranges and List segments from array
* of indices with given length.
*
*************************************************************************
*/

void buildIndexSetAligned(RAJA::TypedIndexSet<RAJA::RangeSegment,
                          RAJA::ListSegment>& hiset,
                          const Index_type* const indices_in,
                          Index_type length)
{
  if (length == 0) return;

  /* only transform relatively large */
  if (length > RANGE_MIN_LENGTH) {
    /* build a rindex array from an index array */
    Index_type docount = 0;
    Index_type inrange = -1;

    /****************************/
    /* first, gather statistics */
    /****************************/

    Index_type scanVal = indices_in[0];
    Index_type sliceCount = 0;
    for (Index_type ii = 1; ii < length; ++ii) {
      Index_type lookAhead = indices_in[ii];

      if (inrange == -1) {
        if ((lookAhead == scanVal + 1) && ((scanVal % RANGE_ALIGN) == 0)) {
          inrange = 1;
        } else {
          inrange = 0;
        }
      }

      if (lookAhead == scanVal + 1) {
        if ((inrange == 0) && ((scanVal % RANGE_ALIGN) == 0)) {
          if (sliceCount != 0) {
            docount += 1 + sliceCount; /* length + singletons */
          }
          inrange = 1;
          sliceCount = 0;
        }
        ++sliceCount; /* account for scanVal */
      } else {
        if (inrange == 1) {
          /* we can tighten this up by schleping any trailing */
          /* sigletons off into the subsequent singleton */
          /* array.  We would then also need to recheck the */
          /* final length of the range to make sure it meets */
          /* our minimum length crietria.  If it doesnt, */
          /* we need to emit a random array instead of */
          /* a range array */
          ++sliceCount;
          docount += 2; /* length + begin */
          inrange = 0;
          sliceCount = 0;
        } else {
          ++sliceCount; /* account for scanVal */
        }
      }

      scanVal = lookAhead;
    }  // end loop to gather statistics

    if (inrange != -1) {
      if (inrange) {
        ++sliceCount;
        docount += 2; /* length + begin */
      } else {
        ++sliceCount;
        docount += 1 + sliceCount; /* length + singletons */
      }
    } else if (scanVal != -1) {
      ++sliceCount;
      docount += 2;
    }
    ++docount; /* zero length termination */

    /* What is the cutoff criteria for generating the rindex array? */
    if (docount < (length * (RANGE_ALIGN - 1)) / RANGE_ALIGN) {
      /* The rindex array can either contain a pointer into the */
      /* original index array, *or* it can repack the data from the */
      /* original index array.  Benefits of repacking could include */
      /* better use of hardware prefetch streams, or guaranteeing */
      /* alignment of index array segments. */

      /*******************************/
      /* now, build the rindex array */
      /*******************************/

      Index_type dobegin;
      inrange = -1;

      scanVal = indices_in[0];
      sliceCount = 0;
      dobegin = scanVal;
      for (Index_type ii = 1; ii < length; ++ii) {
        Index_type lookAhead = indices_in[ii];

        if (inrange == -1) {
          if ((lookAhead == scanVal + 1) && ((scanVal % RANGE_ALIGN) == 0)) {
            inrange = 1;
          } else {
            inrange = 0;
            dobegin = ii - 1;
          }
        }
        if (lookAhead == scanVal + 1) {
          if ((inrange == 0) && ((scanVal % RANGE_ALIGN) == 0)) {
            if (sliceCount != 0) {
              hiset.push_back(ListSegment(&indices_in[dobegin], sliceCount));
            }
            inrange = 1;
            dobegin = scanVal;
            sliceCount = 0;
          }
          ++sliceCount; /* account for scanVal */
        } else {
          if (inrange == 1) {
            /* we can tighten this up by schleping any trailing */
            /* sigletons off into the subsequent singleton */
            /* array.  We would then also need to recheck the */
            /* final length of the range to make sure it meets */
            /* our minimum length crietria.  If it doesnt, */
            /* we need to emit a random array instead of */
            /* a range array */
            ++sliceCount;
            hiset.push_back(RangeSegment(dobegin, dobegin + sliceCount));
            inrange = 0;
            sliceCount = 0;
            dobegin = ii;
          } else {
            ++sliceCount; /* account for scanVal */
          }
        }

        scanVal = lookAhead;
      }  // for (Index_type ii ...

      if (inrange != -1) {
        if (inrange) {
          ++sliceCount;
          hiset.push_back(RangeSegment(dobegin, dobegin + sliceCount));
        } else {
          ++sliceCount;
          hiset.push_back(ListSegment(&indices_in[dobegin], sliceCount));
        }
      } else if (scanVal != -1) {
        hiset.push_back(ListSegment(&scanVal, 1));
      }
    } else {  // !(docount < (length*RANGE_ALIGN-1))/RANGE_ALIGN)
      hiset.push_back(ListSegment(indices_in, length));
    }
  } else {  // else !(length > RANGE_MIN_LENGTH)
    hiset.push_back(ListSegment(indices_in, length));
  }
}

}  // closing brace for RAJA namespace
