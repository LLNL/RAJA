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
// For additional details, please also read RAJA/README.
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

///
/// Source file containing methods that build various index sets for testing...
///

#include "buildIndexSet.hpp"

#include <vector>

using namespace RAJA;
using namespace std;

//
//  Initialize hybrid index set by adding segments as index set objects.
//
Index_type buildIndexSet(RAJA::IndexSet<RAJA::RangeSegment,
                                        RAJA::ListSegment,
                                        RAJA::RangeStrideSegment>* hindex,
                         IndexSetBuildMethod build_method)
{
  //
  // Record last index in index set for return.
  //
  Index_type last_indx = 0;

  //
  // Build vector of integers for creating ListSegments.
  //
  Index_type lindx_end = 0;
  RAJAVec<Index_type> lindices;
  for (Index_type i = 0; i < 5; ++i) {
    Index_type istart = lindx_end;
    lindices.push_back(istart + 1);
    lindices.push_back(istart + 4);
    lindices.push_back(istart + 5);
    lindices.push_back(istart + 9);
    lindices.push_back(istart + 10);
    lindices.push_back(istart + 11);
    lindices.push_back(istart + 12);
    lindices.push_back(istart + 14);
    lindices.push_back(istart + 15);
    lindices.push_back(istart + 21);
    lindices.push_back(istart + 27);
    lindices.push_back(istart + 28);
    lindx_end = istart + 28;
  }

  //
  // Create a vector of interleaved Range and List segments.
  //

  const int seg_chunk_size = 5;
  IndexSet<RAJA::RangeSegment,
           RAJA::ListSegment,
           RAJA::RangeStrideSegment> iset_master;

  for (int i = 0; i < seg_chunk_size; ++i) {
    Index_type rbeg;
    Index_type rend;
    Index_type lseg_len = lindices.size();
    RAJAVec<Index_type> lseg(lseg_len);

    // Create Range segment
    rbeg = last_indx + 2;
    rend = rbeg + 32;
    iset_master.push_back(RangeSegment(rbeg, rend));
    last_indx = rend;

    // Create List segment
    for (Index_type i = 0; i < lseg_len; ++i) {
      lseg[i] = lindices[i] + last_indx;
    }
    iset_master.push_back(ListSegment(&lseg[0], lseg_len));
    last_indx = lseg[lseg_len - 1];

    // Create Range segment
    rbeg = last_indx + 16;
    rend = rbeg + 128;
    iset_master.push_back(RangeSegment(rbeg, rend));
    last_indx = rend;

    // Create Range segment
    rbeg = last_indx + 4;
    rend = rbeg + 256;
    iset_master.push_back(RangeSegment(rbeg, rend));
    last_indx = rend;

    // Create List segment
    for (Index_type i = 0; i < lseg_len; ++i) {
      lseg[i] = lindices[i] + last_indx + 5;
    }
    iset_master.push_back(ListSegment(&lseg[0], lseg_len));
    last_indx = lseg[lseg_len - 1];
  }

#if 0  // print index set for debugging
  cout << "\n\nIndexSet( master ) " << endl;
  iset_master.print(cout);
#endif

  //
  // Generate IndexSet from segments using specified build method.
  //
  switch (build_method) {

    // This is already being done above as iset_master
    case AddSegments: {
      // This is already being done above as iset_master
      for (size_t i = 0; i < iset_master.getNumSegments(); ++i) {
        iset_master.segment_push_into(i,
                                      hindex[build_method],
                                      PUSH_BACK,
                                      PUSH_COPY);
      }
      break;
    }

    case AddSegmentsReverse: {
      RAJA::IndexSet<RAJA::RangeSegment,
                     RAJA::ListSegment,
                     RAJA::RangeStrideSegment>& iset_master = hindex[0];
      for (int i = iset_master.getNumSegments()-1; i >= 0; --i) {
        iset_master.segment_push_into(i,
                                      hindex[build_method],
                                      PUSH_FRONT,
                                      PUSH_COPY);
      }

      break;
    }

    case AddSegmentsNoCopy: {
      RAJA::IndexSet<RAJA::RangeSegment,
                     RAJA::ListSegment,
                     RAJA::RangeStrideSegment>& iset_master = hindex[0];
      for (size_t i = 0; i < iset_master.getNumSegments(); ++i) {
        iset_master.segment_push_into(i,
                                      hindex[build_method],
                                      PUSH_BACK,
                                      PUSH_NOCOPY);
      }

      break;
    }

    case AddSegmentsNoCopyReverse: {
      RAJA::IndexSet<RAJA::RangeSegment,
                     RAJA::ListSegment,
                     RAJA::RangeStrideSegment>& iset_master = hindex[0];
      for ( int i = iset_master.getNumSegments() - 1; i >= 0 ; --i ) {
        iset_master.segment_push_into(i,
                                      hindex[build_method],
                                      PUSH_FRONT,
                                      PUSH_NOCOPY);
      }

      break;
    }

    case MakeSliceRange: {
      RAJA::IndexSet<RAJA::RangeSegment,
                     RAJA::ListSegment,
                     RAJA::RangeStrideSegment>& iset_master = hindex[0];
      size_t num_segs = iset_master.getNumSegments();
      RAJA::IndexSet<RAJA::RangeSegment,
                     RAJA::ListSegment,
                     RAJA::RangeStrideSegment>* iset_slice
        = iset_master.createSlice(0, num_segs);

      for (size_t i = 0; i < iset_slice->getNumSegments(); ++i) {
        iset_slice->segment_push_into(i,
                                      hindex[build_method],
                                      PUSH_BACK,
                                      PUSH_NOCOPY);
      }

      break;
    }

    case MakeSliceArray: {
      RAJA::IndexSet<RAJA::RangeSegment,
                     RAJA::ListSegment,
                     RAJA::RangeStrideSegment>& iset_master = hindex[0];
      size_t num_segs = iset_master.getNumSegments();
      int* segIds = new int[num_segs];

      for (size_t i = 0; i < num_segs; ++i) {
        segIds[i] = i;
      }

      RAJA::IndexSet<RAJA::RangeSegment,
                     RAJA::ListSegment,
                     RAJA::RangeStrideSegment>* iset_slice
        = iset_master.createSlice(segIds, num_segs);

      for (size_t i = 0; i < iset_slice->getNumSegments(); ++i) {
        iset_slice->segment_push_into(i,
                                      hindex[build_method],
                                      PUSH_BACK,
                                      PUSH_NOCOPY);
      }

      delete[] segIds;

      break;
    }

#if defined(RAJA_USE_STL)
    case MakeSliceVector: {
      RAJA::IndexSet<RAJA::RangeSegment,
                     RAJA::ListSegment,
                     RAJA::RangeStrideSegment>& iset_master = hindex[0];
      size_t num_segs = iset_master.getNumSegments();
      std::vector<int> segIds(num_segs);
      for (int i = 0; i < num_segs; ++i) {
        segIds[i] = i;
      }

      RAJA::IndexSet<RAJA::RangeSegment,
                     RAJA::ListSegment,
                     RAJA::RangeStrideSegment>* iset_slice
        = iset_master.createSlice(segIds);

      for (size_t i = 0; i < iset_slice->getNumSegments(); ++i) {
        iset_slice->segment_push_into(i,
                                      hindex[build_method],
                                      PUSH_BACK,
                                      PUSH_NOCOPY);
      }

      break;
    }
#endif

    default: {
    }

  }  // end switch (build_method)

  return last_indx;
}
