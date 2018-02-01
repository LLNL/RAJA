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
Index_type buildIndexSet(UnitIndexSet* hindex,
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
  UnitIndexSet iset_master;

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
  cout << "\n\nUnitIndexSet( master ) " << endl;
  iset_master.print(cout);
#endif

  //
  // Generate UnitIndexSet from segments using specified build method.
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
      UnitIndexSet& iset_master = hindex[0];
      for (int i = iset_master.getNumSegments()-1; i >= 0; --i) {
        iset_master.segment_push_into(i,
                                      hindex[build_method],
                                      PUSH_FRONT,
                                      PUSH_COPY);
      }

      break;
    }

    case AddSegmentsNoCopy: {
      UnitIndexSet& iset_master = hindex[0];
      for (size_t i = 0; i < iset_master.getNumSegments(); ++i) {
        iset_master.segment_push_into(i,
                                      hindex[build_method],
                                      PUSH_BACK,
                                      PUSH_NOCOPY);
      }

      break;
    }

    case AddSegmentsNoCopyReverse: {
      UnitIndexSet& iset_master = hindex[0];
      for ( int i = iset_master.getNumSegments() - 1; i >= 0 ; --i ) {
        iset_master.segment_push_into(i,
                                      hindex[build_method],
                                      PUSH_FRONT,
                                      PUSH_NOCOPY);
      }

      break;
    }

    case MakeSliceRange: {
      UnitIndexSet& iset_master = hindex[0];
      size_t num_segs = iset_master.getNumSegments();
      UnitIndexSet* iset_slice
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
      UnitIndexSet& iset_master = hindex[0];
      size_t num_segs = iset_master.getNumSegments();
      int* segIds = new int[num_segs];

      for (size_t i = 0; i < num_segs; ++i) {
        segIds[i] = i;
      }

      UnitIndexSet* iset_slice
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
      UnitIndexSet& iset_master = hindex[0];
      size_t num_segs = iset_master.getNumSegments();
      std::vector<int> segIds(num_segs);
      for (int i = 0; i < num_segs; ++i) {
        segIds[i] = i;
      }

      UnitIndexSet* iset_slice
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
