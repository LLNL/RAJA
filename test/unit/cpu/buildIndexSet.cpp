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
Index_type buildIndexSet(IndexSet* hindex, IndexSetBuildMethod build_method)
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
  RAJAVec<BaseSegment*> segments;
  for (int i = 0; i < seg_chunk_size; ++i) {
    Index_type rbeg;
    Index_type rend;
    Index_type lseg_len = lindices.size();
    RAJAVec<Index_type> lseg(lseg_len);

    // Create Range segment
    rbeg = last_indx + 2;
    rend = rbeg + 32;
    segments.push_back(new RangeSegment(rbeg, rend));
    last_indx = rend;

    // Create List segment
    for (Index_type i = 0; i < lseg_len; ++i) {
      lseg[i] = lindices[i] + last_indx;
    }
    segments.push_back(new ListSegment(&lseg[0], lseg_len));
    last_indx = lseg[lseg_len - 1];

    // Create Range segment
    rbeg = last_indx + 16;
    rend = rbeg + 128;
    segments.push_back(new RangeSegment(rbeg, rend));
    last_indx = rend;

    // Create Range segment
    rbeg = last_indx + 4;
    rend = rbeg + 256;
    segments.push_back(new RangeSegment(rbeg, rend));
    last_indx = rend;

    // Create List segment
    for (Index_type i = 0; i < lseg_len; ++i) {
      lseg[i] = lindices[i] + last_indx + 5;
    }
    segments.push_back(new ListSegment(&lseg[0], lseg_len));
    last_indx = lseg[lseg_len - 1];
  }

  //
  // Generate IndexSet from segments using specified build method.
  //
  switch (build_method) {
    case AddSegments: {
      for (size_t i = 0; i < segments.size(); ++i) {
        hindex[build_method].push_back(*segments[i]);
      }

      break;
    }

    case AddSegmentsReverse: {
      int last_i = static_cast<int>(segments.size() - 1);
      for (int i = last_i; i >= 0; --i) {
        hindex[build_method].push_front(*segments[i]);
      }

      break;
    }

    case AddSegmentsNoCopy: {
      IndexSet& iset_master = hindex[0];

      for (size_t i = 0; i < iset_master.getNumSegments(); ++i) {
        hindex[build_method].push_back_nocopy(iset_master.getSegment(i));
      }

      break;
    }

    case AddSegmentsNoCopyReverse: {
      IndexSet& iset_master = hindex[0];

      int last_i = static_cast<int>(iset_master.getNumSegments() - 1);
      for (int i = last_i; i >= 0; --i) {
        hindex[build_method].push_front_nocopy(iset_master.getSegment(i));
      }

      break;
    }

    case MakeViewRange: {
      IndexSet& iset_master = hindex[0];
      size_t num_segs = iset_master.getNumSegments();

      IndexSet* iset_view = iset_master.createView(0, num_segs);

      for (size_t i = 0; i < iset_view->getNumSegments(); ++i) {
        hindex[build_method].push_back_nocopy(iset_view->getSegment(i));
      }

      break;
    }

    case MakeViewArray: {
      IndexSet& iset_master = hindex[0];
      size_t num_segs = iset_master.getNumSegments();
      int* segIds = new int[num_segs];
      for (size_t i = 0; i < num_segs; ++i) {
        segIds[i] = i;
      }

      IndexSet* iset_view = iset_master.createView(segIds, num_segs);

      for (size_t i = 0; i < iset_view->getNumSegments(); ++i) {
        hindex[build_method].push_back_nocopy(iset_view->getSegment(i));
      }

      delete[] segIds;

      break;
    }

#if defined(RAJA_USE_STL)
    case MakeViewVector: {
      IndexSet& iset_master = hindex[0];
      size_t num_segs = iset_master.getNumSegments();
      std::vector<int> segIds(num_segs);
      for (int i = 0; i < num_segs; ++i) {
        segIds[i] = i;
      }

      IndexSet* iset_view = iset_master.createView(segIds);

      for (size_t i = 0; i < iset_view->getNumSegments(); ++i) {
        hindex[build_method].push_back_nocopy(iset_view->getSegment(i));
      }

      break;
    }
#endif

    default: {
    }

  }  // switch (build_method)

  for (size_t i = 0; i < segments.size(); ++i) {
    delete segments[i];
  }

  return last_indx;
}
