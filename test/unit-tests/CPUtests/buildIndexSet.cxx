/*
 * Copyright (c) 2016, Lawrence Livermore National Security, LLC.
 *
 * Produced at the Lawrence Livermore National Laboratory.
 *
 * All rights reserved.
 *
 * For release details and restrictions, please see raja/README-license.txt
 */

//
// Source file containing methods that build various index sets for testing...
//

#include "buildIndexSet.hxx"

#include<vector>

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
      lindices.push_back(istart+1);
      lindices.push_back(istart+4);
      lindices.push_back(istart+5);
      lindices.push_back(istart+9);
      lindices.push_back(istart+10);
      lindices.push_back(istart+11);
      lindices.push_back(istart+12);
      lindices.push_back(istart+14);
      lindices.push_back(istart+15);
      lindices.push_back(istart+21);
      lindices.push_back(istart+27);
      lindices.push_back(istart+28);
      lindx_end = istart+28;
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
      segments.push_back( new RangeSegment(rbeg, rend) );
      last_indx = rend;

      // Create List segment
      for (Index_type i = 0; i < lseg_len; ++i) {  
         lseg[i] = lindices[i] + last_indx;
      }
      segments.push_back( new ListSegment(&lseg[0], lseg_len) ) ;
      last_indx = lseg[lseg_len-1];

      // Create Range segment
      rbeg = last_indx + 16;
      rend = rbeg + 128;
      segments.push_back( new RangeSegment(rbeg, rend) );
      last_indx = rend;

      // Create Range segment
      rbeg = last_indx + 4;
      rend = rbeg + 256;
      segments.push_back( new RangeSegment(rbeg, rend) );
      last_indx = rend;

      // Create List segment
      for (Index_type i = 0; i < lseg_len; ++i) {
         lseg[i] = lindices[i] + last_indx + 5;
      }
      segments.push_back( new ListSegment(&lseg[0], lseg_len) ) ;
      last_indx = lseg[lseg_len-1]; 
   }

   //
   // Generate IndexSet from segments using specified build method.
   //
   switch (build_method) {

      case AddSegments : {

         for (int i = 0; i < segments.size(); ++i) {
            hindex[build_method].push_back(*segments[i]);
         }

         break;
      }

      case AddSegmentsReverse : {

         for (int i = segments.size()-1; i >= 0; --i) {
            hindex[build_method].push_front(*segments[i]);
         }

         break;
      }

      case AddSegmentsNoCopy : {

         IndexSet& iset_master = hindex[0];

         for ( int i = 0; i < iset_master.getNumSegments(); ++i ) {
            hindex[build_method].push_back_nocopy( iset_master.getSegment(i) );
         }

         break;
      }

      case AddSegmentsNoCopyReverse : {

         IndexSet& iset_master = hindex[0];

         for ( int i = iset_master.getNumSegments() - 1; i >= 0 ; --i ) {
            hindex[build_method].push_front_nocopy( iset_master.getSegment(i) );
         }

         break;
      }

      case MakeViewRange : {

         IndexSet& iset_master = hindex[0];
         int num_segs = iset_master.getNumSegments();

         IndexSet* iset_view = iset_master.createView(0, num_segs);

         for ( int i = 0; i < iset_view->getNumSegments(); ++i ) {
            hindex[build_method].push_back_nocopy( iset_view->getSegment(i) );
         }

         break;
      }

      case MakeViewArray : {

         IndexSet& iset_master = hindex[0];
         int num_segs = iset_master.getNumSegments();
         int* segIds = new int[num_segs];
         for ( int i = 0; i < num_segs; ++i ) { segIds[i] = i; }

         IndexSet* iset_view = iset_master.createView(segIds, num_segs);

         for ( int i = 0; i < iset_view->getNumSegments(); ++i ) {
            hindex[build_method].push_back_nocopy( iset_view->getSegment(i) );
         }

         delete [] segIds;

         break;
      }

#if defined(RAJA_USE_STL)
      case MakeViewVector : {

         IndexSet& iset_master = hindex[0];
         int num_segs = iset_master.getNumSegments();
         std::vector<int> segIds(num_segs);
         for ( int i = 0; i < num_segs; ++i ) { segIds[i] = i; }

         IndexSet* iset_view = iset_master.createView(segIds);

         for ( int i = 0; i < iset_view->getNumSegments(); ++i ) { 
            hindex[build_method].push_back_nocopy( iset_view->getSegment(i) );
         }

         break;
      }
#endif

      default : {

      }

   }  // switch (build_method)

   for (int i = 0; i < segments.size(); ++i) {
      delete segments[i];
   } 

  return last_indx;
}
