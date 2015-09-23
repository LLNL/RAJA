//
// Source file containing methods that build various hybrid index
// sets for testing...
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
#if 1

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
#else
   
   //
   // Create 8 hybrid segments and use to initialize hybrid in 
   // different ways below.
   // 

   // structured range: length 64, starting at 0
   RangeSegment seg0(0, 64);

   // unstructured list: length 7
   Index_type list_indx1[] = { 67, 69, 75, 77, 83, 85, 87 };
   ListSegment seg1(list_indx1, 7);

   // structured range: length 128, starting at 128
   RangeSegment seg2(128, 256);

   // unstructured list: length 3
   Index_type list_indx3[] = { 273, 284, 317 };
   ListSegment seg3(list_indx3, 3);

   // structured range: length 32, starting at 320
   RangeSegment seg4(320, 352);

   // unstructured list: length 8
   Index_type list_indx5[] = { 323, 334, 335, 401, 402, 403, 404, 407 };
   ListSegment seg5(list_indx5, 8);

   // structured range: length 64, starting at 480
   RangeSegment seg6(480, 544);

   // structured range: length 128, starting at 576
   RangeSegment seg7(576, 704);

   switch (build_method) {

      case AddSegments : {

         hindex[build_method].push_back(seg0);

         hindex[build_method].push_back(seg1);

         hindex[build_method].push_back(seg2);

         hindex[build_method].push_back(seg3);

         hindex[build_method].push_back(seg4);

         hindex[build_method].push_back(seg5);
   
         hindex[build_method].push_back(seg6);

         hindex[build_method].push_back(seg7);

         break;
      }

      case AddSegmentsReverse : {

         hindex[build_method].push_front(seg7);

         hindex[build_method].push_front(seg6);

         hindex[build_method].push_front(seg5);
  
         hindex[build_method].push_front(seg4);
  
         hindex[build_method].push_front(seg3);

         hindex[build_method].push_front(seg2);

         hindex[build_method].push_front(seg1);

         hindex[build_method].push_front(seg0);

         break;
      }

#if defined(RAJA_USE_STL)
      case AddSegmentsAsVectors : {

         hindex[build_method].push_back(seg0);

         std::vector<Index_type> vec1(list_indx1, list_indx1+7);
         ListSegment vseg1(vec1);
         hindex[build_method].push_back(vseg1);

         hindex[build_method].push_back(seg2);

         std::vector<Index_type> vec3(list_indx3, list_indx3+3);
         ListSegment vseg3(vec3);
         hindex[build_method].push_back(vseg3);

         hindex[build_method].push_back(seg4);

         std::vector<Index_type> vec5(list_indx5, list_indx5+8);
         ListSegment vseg5(vec5);
         hindex[build_method].push_back(vseg5);
   
         hindex[build_method].push_back(seg6);

         hindex[build_method].push_back(seg7);

         break;
      }

      case AddSegmentsAsVectorsReverse : {

         hindex[build_method].push_front(seg7);

         hindex[build_method].push_front(seg6);

         std::vector<Index_type> vec5(list_indx5, list_indx5+8);
         ListSegment vseg5(vec5);
         hindex[build_method].push_front(vseg5);

         hindex[build_method].push_front(seg4);

         std::vector<Index_type> vec3(list_indx3, list_indx3+3);
         ListSegment vseg3(vec3);
         hindex[build_method].push_front(vseg3);

         hindex[build_method].push_front(seg2);

         std::vector<Index_type> vec1(list_indx1, list_indx1+7);
         ListSegment vseg1(vec1);
         hindex[build_method].push_front(vseg1);

         hindex[build_method].push_front(seg0);

         break;
      }
#endif

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
#endif

  return last_indx;

}
