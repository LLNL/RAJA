//
// Source file containing methods that build various hybrid index
// sets for testing...
//

#include "buildIndexSet.hxx"

#include<vector>

using namespace RAJA;

//
//  Initialize hybrid index set by adding segments as index set objects.
//
void buildIndexSet(IndexSet& hindex, IndexSetBuildMethod build_method)
{
   //
   // Create 8 hybrid segments and use to initialize hybrid in 
   // different ways below.
   // 

   // structured range: length 64, starting at 0
   RangeSegment seg0(0, 64);
   // unstructured list: length 7
   Index_type list_indx1[] = { 67, 69, 75, 77, 83, 85, 87 };
   // structured range: length 128, starting at 128
   RangeSegment seg2(128, 256);
   // unstructured list: length 3
   Index_type list_indx3[] = { 273, 284, 317 };
   // structured range: length 32, starting at 320
   RangeSegment seg4(320, 352);
   // unstructured list: length 8
   Index_type list_indx5[] = { 323, 334, 335, 401, 402, 403, 404, 407 };
   // structured range: length 64, starting at 480
   RangeSegment seg6(480, 544);
   // structured range: length 128, starting at 576
   RangeSegment seg7(576, 704);

   switch (build_method) {

      case AddSegments : {

         hindex.push_back(seg0);

         ListSegment seg1(list_indx1, 7);
         hindex.push_back(seg1);

         hindex.push_back(seg2);

         ListSegment seg3(list_indx3, 3);
         hindex.push_back(seg3);

         hindex.push_back(seg4);

         ListSegment seg5(list_indx5, 8);
         hindex.push_back(seg5);
   
         hindex.push_back(seg6);

         hindex.push_back(seg7);

         break;
      }

      case AddSegmentsReverse : {

         hindex.push_front(seg7);

         hindex.push_front(seg6);

         ListSegment seg5(list_indx5, 8);
         hindex.push_front(seg5);
  
         hindex.push_front(seg4);
  
         ListSegment seg3(list_indx3, 3);
         hindex.push_front(seg3);

         hindex.push_front(seg2);

         ListSegment seg1(list_indx1, 7);
         hindex.push_front(seg1);

         hindex.push_front(seg0);

         break;
      }

#if defined(RAJA_USE_STL)
      case AddSegmentsAsVectors : {

         hindex.push_back(seg0);

         std::vector<Index_type> vec1(list_indx1, list_indx1+7);
         ListSegment seg1(vec1);
         hindex.push_back(seg1);

         hindex.push_back(seg2);

         std::vector<Index_type> vec3(list_indx3, list_indx3+3);
         ListSegment seg3(vec3);
         hindex.push_back(seg3);

         hindex.push_back(seg4);

         std::vector<Index_type> vec5(list_indx5, list_indx5+8);
         ListSegment seg5(vec5);
         hindex.push_back(seg5);
   
         hindex.push_back(seg6);

         hindex.push_back(seg7);

         break;
      }

      case AddSegmentsAsVectorsReverse : {

         hindex.push_front(seg7);

         hindex.push_front(seg6);

         std::vector<Index_type> vec5(list_indx5, list_indx5+8);
         ListSegment seg5(vec5);
         hindex.push_front(seg5);

         hindex.push_front(seg4);

         std::vector<Index_type> vec3(list_indx3, list_indx3+3);
         ListSegment seg3(vec3);
         hindex.push_front(seg3);

         hindex.push_front(seg2);

         std::vector<Index_type> vec1(list_indx1, list_indx1+7);
         ListSegment seg1(vec1);
         hindex.push_front(seg1);

         hindex.push_front(seg0);

         break;
      }
#endif

      case AddSegmentsAsIndices : {

         hindex.push_back(seg0);

         hindex.push_back(ListSegment(list_indx1, 7));

         hindex.push_back(seg2);

         hindex.push_back(ListSegment(list_indx3, 3));

         hindex.push_back(seg4);

         hindex.push_back(ListSegment(list_indx5, 8));

         hindex.push_back(seg6);

         hindex.push_back(seg7);

         break;
      } 



      default : {

      }

   }  // switch (build_method)

}
