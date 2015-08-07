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
void buildIndexSet(IndexSet& hindex)
{
   //
   // Create 8 index set segments and add to index set.
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

   hindex.push_back(seg0);
   hindex.push_back(seg1);
   hindex.push_back(seg2);
   hindex.push_back(seg3);
   hindex.push_back(seg4);
   hindex.push_back(seg5);
   hindex.push_back(seg6);
   hindex.push_back(seg7);

}
