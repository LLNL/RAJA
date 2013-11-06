//
// Source file containing methods that build various hybrid index
// sets for testing...
//

#include "buildHybrid.hxx"

#include<vector>

using namespace RAJA;

//
//  Create hybrid index set by creating segments and adding to hybrid
//
HybridISet* buildHybrid_addSegments(bool use_vector)
{
   HybridISet* hindex = new HybridISet();

   // structured range: length 64, starting at 0
   RangeISet index0(0, 64);
   hindex->addISet(index0);

   // unstructured list: length 7
   Index_type unst_indx1[] = { 67, 69, 75, 77, 83, 85, 87 };
   if (use_vector) {
      std::vector<Index_type> vec(unst_indx1, unst_indx1+7);
      UnstructuredISet index(vec);
      hindex->addISet(index);
   } else {
      UnstructuredISet index(unst_indx1, 7);
      hindex->addISet(index);
   }

   // structured range: length 128, starting at 128
   RangeISet index2(128, 256);
   hindex->addISet(index2);

   // unstructured list: length 3
   Index_type unst_indx3[] = { 273, 284, 317 };
   if (use_vector) {
      std::vector<Index_type> vec(unst_indx3, unst_indx3+3);
      UnstructuredISet index(vec);
      hindex->addISet(index);
   } else {
      UnstructuredISet index(unst_indx3, 3);
      hindex->addISet(index);
   }

   // structured range: length 32, starting at 320
   RangeISet index4(320, 352);
   hindex->addISet(index4);

   // unstructured list: length 8
   Index_type unst_indx5[] = { 323, 334, 335, 401, 402, 403, 404, 407 };
   if (use_vector) {
      std::vector<Index_type> vec(unst_indx5, unst_indx5+8);
      UnstructuredISet index(vec);
      hindex->addISet(index);
   } else {
      UnstructuredISet index(unst_indx5, 8);
      hindex->addISet(index);
   }

   // structured range: length 64, starting at 480
   RangeISet index6(480, 544);
   hindex->addISet(index6);

   // structured range: length 128, starting at 576
   RangeISet index7(576, 704);
   hindex->addISet(index7);

   return hindex;
}


//
//  Create hybrid index set by adding indices for parts 
//
HybridISet* buildHybrid_addIndices()
{
   HybridISet* hindex = new HybridISet();

   // structured range: length 64, starting at 0
   hindex->addRangeIndices(0, 64);

   // unstructured list: length 7
   Index_type unst_indx1[] = { 67, 69, 75, 77, 83, 85, 87 };
   hindex->addUnstructuredIndices(unst_indx1, 7);

   // structured range: length 128, starting at 128
   hindex->addRangeIndices(128, 256);

   // unstructured list: length 3
   Index_type unst_indx3[] = { 273, 284, 317 };
   hindex->addUnstructuredIndices(unst_indx3, 3);

   // structured range: length 32, starting at 320
   hindex->addRangeIndices(320, 352);

   // unstructured list: length 8
   Index_type unst_indx5[] = { 323, 334, 335, 401, 402, 403, 404, 407 };
   hindex->addUnstructuredIndices(unst_indx5, 8);

   // structured range: length 64, starting at 480
   hindex->addRangeIndices(480, 544);

   // structured range: length 128, starting at 576
   hindex->addRangeIndices(576, 704);

   return hindex;
}

