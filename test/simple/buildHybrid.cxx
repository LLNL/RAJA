//
// Source file containing methods that build various hybrid index
// sets for testing...
//

#include "buildHybrid.hxx"

#include<vector>

using namespace RAJA;

//
//  Initialize hybrid index set by adding segments as index set objects.
//
void buildHybrid(HybridISet& hindex, HybridBuildMethod build_method)
{
   //
   // Create 8 hybrid segments and use to initialize hybrid in 
   // different ways below.
   // 

   // structured range: length 64, starting at 0
   RangeISet index0(0, 64);
   // unstructured list: length 7
   Index_type unst_indx1[] = { 67, 69, 75, 77, 83, 85, 87 };
   // structured range: length 128, starting at 128
   RangeISet index2(128, 256);
   // unstructured list: length 3
   Index_type unst_indx3[] = { 273, 284, 317 };
   // structured range: length 32, starting at 320
   RangeISet index4(320, 352);
   // unstructured list: length 8
   Index_type unst_indx5[] = { 323, 334, 335, 401, 402, 403, 404, 407 };
   // structured range: length 64, starting at 480
   RangeISet index6(480, 544);
   // structured range: length 128, starting at 576
   RangeISet index7(576, 704);

   switch (build_method) {

      case AddSegments : {

         hindex.addISet(index0);

         UnstructuredISet index1(unst_indx1, 7);
         hindex.addISet(index1);

         hindex.addISet(index2);

         UnstructuredISet index3(unst_indx3, 3);
         hindex.addISet(index3);

         hindex.addISet(index4);

         UnstructuredISet index5(unst_indx5, 8);
         hindex.addISet(index5);
   
         hindex.addISet(index6);

         hindex.addISet(index7);

         break;
      }

#if defined(RAJA_USE_STL)
      case AddSegmentsAsVectors : {

         hindex.addISet(index0);

         std::vector<Index_type> vec1(unst_indx1, unst_indx1+7);
         UnstructuredISet index1(vec1);
         hindex.addISet(index1);

         hindex.addISet(index2);

         std::vector<Index_type> vec3(unst_indx3, unst_indx3+3);
         UnstructuredISet index3(vec3);
         hindex.addISet(index3);

         hindex.addISet(index4);

         std::vector<Index_type> vec5(unst_indx5, unst_indx5+8);
         UnstructuredISet index5(vec5);
         hindex.addISet(index5);
   
         hindex.addISet(index6);

         hindex.addISet(index7);

         break;
      }
#endif

      case AddSegmentsAsIndices : {

         hindex.addRangeIndices(index0.getBegin(), index0.getEnd());

         hindex.addUnstructuredIndices(unst_indx1, 7);

         hindex.addRangeIndices(index2.getBegin(), index2.getEnd());

         hindex.addUnstructuredIndices(unst_indx3, 3);

         hindex.addRangeIndices(index4.getBegin(), index4.getEnd());

         hindex.addUnstructuredIndices(unst_indx5, 8);

         hindex.addRangeIndices(index6.getBegin(), index6.getEnd());

         hindex.addRangeIndices(index7.getBegin(), index7.getEnd());

         break;
      } 



      default : {

      }

   }  // switch (build_method)

}
