
#include <stdio.h>
#include <cfloat>
#include <random>

#include "RAJA/RAJA.hxx"

#define TEST_VEC_LEN  1024 * 1024

using namespace RAJA;

//
// Global variables for counting tests executed/passed.
//
unsigned s_ntests_run = 0;
unsigned s_ntests_passed = 0;

int main(int argc, char *argv[])
{
   std::cout << "\n Begin RAJA GPU ReduceMax tests!!! " << std::endl;

   const int test_repeat = 10;

   double *dvalue ;

   cudaMallocManaged((void **)&dvalue,
      sizeof(double)*TEST_VEC_LEN,
      cudaMemAttachGlobal) ;

   std::random_device rd;
   std::mt19937 mt(rd());
   std::uniform_real_distribution<double> dist(-10, 10);
   std::uniform_real_distribution<double> dist2(0, TEST_VEC_LEN-1);

   //
   // Run TEST 1 only once
   //
   { // begin test 1

      for (int i=0; i<TEST_VEC_LEN; ++i) {
         dvalue[i] = -DBL_MAX ;
      }

      double BIG_MAX = 500.0;
      ReduceMax<cuda_reduce, double> dmax0(-DBL_MAX);
      ReduceMax<cuda_reduce, double> dmax1(-DBL_MAX);
      ReduceMax<cuda_reduce, double> dmax2(BIG_MAX);

      int loops = 16;
      double dcurrentMax = -DBL_MAX;

      for(int k=0; k < loops ; k++) {
         double droll = dist(mt) ; 
         int index = int(dist2(mt));
         dvalue[index] = droll;
         dcurrentMax = RAJA_MAX(dcurrentMax,dvalue[index]); 
         forall<cuda_exec>(0, TEST_VEC_LEN, [=] __device__ (int i) {
            dmax0.max(dvalue[i]) ;
            dmax1.max(2*dvalue[i]) ;
            dmax2.max(dvalue[i]) ;
         } ) ;

         s_ntests_run++;

         if ( double(dmax0) != dcurrentMax ||
              double(dmax1) != 2*dcurrentMax ||
              double(dmax2) != BIG_MAX ) {

            printf("\n TEST 1 FAILURE...\n");
            printf("(droll[%d] = %lf, \t check0 = %lf \t check1= %lf) \t dmax0 = %lf \t dmax1 = %lf\n",
            index,droll,dcurrentMax,2*dcurrentMax,double(dmax0),double(dmax1));
            printf("(check2 = %lf) \t dmax2 = %lf\n\n", BIG_MAX,double(dmax2));
         } else {
            s_ntests_passed++;
         }
      }

   }  // end test 1

////////////////////////////////////////////////////////////////////////////

   for (int tcount = 0; tcount < test_repeat; ++tcount) {

      printf("\ttcount = %d\n",tcount);

      { // begin test 2

      s_ntests_run++;

      for (int i=0; i<TEST_VEC_LEN; ++i) {
         dvalue[i] = -DBL_MAX ;
      }

      RangeSegment seg0(0, TEST_VEC_LEN/2);
      RangeSegment seg1(TEST_VEC_LEN/2 + 1, TEST_VEC_LEN);

      IndexSet iset;
      iset.push_back(seg0);
      iset.push_back(seg1);

      ReduceMax<cuda_reduce, double> dmax0(-DBL_MAX);
      ReduceMax<cuda_reduce, double> dmax1(-DBL_MAX);

      int index = int(dist2(mt));

      double droll = dist(mt) ;
      dvalue[index] = droll;

      double dcurrentMax = -DBL_MAX;
      dcurrentMax = RAJA_MAX(dcurrentMax,dvalue[index]);

      forall< IndexSet::ExecPolicy<seq_segit,cuda_exec> >(iset,
         [=] __device__ (int i) {
            dmax0.max(dvalue[i]) ;
            dmax1.max(2*dvalue[i]) ;
      } ) ;

      if ( double(dmax0) != dcurrentMax ||
           double(dmax1) != 2*dcurrentMax ) {
         printf("\n TEST 1 FAILURE...\n");
         printf("(droll[%d] = %lf, \t check0 = %lf \t check1= %lf) \t dmax0 = %lf \t dmax1 = %lf\n",
         index,droll,dcurrentMax,2*dcurrentMax,double(dmax0),double(dmax1));
      } else {
         s_ntests_passed++;
      }

      } // end test 2

////////////////////////////////////////////////////////////////////////////

      { // begin test 3

      s_ntests_run++;

      for (int i=0; i<TEST_VEC_LEN; ++i) {
         dvalue[i] = -DBL_MAX ;
      }

      RangeSegment seg0(1, 1230);
      RangeSegment seg1(1237, 3385);
      RangeSegment seg2(4860, 10110);
      RangeSegment seg3(20490, 32003);

      IndexSet iset;
      iset.push_back(seg0);
      iset.push_back(seg1);
      iset.push_back(seg2);
      iset.push_back(seg3);

      ReduceMax<cuda_reduce, double> dmax0(-DBL_MAX);
      ReduceMax<cuda_reduce, double> dmax1(-DBL_MAX);

      int index = 1297;

      double droll = dist(mt) ;
      dvalue[index] = droll;

      double dcurrentMax = -DBL_MAX;
      dcurrentMax = RAJA_MAX(dcurrentMax,dvalue[index]);

      forall< IndexSet::ExecPolicy<seq_segit,cuda_exec> >(iset,
         [=] __device__ (int i) {
            dmax0.max(dvalue[i]) ;
            dmax1.max(2*dvalue[i]) ;
      } ) ;

      if ( double(dmax0) != dcurrentMax ||
           double(dmax1) != 2*dcurrentMax ) {
         printf("\n TEST 1 FAILURE...\n");
         printf("(droll[%d] = %lf, \t check0 = %lf \t check1= %lf) \t dmax0 = %lf \t dmax1 = %lf\n",
         index,droll,dcurrentMax,2*dcurrentMax,double(dmax0),double(dmax1));
      } else {
         s_ntests_passed++;
      }

      } // end test 3

   } // end test repeat loop

   ///
   /// Print total number of tests passed/run.
   ///
   std::cout << "\n Tests Passed / Tests Run = "
             << s_ntests_passed << " / " << s_ntests_run << std::endl;

   cudaFree(dvalue); 

   std::cout << "\n RAJA GPU ReduceMax tests DONE!!! " << std::endl;
   
   return 0 ;
}

