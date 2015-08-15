
#include <stdio.h>
#include <cfloat>
#include <random>

#include "RAJA/RAJA.hxx"

#define TEST_VEC_LEN  1024 * 1024

using namespace RAJA;

int main(int argc, char *argv[])
{
   double tinit = 5.0;

   ReduceSum<cuda_reduce, double> dsum0(0.0);
   ReduceSum<cuda_reduce, double> dsum1(tinit * 1.0);
   ReduceSum<cuda_reduce, double> dsum2(0.0);
   ReduceSum<cuda_reduce, double> dsum3(tinit * 3.0);
   ReduceSum<cuda_reduce, double> dsum4(0.0);
   ReduceSum<cuda_reduce, double> dsum5(tinit * 5.0);
   ReduceSum<cuda_reduce, double> dsum6(0.0);
   ReduceSum<cuda_reduce, double> dsum7(tinit * 7.0);

   double *value ;

   cudaMallocManaged((void **)&value,
      sizeof(double)*TEST_VEC_LEN,
      cudaMemAttachGlobal) ;

   for (int i=0; i<TEST_VEC_LEN; ++i) {
      value[i] = 1 ;
   }

   int loops = 2;

   for(int k=0; k < loops ; k++) {
      forall<cuda_exec>(0, TEST_VEC_LEN, [=] __device__ (int i) {
         dsum0 += value[i] ;
         dsum1 += value[i] * 2;
         dsum2 += value[i] * 3;
         dsum3 += value[i] * 4;
         dsum4 += value[i] * 5;
         dsum5 += value[i] * 6;
         dsum6 += value[i] * 7;
         dsum7 += value[i] * 8;
      } ) ;

      if ( k < loops-1) { 
         printf("\n\n loop count = %d....\n", k);
         int kloop = k+1; 
         printf("(check = %lf) dsum0 = %lf\n",(double)(TEST_VEC_LEN)*1*kloop, double(dsum0));
         printf("(check = %lf) dsum1 = %lf\n",(double)(TEST_VEC_LEN)*2*kloop+tinit*1.0, double(dsum1));
         printf("(check = %lf) dsum2 = %lf\n",(double)(TEST_VEC_LEN)*3*kloop, double(dsum2));
         printf("(check = %lf) dsum3 = %lf\n",(double)(TEST_VEC_LEN)*4*kloop+tinit*3.0, double(dsum3));
         printf("(check = %lf) dsum4 = %lf\n",(double)(TEST_VEC_LEN)*5*kloop, double(dsum4));
         printf("(check = %lf) dsum5 = %lf\n",(double)(TEST_VEC_LEN)*6*kloop+tinit*5.0, double(dsum5));
         printf("(check = %lf) dsum6 = %lf\n",(double)(TEST_VEC_LEN)*7*kloop, double(dsum6));
         printf("(check = %lf) dsum7 = %lf\n",(double)(TEST_VEC_LEN)*8*kloop+tinit*7.0, double(dsum7));
      }
   }

#if 0
   CudaReductionBlockDataType* blockdata = getCudaReductionMemBlock(); 
   for(int k=0;k<8;k++) {
      int blockoffset = getCudaReductionMemBlockOffset(k);
      printf("blockSum[%d]= %lf\n",blockoffset,blockdata[blockoffset]);
   }
#endif

   printf("\n\nFINAL RESULTS....\n");
   printf("(check = %lf) dsum0 = %lf\n",(double)(TEST_VEC_LEN)*1*loops, double(dsum0));
   printf("(check = %lf) dsum1 = %lf\n",(double)(TEST_VEC_LEN)*2*loops+tinit*1.0, double(dsum1));
   printf("(check = %lf) dsum2 = %lf\n",(double)(TEST_VEC_LEN)*3*loops, double(dsum2));
   printf("(check = %lf) dsum3 = %lf\n",(double)(TEST_VEC_LEN)*4*loops+tinit*3.0, double(dsum3));
   printf("(check = %lf) dsum4 = %lf\n",(double)(TEST_VEC_LEN)*5*loops, double(dsum4));
   printf("(check = %lf) dsum5 = %lf\n",(double)(TEST_VEC_LEN)*6*loops+tinit*5.0, double(dsum5));
   printf("(check = %lf) dsum6 = %lf\n",(double)(TEST_VEC_LEN)*7*loops, double(dsum6));
   printf("(check = %lf) dsum7 = %lf\n",(double)(TEST_VEC_LEN)*8*loops+tinit*7.0, double(dsum7));

   return 0 ;
}
