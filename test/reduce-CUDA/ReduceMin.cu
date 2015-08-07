
#include <stdio.h>
#include <cfloat>
#include <random>

#include "RAJA/RAJA.hxx"

#define TEST_VEC_LEN  1024 * 1024

using namespace RAJA;

int main(int argc, char *argv[])
{
   double BIGMIN = -500.0;
   ReduceMin<cuda_reduce, double> dmin0(50.0);
   ReduceMin<cuda_reduce, int>    dmin1(50);
   ReduceMin<cuda_reduce, double> dmin2(BIGMIN);
   ReduceMin<cuda_reduce, int>    dmin3(BIGMIN);

   double *dvalue ;
   int *ivalue ;

   cudaMallocManaged((void **)&dvalue,
      sizeof(double)*TEST_VEC_LEN,
      cudaMemAttachGlobal) ;

   cudaMallocManaged((void **)&ivalue,
      sizeof(int)*TEST_VEC_LEN,
      cudaMemAttachGlobal) ;

   for (int i=0; i<TEST_VEC_LEN; ++i) {
      dvalue[i] = DBL_MAX ;
      ivalue[i] = INT_MAX ;
   }


   std::random_device rd;
   std::mt19937 mt(rd());
   std::uniform_real_distribution<double> dist(-10, 10);
   std::uniform_real_distribution<double> dist2(0, TEST_VEC_LEN-1);

// int loops = 16;
   int loops = 1;
   double dcurrentMin = DBL_MAX;
   int icurrentMin = INT_MAX;

   for(int k=0; k < loops ; k++) {
#if 0 // RDH
      double droll = dist(mt) ; 
#else
      double droll = -100.0;
      int iroll = -100;
#endif
      int index = int(dist2(mt));
      dvalue[index] = droll;
      ivalue[index] = iroll;
      dcurrentMin = RAJA_MIN(dcurrentMin,dvalue[index]); 
      icurrentMin = RAJA_MIN(icurrentMin,ivalue[index]); 
      forall<cuda_exec>(0, TEST_VEC_LEN, [=] __device__ (int i) {
         dmin0.min(dvalue[i]) ;
         dmin1.min(ivalue[i]) ;
         dmin2.min(2*dvalue[i]) ;
         dmin3.min(2*ivalue[i]) ;
      } ) ;
      printf("(droll[%d] = %lf, \t check0 = %lf \t check2 = %lf) \t dmin0 = %lf \t dmin2 = %lf\n",index,droll,
         (double)dcurrentMin, BIGMIN,double(dmin0),double(dmin2));
      printf("\n(iroll[%d] = %d, \t check1 = %d \t check3 = %d) \t dmin1 = %d \t dmin3 = %d\n",index,iroll,
         (int)icurrentMin, (int)BIGMIN,int(dmin1),int(dmin3));
   }
   
   return 0 ;
}

