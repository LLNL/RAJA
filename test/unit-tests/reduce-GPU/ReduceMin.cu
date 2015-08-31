
#include <stdio.h>
#include <cfloat>
#include <random>

#include "RAJA/RAJA.hxx"

#define TEST_VEC_LEN  1024 * 1024

using namespace RAJA;

int main(int argc, char *argv[])
{
   double BIG_MIN = -500.0;
   ReduceMin<cuda_reduce, double> dmin0(DBL_MAX);
   ReduceMin<cuda_reduce, double> dmin1(DBL_MAX);
   ReduceMin<cuda_reduce, double> dmin2(BIG_MIN);

   double *dvalue ;

   cudaMallocManaged((void **)&dvalue,
      sizeof(double)*TEST_VEC_LEN,
      cudaMemAttachGlobal) ;

   for (int i=0; i<TEST_VEC_LEN; ++i) {
      dvalue[i] = DBL_MAX ;
   }


   std::random_device rd;
   std::mt19937 mt(rd());
   std::uniform_real_distribution<double> dist(-10, 10);
   std::uniform_real_distribution<double> dist2(0, TEST_VEC_LEN-1);

   int loops = 16;
   double dcurrentMin = DBL_MAX;

   for(int k=0; k < loops ; k++) {
      double droll = dist(mt) ; 
      int index = int(dist2(mt));
      dvalue[index] = droll;
      dcurrentMin = RAJA_MIN(dcurrentMin,dvalue[index]); 
      forall<cuda_exec>(0, TEST_VEC_LEN, [=] __device__ (int i) {
         dmin0.min(dvalue[i]) ;
         dmin1.min(2*dvalue[i]) ;
         dmin2.min(dvalue[i]) ;
      } ) ;
      printf("(droll[%d] = %lf, \t check0 = %lf \t check1= %lf) \t dmin0 = %lf \t dmin1 = %lf\n",
         index,droll,dcurrentMin,2*dcurrentMin,double(dmin0),double(dmin1));
      printf("(check2 = %lf) \t dmin2 = %lf\n\n",
             BIG_MIN,double(dmin2));
   }
   
   return 0 ;
}

