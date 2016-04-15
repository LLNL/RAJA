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
// Source file containing test for nested reductions...
//


#include "RAJA/RAJA.hxx"
#include <stdio.h>

int main(int argc, char *argv[])
{
   int run = 0 ;
   int passed = 0 ;

   int const xExtent = 10 ;
   int const yExtent = 10 ;
   int const area = xExtent*yExtent ;

   RAJA::ReduceSum<RAJA::omp_reduce, double> sumA(0.0) ;
   RAJA::ReduceMin<RAJA::omp_reduce, double> minA(10000.0) ;
   RAJA::ReduceMax<RAJA::omp_reduce, double> maxA(0.0) ;

   RAJA::forall<RAJA::omp_parallel_for_exec>(0, yExtent, [=] (int y) {
      RAJA::forall<RAJA::seq_exec>(0, xExtent, [=] (int x) {
         sumA += double(y*xExtent + x + 1) ;
         minA.min( double(y*xExtent + x + 1) ) ;
         maxA.max( double(y*xExtent + x + 1) ) ;
      } ) ;
   } ) ;

   printf("sum(%6.1f) = %6.1f, min(1.0) = %3.1f, max(%5.1f) = %5.1f\n",
          (area*(area+1) / 2.0), double(sumA),
          double(minA), double(area), double(maxA) ) ;

   run += 3 ;
   if (double(sumA) == (area*(area+1) / 2.0)) {
      ++passed ;
   }
   if (double(minA) == 1.0) {
      ++passed ;
   }
   if (double(maxA) == double(area)) {
      ++passed ;
   }

   RAJA::ReduceSum<RAJA::omp_reduce, double> sumB(0.0) ;
   RAJA::ReduceMin<RAJA::omp_reduce, double> minB(10000.0) ;
   RAJA::ReduceMax<RAJA::omp_reduce, double> maxB(0.0) ;

   RAJA::forall<RAJA::seq_exec>(0, yExtent, [=] (int y) {
      RAJA::forall<RAJA::omp_parallel_for_exec>(0, xExtent, [=] (int x) {
         sumB += double(y*xExtent + x + 1) ;
         minB.min( double(y*xExtent + x + 1) ) ;
         maxB.max( double(y*xExtent + x + 1) ) ;
      } ) ;
   } ) ;

   printf("sum(%6.1f) = %6.1f, min(1.0) = %3.1f, max(%5.1f) = %5.1f\n",
          (area*(area+1) / 2.0), double(sumB),
          double(minB),
          double(area), double(maxB) ) ;

   run += 3 ;
   if (double(sumB) == (area*(area+1) / 2.0)) {
      ++passed ;
   }
   if (double(minB) == 1.0) {
      ++passed ;
   }
   if (double(maxB) == double(area)) {
      ++passed ;
   }

   printf("\n All Tests : # passed / # run = %d / %d\n\n DONE!!!\n",
          passed, run) ;

   return 0 ;
}
