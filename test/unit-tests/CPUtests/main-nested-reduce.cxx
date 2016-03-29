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
   RAJA::ReduceSum<RAJA::omp_reduce, double> sumA(0.0) ;
   RAJA::ReduceMin<RAJA::omp_reduce, double> minA(10000.0) ;
   RAJA::ReduceMax<RAJA::omp_reduce, double> maxA(0.0) ;

   RAJA::forall<RAJA::omp_parallel_for_exec>(0, 10, [=] (int y) {
      RAJA::forall<RAJA::seq_exec>(0, 10, [=] (int x) {
         sumA += double(y*10 + x + 1) ;
         minA.min( double(y*10 + x + 1) ) ;
         maxA.max( double(y*10 + x + 1) ) ;
      } ) ;
   } ) ;
   printf("sum = %f, checksum = %f, min = %f, max = %f\n",
          double(sumA), 100.0*101.0 / 2.0, double(minA), double(maxA) ) ;

   RAJA::ReduceSum<RAJA::omp_reduce, double> sumB(0.0) ;
   RAJA::ReduceMin<RAJA::omp_reduce, double> minB(10000.0) ;
   RAJA::ReduceMax<RAJA::omp_reduce, double> maxB(0.0) ;

   RAJA::forall<RAJA::seq_exec>(0, 10, [=] (int y) {
      RAJA::forall<RAJA::omp_parallel_for_exec>(0, 10, [=] (int x) {
         sumB += double(y*10 + x + 1) ;
         minB.min( double(y*10 + x + 1) ) ;
         maxB.max( double(y*10 + x + 1) ) ;
      } ) ;
   } ) ;
   printf("sum = %f, checksum = %f, min = %f, max = %f\n",
          double(sumB), 100.0*101.0 / 2.0, double(minB), double(maxB) ) ;


   return 0 ;
}
