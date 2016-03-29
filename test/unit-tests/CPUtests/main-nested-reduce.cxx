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

   RAJA::forall<RAJA::omp_parallel_for_exec>(0, 10, [=] (int y) {
      RAJA::forall<RAJA::seq_exec>(0, 10, [=] (int x) {
         sumA += double(y*10 + x + 1) ;
      } ) ;
   } ) ;
   printf("sum = %f, check = %f\n", double(sumA), 100.0*101.0 / 2.0) ;

   RAJA::ReduceSum<RAJA::omp_reduce, double> sumB(0.0) ;

   RAJA::forall<RAJA::seq_exec>(0, 10, [=] (int y) {
      RAJA::forall<RAJA::omp_parallel_for_exec>(0, 10, [=] (int x) {
         sumB += double(y*10 + x + 1) ;
      } ) ;
   } ) ;
   printf("sum = %f, check = %f\n", double(sumB), 100.0*101.0 / 2.0) ;


   return 0 ;
}
