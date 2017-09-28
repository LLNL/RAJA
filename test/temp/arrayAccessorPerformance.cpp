/*
 * main.cpp
 *
 *  Created on: Jul 30, 2016
 *      Author: rrsettgast
 */


#include "arrayAccessorPerformance.hpp"


double MatrixMultiply_1D( integer_t const num_i,
                          integer_t const num_j,
                          integer_t const num_k,
                          integer_t const ITERATIONS,
                          double const * const  A,
                          double const * const  B,
                          double * const  C )
{
  uint64_t startTime = GetTimeMs64();
  for( integer_t iter = 0 ; iter < ITERATIONS ; ++iter )
  {
    for( integer_t i = 0 ; i < num_i ; ++i )
    {
      for( integer_t j = 0 ; j < num_j ; ++j )
      {
        for( integer_t k = 0 ; k < num_k ; ++k )
        {
          C[ i*num_j+j ] += A[ i*num_k+k ] * B[ k*num_j+j ] + 3.1415 * A[ i*num_k+k ] + 1.61803 * B[ k*num_j+j ];
        }
      }
    }
  }
  uint64_t endTime = GetTimeMs64();
  return ( endTime - startTime ) / 1000.0;
}

double MatrixMultiply_1Dr( integer_t const num_i,
                          integer_t const num_j,
                          integer_t const num_k,
                          integer_t const ITERATIONS,
                          double const * const __restrict__  A,
                          double const * const __restrict__ B,
                          double * const __restrict__ C )
{
  uint64_t startTime = GetTimeMs64();
  for( integer_t iter = 0 ; iter < ITERATIONS ; ++iter )
  {
    for( integer_t i = 0 ; i < num_i ; ++i )
    {
      for( integer_t j = 0 ; j < num_j ; ++j )
      {
        for( integer_t k = 0 ; k < num_k ; ++k )
        {
          C[ i*num_j+j ] += A[ i*num_k+k ] * B[ k*num_j+j ] + 3.1415 * A[ i*num_k+k ] + 1.61803 * B[ k*num_j+j ];
        }
      }
    }
  }
  uint64_t endTime = GetTimeMs64();
  return ( endTime - startTime ) / 1000.0;
}



#define MATMULT \
uint64_t startTime = GetTimeMs64(); \
for( integer_t iter = 0 ; iter < ITERATIONS ; ++iter ) \
{ \
  for( integer_t i = 0 ; i < num_i ; ++i ) \
  { \
    for( integer_t j = 0 ; j < num_j ; ++j ) \
    { \
      for( integer_t k = 0 ; k < num_k ; ++k ) \
      { \
        C[i][j] += A[i][k] * B[k][j] + 3.1415 * A[i][k] + 1.61803 * B[k][j]; \
      } \
    } \
  } \
} \
uint64_t endTime = GetTimeMs64(); \
return ( endTime - startTime ) / 1000.0;

#define MATMULT2 \
uint64_t startTime = GetTimeMs64(); \
for( integer_t iter = 0 ; iter < ITERATIONS ; ++iter ) \
{ \
  for( integer_t i = 0 ; i < num_i ; ++i ) \
  { \
    for( integer_t j = 0 ; j < num_j ; ++j ) \
    { \
      for( integer_t k = 0 ; k < num_k ; ++k ) \
      { \
        C(i,j) += A(i,k) * B(k,j) + 3.1415 * A(i,k) + 1.61803 * B(k,j); \
      } \
    } \
  } \
} \
uint64_t endTime = GetTimeMs64(); \
return ( endTime - startTime ) / 1000.0;




double MatrixMultiply_2D_accessor( integer_t const num_i,
                                   integer_t const num_j,
                                   integer_t const num_k,
                                   integer_t const ITERATIONS,
                                   ArrayAccessor<double const,2> const A,
                                   ArrayAccessor<double const,2> const B,
                                   ArrayAccessor<double,2> C )
{
  MATMULT
}



double MatrixMultiply_2D_accessorRef( integer_t const num_i,
                                   integer_t const num_j,
                                   integer_t const num_k,
                                   integer_t const ITERATIONS,
                                   ArrayAccessor<double const,2> const & A,
                                   ArrayAccessor<double const,2> const & B,
                                   ArrayAccessor<double,2>& C )
{
  MATMULT
}






double MatrixMultiply_2D_accessorPBV2( integer_t const num_i,
                                   integer_t const num_j,
                                   integer_t const num_k,
                                   integer_t const ITERATIONS,
                                   ArrayAccessor<double const,2> const  A,
                                   ArrayAccessor<double const,2> const  B,
                                   ArrayAccessor<double,2> C )
{
  MATMULT2
}

double MatrixMultiply_2D_accessorRef2( integer_t const num_i,
                                   integer_t const num_j,
                                   integer_t const num_k,
                                   integer_t const ITERATIONS,
                                   ArrayAccessor<double const,2> const & A,
                                   ArrayAccessor<double const,2> const & B,
                                   ArrayAccessor<double,2>& C )
{
  MATMULT2
}

double MatrixMultiply_2D_RajaViewParen_PBV( integer_t const num_i,
                                       integer_t const num_j,
                                       integer_t const num_k,
                                       integer_t const ITERATIONS,
                                       RAJA::View< double const, RAJA::Layout<2> > A,
                                       RAJA::View< double const, RAJA::Layout<2> > B,
                                       RAJA::View< double, RAJA::Layout<2> > C )
{
  MATMULT2
}


double MatrixMultiply_2D_RajaViewParen_PBR( integer_t const num_i,
                                       integer_t const num_j,
                                       integer_t const num_k,
                                       integer_t const ITERATIONS,
                                       RAJA::View< double const, RAJA::Layout<2> > const & A,
                                       RAJA::View< double const, RAJA::Layout<2> > const & B,
                                       RAJA::View< double, RAJA::Layout<2> > & C )
{
  MATMULT2
}

double MatrixMultiply_2D_RajaViewSquare_PBV( integer_t const num_i,
                                       integer_t const num_j,
                                       integer_t const num_k,
                                       integer_t const ITERATIONS,
                                       RAJA::View< double const, RAJA::Layout2<2> > A,
                                       RAJA::View< double const, RAJA::Layout2<2> > B,
                                       RAJA::View< double, RAJA::Layout2<2> > C )
{
  MATMULT
}


double MatrixMultiply_2D_RajaViewSquare_PBR( integer_t const num_i,
                                       integer_t const num_j,
                                       integer_t const num_k,
                                       integer_t const ITERATIONS,
                                       RAJA::View< double const, RAJA::Layout2<2> > const & A,
                                       RAJA::View< double const, RAJA::Layout2<2> > const & B,
                                       RAJA::View< double, RAJA::Layout2<2> > & C )
{
  MATMULT
}

double MatrixMultiply_2D_constructAccessorR( integer_t const num_i,
                                            integer_t const num_j,
                                            integer_t const num_k,
                                            integer_t const ITERATIONS,
                                            double const * const __restrict__ ptrA,
                                            integer_t const * const lengthA,
                                            double const * const __restrict__ ptrB,
                                            integer_t const * const lengthB,
                                            double * const __restrict__ ptrC,
                                            integer_t const * const lengthC )
{

  ArrayAccessor<double const,2> const A( ptrA, lengthA );
  ArrayAccessor<double const,2> const B( ptrB, lengthB );
  ArrayAccessor<double,2> C( ptrC, lengthC );

  MATMULT
}

