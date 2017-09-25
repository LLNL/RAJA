/*
 * main.cpp
 *
 *  Created on: Jul 30, 2016
 *      Author: rrsettgast
 */


#include "arrayAccessorPerformance.hpp"

int main( int /*argc*/, char* argv[] )
{
//  int testLengths[] = {2,4,3};
//  std::cout<<"stride<3>(testLengths) = "<<stride<3>(testLengths)<<std::endl;
//  std::cout<<"stride<2>(testLengths) = "<<stride<2>(testLengths+1)<<std::endl;
//  std::cout<<"stride<1>(testLengths) = "<<stride<1>(testLengths+2)<<std::endl;
//
//
//  dimensions<2,4,3> dims;
//  std::cout<<"dims.ndim() = "<<dimensions<1,2,3> ::NDIMS()<<std::endl;
////  std::cout<<"dims.ndim() = "<<dims.ndims()<<std::endl;
//  std::cout<<"dims.dim0   = "<<dims.DIMENSION<0>()<<std::endl;
//  std::cout<<"dims.dim1   = "<<dims.DIMENSION<1>()<<std::endl;
//  std::cout<<"dims.dim2   = "<<dims.DIMENSION<2>()<<std::endl;
//
//  std::cout<<"dims.STRIDE<0>   = "<<dims.STRIDE<0>()<<std::endl;
//  std::cout<<"dims.STRIDE<1>   = "<<dims.STRIDE<1>()<<std::endl;
//  std::cout<<"dims.STRIDE<2>   = "<<dims.STRIDE<2>()<<std::endl;

  integer_t seed = time( NULL );

  const integer_t num_i = std::stoi( argv[1] );
  const integer_t num_k = std::stoi( argv[2] );
  const integer_t num_j = std::stoi( argv[3] );
  const integer_t ITERATIONS = std::stoi( argv[4] );
  const integer_t seedmod = std::stoi( argv[5] );

  const integer_t output = std::stoi( argv[6] );

  //***************************************************************************
  //***** Setup Arrays ********************************************************
  //***************************************************************************

  double * const restrict A  = new double[num_i*num_k];
  double * const restrict B  = new double[num_k*num_j];
  double * const restrict C1D          = new double[num_i*num_j];
  double * const restrict C1D_restrict = new double[num_i*num_j];

  double * const restrict ptrC_SquareNFC = new double[num_i*num_j];
  double * const restrict ptrC_SquarePBV = new double[num_i*num_j];
  double * const restrict ptrC_SquarePBR = new double[num_i*num_j];
  double * const restrict ptrC_SquareLAC = new double[num_i*num_j];

  double * const restrict ptrC_ParenNFC = new double[num_i*num_j];
  double * const restrict ptrC_ParenPBV = new double[num_i*num_j];
  double * const restrict ptrC_ParenPBR = new double[num_i*num_j];

  double * const restrict ptrC_NFCRajaView = new double[num_i*num_j];
  double * const restrict ptrC_PBVRajaView = new double[num_i*num_j];
  double * const restrict ptrC_PBRRajaView = new double[num_i*num_j];


  double * const restrict ptrA_SelfDescribing  = new double[num_i*num_k + maxDim()];
  double * const restrict ptrB_SelfDescribing  = new double[num_k*num_j + maxDim()];
  double * const restrict ptrC_SelfDescribing  = new double[num_i*num_j + maxDim()];

  integer_t * A2_dims = reinterpret_cast<integer_t*>( &(ptrA_SelfDescribing[0]) );
  A2_dims[0] = 2;
  A2_dims[1] = num_i;
  A2_dims[2] = num_k;
  double * ptrAdata_SelfDescribing = &(ptrA_SelfDescribing[maxDim()]);

  integer_t * B2_dims = reinterpret_cast<integer_t*>( &(ptrB_SelfDescribing[0]) );
  B2_dims[0] = 2;
  B2_dims[1] = num_k;
  B2_dims[2] = num_j;
  double * ptrBdata_SelfDescribing = &(ptrB_SelfDescribing[maxDim()]);


  integer_t * C2_dims = reinterpret_cast<integer_t*>( &(ptrC_SelfDescribing[0]) );
  C2_dims[0] = 2;
  C2_dims[1] = num_i;
  C2_dims[2] = num_j;
  double * ptrCdata_SelfDescribing = &(ptrC_SelfDescribing[maxDim()]);



  srand( seed * seedmod );

  for( integer_t i = 0 ; i < num_i ; ++i )
    for( integer_t k = 0 ; k < num_k ; ++k )
    {
      A[i*num_k+k] = rand();
      ptrAdata_SelfDescribing[i*num_k+k] = A[i*num_k+k];
    }

  for( integer_t k = 0 ; k < num_k ; ++k )
    for( integer_t j = 0 ; j < num_j ; ++j )
    {
      B[k*num_j+j] = rand();
      ptrBdata_SelfDescribing[k*num_j+j] = B[k*num_j+j];
    }

  for( integer_t i = 0 ; i < num_i ; ++i )
  {
    for( integer_t j = 0 ; j < num_j ; ++j )
    {
      C1D[i*num_j+j] = 0.0;
      C1D_restrict[i*num_j+j] = 0.0;

      ptrC_SquareNFC[i*num_j+j] = 0.0;
      ptrC_SquarePBV[i*num_j+j] = 0.0;
      ptrC_SquarePBR[i*num_j+j] = 0.0;
      ptrC_SquareLAC[i*num_j+j] = 0.0;

      ptrC_ParenNFC[i*num_j+j] = 0.0;
      ptrC_ParenPBV[i*num_j+j] = 0.0;
      ptrC_ParenPBR[i*num_j+j] = 0.0;

      ptrC_NFCRajaView[i*num_j+j] = 0.0;
      ptrC_PBVRajaView[i*num_j+j] = 0.0;
      ptrC_PBRRajaView[i*num_j+j] = 0.0;
      ptrCdata_SelfDescribing[i*num_j+j] = 0.0;
    }
  }




  integer_t lengthsA[] = { num_i , num_k };
  integer_t lengthsB[] = { num_k , num_j };
  integer_t lengthsC[] = { num_i , num_j };

  ArrayAccessor<double,2> accessorA( A, lengthsA );
  ArrayAccessor<double,2> accessorB( B, lengthsB );
  ArrayAccessor<double,2> accessorC_SquareNFC( ptrC_SquareNFC, lengthsC );
  ArrayAccessor<double,2> accessorC_SquarePBV( ptrC_SquarePBV, lengthsC );
  ArrayAccessor<double,2> accessorC_SquarePBR( ptrC_SquarePBR, lengthsC );

  ArrayAccessor<double,2> accessorC_ParenNFC( ptrC_ParenNFC, lengthsC );
  ArrayAccessor<double,2> accessorC_ParenPBV( ptrC_ParenPBV, lengthsC );
  ArrayAccessor<double,2> accessorC_ParenPBR( ptrC_ParenPBR, lengthsC );


  ArrayAccessor<double,2> accessorA_SelfDescribing( ptrAdata_SelfDescribing, lengthsA );
  ArrayAccessor<double,2> accessorB_SelfDescribing( ptrBdata_SelfDescribing, lengthsB );
  ArrayAccessor<double,2> accessorC_SelfDescribing( ptrCdata_SelfDescribing, lengthsC );

  RAJA::View< double const, RAJA::Layout<2> > A_View( A, num_i, num_k );
  RAJA::View< double const, RAJA::Layout<2> > B_View( B, num_k, num_j );
  RAJA::View< double, RAJA::Layout<2> > C_ViewNFC( ptrC_NFCRajaView, num_i, num_j );
  RAJA::View< double, RAJA::Layout<2> > C_ViewPBV( ptrC_PBVRajaView, num_i, num_j );
  RAJA::View< double, RAJA::Layout<2> > C_ViewPBR( ptrC_PBRRajaView, num_i, num_j );




  double runTime_Direct1dAccess  = MatrixMultiply_1D ( num_i, num_j, num_k, ITERATIONS, A, B, C1D );

  double runTime_Direct1dRestrict = MatrixMultiply_1Dr( num_i, num_j, num_k, ITERATIONS, A, B, C1D_restrict );








  double startTime = GetTimeMs64();
  for( integer_t iter = 0 ; iter < ITERATIONS ; ++iter )
  {
    for( integer_t i = 0 ; i < num_i ; ++i )
    {
      for( integer_t j = 0 ; j < num_j ; ++j )
      {
        for( integer_t k = 0 ; k < num_k ; ++k )
        {
          accessorC_SquareNFC[i][j] += accessorA[i][k] * accessorB[k][j] + 3.1415 * accessorA[i][k] + 1.61803 * accessorB[k][j];;
        }
      }
    }
  }
  double endTime = GetTimeMs64();
  double runTime_SquareNFC = ( endTime - startTime ) / 1000.0;


  double runTime_SquarePBV = MatrixMultiply_2D_accessor(       num_i, num_j, num_k, ITERATIONS, accessorA, accessorB, accessorC_SquarePBV );

  double runTime_SquarePBR = MatrixMultiply_2D_accessorRef(       num_i, num_j, num_k, ITERATIONS, accessorA, accessorB, accessorC_SquarePBR );


  double runTime_SquareLAC = MatrixMultiply_2D_constructAccessorR( num_i, num_j, num_k, ITERATIONS,
                                                                                   A, lengthsA,
                                                                                   B, lengthsB,
                                                                                   ptrC_SquareLAC, lengthsC );



  double runTime_PBVSelfDescribing = MatrixMultiply_2D_accessorRef( num_i, num_j, num_k, ITERATIONS, accessorA_SelfDescribing, accessorB_SelfDescribing, accessorC_SelfDescribing );

  startTime = GetTimeMs64();
  for( integer_t iter = 0 ; iter < ITERATIONS ; ++iter )
  {
    for( integer_t i = 0 ; i < num_i ; ++i )
    {
      for( integer_t j = 0 ; j < num_j ; ++j )
      {
        for( integer_t k = 0 ; k < num_k ; ++k )
        {
          accessorC_ParenNFC(i,j) += accessorA(i,k) * accessorB(k,j) + 3.1415 * accessorA(i,k) + 1.61803 * accessorB(k,j);
        }
      }
    }
  }
  endTime = GetTimeMs64();
  double runTime_ParenNFC =( endTime - startTime ) / 1000.0;

  double runTime_ParenPBV = MatrixMultiply_2D_accessorPBV2( num_i, num_j, num_k, ITERATIONS, accessorA, accessorB, accessorC_ParenPBV );
  double runTime_ParenPBR = MatrixMultiply_2D_accessorRef2( num_i, num_j, num_k, ITERATIONS, accessorA, accessorB, accessorC_ParenPBR );






  startTime = GetTimeMs64();
  for( integer_t iter = 0 ; iter < ITERATIONS ; ++iter )
  {
    for( integer_t i = 0 ; i < num_i ; ++i )
    {
      for( integer_t j = 0 ; j < num_j ; ++j )
      {
        for( integer_t k = 0 ; k < num_k ; ++k )
        {
          C_ViewNFC(i,j) += A_View(i,k) * B_View(k,j) + 3.1415 * A_View(i,k) + 1.61803 * B_View(k,j);
        }
      }
    }
  }
  endTime = GetTimeMs64();
  double runTime_RAJA_NFC =( endTime - startTime ) / 1000.0;

  double runTime_RAJA_PBV = MatrixMultiply_2D_RajaView_PBV( num_i, num_j, num_k, ITERATIONS, A_View, B_View, C_ViewPBV );
  double runTime_RAJA_PBR = MatrixMultiply_2D_RajaView_PBV( num_i, num_j, num_k, ITERATIONS, A_View, B_View, C_ViewPBR );



  double minRunTime = 1.0e99;
  minRunTime = std::min(runTime_Direct1dAccess,runTime_Direct1dRestrict);
  minRunTime = std::min( minRunTime, runTime_SquareNFC );
  minRunTime = std::min( minRunTime, runTime_SquarePBV );
  minRunTime = std::min( minRunTime, runTime_SquarePBR );
  minRunTime = std::min( minRunTime, runTime_SquareLAC );
  minRunTime = std::min( minRunTime, runTime_PBVSelfDescribing );
  minRunTime = std::min( minRunTime, runTime_ParenNFC );
  minRunTime = std::min( minRunTime, runTime_ParenPBV );
  minRunTime = std::min( minRunTime, runTime_ParenPBR );
  minRunTime = std::min( minRunTime, runTime_RAJA_NFC );
  minRunTime = std::min( minRunTime, runTime_RAJA_PBV );
  minRunTime = std::min( minRunTime, runTime_RAJA_PBR );

  if( output > 2 )
  {
    double error_SquareNFC   = 0.0;
    double error_SquarePBV = 0.0;
    double error_SquarePBR = 0.0;
    double error_SquareLAC = 0.0;
    double error_SelfDescribing = 0.0;

    double error_ParenNFC = 0.0;
    double error_ParenPBV = 0.0;
    double error_ParenPBR = 0.0;

    double error_RajaNFC = 0.0;
    double error_RajaPBV = 0.0;
    double error_RajaPBR = 0.0;


    for( integer_t i = 0 ; i < num_i ; ++i )
    {
      for( integer_t j = 0 ; j < num_j ; ++j )
      {
        error_SquareNFC += pow( C1D[i*num_j+j] - ptrC_SquareNFC[i*num_j+j] , 2 ) ;
        error_SquarePBV += pow( C1D[i*num_j+j] - ptrC_SquarePBV[i*num_j+j] , 2 ) ;
        error_SquarePBR += pow( C1D[i*num_j+j] - ptrC_SquarePBR[i*num_j+j] , 2 ) ;
        error_SquareLAC += pow( C1D[i*num_j+j] - ptrC_SquareLAC[i*num_j+j] , 2 ) ;
        error_SelfDescribing += pow( C1D[i*num_j+j] - ptrCdata_SelfDescribing[i*num_j+j] , 2 ) ;

        error_ParenNFC += pow( C1D[i*num_j+j] - ptrC_ParenNFC[i*num_j+j] , 2 ) ;
        error_ParenPBV += pow( C1D[i*num_j+j] - ptrC_ParenPBV[i*num_j+j] , 2 ) ;
        error_ParenPBR += pow( C1D[i*num_j+j] - ptrC_ParenPBR[i*num_j+j] , 2 ) ;


        error_RajaNFC += pow( C1D[i*num_j+j] - ptrC_NFCRajaView[i*num_j+j] , 2 ) ;
        error_RajaPBV += pow( C1D[i*num_j+j] - ptrC_PBVRajaView[i*num_j+j] , 2 ) ;
        error_RajaPBR += pow( C1D[i*num_j+j] - ptrC_PBRRajaView[i*num_j+j] , 2 ) ;
      }
    }
    std::cout<<"error_SquareNFC = "<<error_SquareNFC<<std::endl;
    std::cout<<"error_SquarePBV = "<<error_SquarePBV<<std::endl;
    std::cout<<"error_SquarePBR = "<<error_SquarePBR<<std::endl;
    std::cout<<"error_SquarePBR = "<<error_SquarePBR<<std::endl;
    std::cout<<"error_SquareLAC = "<<error_SquareLAC<<std::endl;
    std::cout<<"error_SquareSD = "<<error_SelfDescribing<<std::endl;

    std::cout<<"error_ParenNFC = "<<error_ParenNFC<<std::endl;
    std::cout<<"error_ParenPBV = "<<error_ParenPBV<<std::endl;
    std::cout<<"error_ParenPBR = "<<error_ParenPBR<<std::endl;

    std::cout<<"error_RajaNFC = "<<error_RajaNFC<<std::endl;
    std::cout<<"error_RajaPBV = "<<error_RajaPBV<<std::endl;
    std::cout<<"error_RajaPBR = "<<error_RajaPBR<<std::endl;
  }

  if( output > 1 )
  {
    printf( "1d array                             : %8.3f, %8.2f\n", runTime_Direct1dAccess, runTime_Direct1dAccess/minRunTime);
    printf( "1d array restrict                    : %8.3f, %8.2f\n", runTime_Direct1dRestrict, runTime_Direct1dRestrict / minRunTime);

    printf( "accessor[] nfc                       : %8.3f, %8.2f\n", runTime_SquareNFC, runTime_SquareNFC / minRunTime);
    printf( "accessor[] pbv                       : %8.3f, %8.2f\n", runTime_SquarePBV, runTime_SquarePBV / minRunTime);
    printf( "accessor[] pbr                       : %8.3f, %8.2f\n", runTime_SquarePBR, runTime_SquarePBR / minRunTime);
    printf( "accessor[] lac                       : %8.3f, %8.2f\n", runTime_SquareLAC, runTime_SquareLAC / minRunTime);

    printf( "accessor() nfc                       : %8.3f, %8.2f\n", runTime_ParenNFC, runTime_ParenNFC / minRunTime);
    printf( "accessor() pbv                       : %8.3f, %8.2f\n", runTime_ParenPBV, runTime_ParenPBV / minRunTime);
    printf( "accessor() pbr                       : %8.3f, %8.2f\n", runTime_ParenPBR, runTime_ParenPBR / minRunTime);

    printf( "RAJA::View::operator() nfc           : %8.3f, %8.2f\n", runTime_RAJA_NFC, runTime_RAJA_NFC / minRunTime);
    printf( "RAJA::View::operator() pbv           : %8.3f, %8.2f\n", runTime_RAJA_PBV, runTime_RAJA_PBV / minRunTime);
    printf( "RAJA::View::operator() pbr           : %8.3f, %8.2f\n", runTime_RAJA_PBR, runTime_RAJA_PBR / minRunTime);
  }


  if( output == 1 )
  {
    printf( "%8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f\n",
    runTime_Direct1dAccess,
    runTime_Direct1dRestrict,

    runTime_SquareNFC,
    runTime_SquarePBV,
    runTime_SquarePBR,
    runTime_SquareLAC,

    runTime_ParenNFC,
    runTime_ParenPBV,
    runTime_ParenPBR,

    runTime_RAJA_NFC,
    runTime_RAJA_PBV,
    runTime_RAJA_PBR );
  }
  return 0;
}
