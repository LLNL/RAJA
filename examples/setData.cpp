//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-18, Lawrence Livermore National Security, LLC.
//
// Produced at the Lawrence Livermore National Laboratory
//
// LLNL-CODE-689114
//
// All rights reserved.
//
// This file is part of RAJA.
//
// For details about use and distribution, please read RAJA/LICENSE.
//
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include <cstdlib>
#include <cstring>
#include <iostream>
#include <chrono>

#include "RAJA/RAJA.hpp"
#include "memoryManager.hpp"

//#define A(elem, row, col) A[col + row*NCOLS + elem*MAT_ENTRIES]
#define A2(elem, row, col) A2[col + NCOLS*(elem + Nelem*row)]

//#define B(col,row, elem) B[col + NCOLS*(row + elem*NROWS)]
#define B2(elem, row, col) B2[col + NCOLS*(elem + Nelem*row)]

const int NROWS = 3;
const int NCOLS = 4;
const int Nelem = 2;
const int MAT_ENTRIES = NROWS*NCOLS;

using RAJA::Index_type;
void compareOutput(double *C, double *CComp, Index_type Nelem);
int main(int RAJA_UNUSED_ARG(argc), char **RAJA_UNUSED_ARG(argv[]))
{

  srand(time(NULL));

  //Layout 1
  double * A  = memoryManager::allocate<double>(Nelem*MAT_ENTRIES);
  double * B  = memoryManager::allocate<double>(Nelem*MAT_ENTRIES);
  RAJA::View<double, RAJA::Layout<3> > Bview(B,Nelem, NROWS,NCOLS);

  //Layout 2
  double * A2 = memoryManager::allocate<double>(Nelem*MAT_ENTRIES);
  double * B2 = memoryManager::allocate<double>(Nelem*MAT_ENTRIES);

  //#define A2(elem, row, col) A2[elem + Nelem*(col  + row*NCOLS)]
  //auto layout = RAJA::make_permuted_layout({{Nelem, NROWS, NCOLS}}, RAJA::as_array<RAJA::Perm<1,2,0> >::get() );

  //#define A2(elem, row, col) A2[elem + Nelem*(row + NROWS*col)]
  //auto layout = RAJA::make_permuted_layout({{Nelem, NROWS, NCOLS}}, RAJA::as_array<RAJA::Perm<2,1,0> >::get() );
  
  //#define A2(elem, row, col) A2[col + NCOLS*(row + NROWS*elem)]
  //auto layout = RAJA::make_permuted_layout({{Nelem, NROWS, NCOLS}}, RAJA::as_array<RAJA::Perm<0,1,2> >::get() );
  
  //#define A(elem, row, col) A[col + NCOLS*(elem + Nelem*row)]
  auto layout = RAJA::make_permuted_layout({{Nelem, NROWS, NCOLS}}, RAJA::as_array<RAJA::Perm<1,0,2> >::get() );
  
  //#define A2(elem, row, col) A2[row + NROWS*(elem + Nelem*col)]
  //auto layout = RAJA::make_permuted_layout({{Nelem, NROWS, NCOLS}}, RAJA::as_array<RAJA::Perm<2,0,1> >::get() );

  //#define A2(elem, row, col) A2[row + NROWS*(col + NCOLS*elem)]
  //auto layout = RAJA::make_permuted_layout({{Nelem, NROWS, NCOLS}}, RAJA::as_array<RAJA::Perm<0,2,1> >::get() );


  
  RAJA::View<double, RAJA::Layout<3> > B2view(B2, layout);

  //--------------------------------------------
  //Initialize data using macros and views
  //--------------------------------------------
  for(Index_type e=0; e<Nelem; ++e)
    {
      for(Index_type row=0; row<NROWS; ++row)
        {
          for(Index_type col=0; col<NCOLS; ++col)
            {
              Index_type id = col + NCOLS*(row *Nelem);
              //A(e,row,col)     = id;
              //Bview(e,row,col) = id;

              A2(e,row,col) = id;

              A2(e,row,col) = id;
              B2view(e,row,col) = id;
              //B2view(col,row,e) = id;
            }
        }
    }
  //--------------------------------------------

  std::cout<<"Permute 1"<<std::endl;
  //compareOutput(A, B, Nelem); 
  std::cout<<"------- \n \n"<<std::endl;


  std::cout<<"Permute 2"<<std::endl;
  compareOutput(A2, B2, Nelem); 
  std::cout<<"------- \n \n"<<std::endl;

  for(Index_type e=0; e<Nelem; ++e)
    {
      for(Index_type row=0; row<NROWS; ++row)
        {
          for(Index_type col=0; col<NCOLS; ++col)
            {
              //Index_type id = c + NROWS*(r + NCOLS*e);
              //std::cout<<B2view(e,row,col)<<" ";
              std::cout<<B2(e,row,col)<<" ";
            } 
          std::cout<<" "<<std::endl;
        }
      std::cout<<"-----------------------"<<std::endl;
    }

  

  return 0;
}



void compareOutput(double *C, double *CComp, Index_type Nelem)
{

  bool status = true;
  for(Index_type e = 0; e<Nelem; ++e)
    {      
      for(Index_type r=0; r<NROWS; ++r)
        {
          for(Index_type c=0; c<NCOLS; ++c)
            {
              Index_type id = c + NROWS*(r + NCOLS*e);
              double terr = std::abs(C[id] - CComp[id]);
              if((terr) > 1e-8)
                {
                  status = false;
                }                        
            }          
        }
    }

  
  if(status==false)
    {
      std::cout<<"Data set - fail"<<std::endl;
    }else{
    std::cout<<"Data set - pass"<<std::endl;
  }
  
}

