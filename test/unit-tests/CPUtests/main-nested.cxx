/*
 * Copyright (c) 2016, Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 *
 * All rights reserved.
 *
 * This source code cannot be distributed without permission and
 * further review from Lawrence Livermore National Laboratory.
 */

//
// Main program illustrating RAJA nested-loop execution
//

#include <cstdlib>
#include <time.h>

#include<string>
#include<vector>
#include<iostream>

#include "RAJA/RAJA.hxx"

using namespace RAJA;
using namespace std;

#include "Compare.hxx"

//
// Global variables for counting tests executed/passed.
//
unsigned s_ntests_run_total = 0;
unsigned s_ntests_passed_total = 0;

unsigned s_ntests_run = 0;
unsigned s_ntests_passed = 0;



#if 0
///////////////////////////////////////////////////////////////////////////
//
// Method that defines and runs a basic RAJA 2d kernel test
//
///////////////////////////////////////////////////////////////////////////
template <typename POL>
void run2dTest(std::string const &policy, Index_type size_i, Index_type size_j)
{

   cout << "\n Test2d " << size_i << "x" << size_j << 
       " array reduction for policy " << policy << "\n";
   
   std::vector<int> values(size_i*size_j, 1);
      
    s_ntests_run++;
    s_ntests_run_total++;



    ////
    //// DO WORK
    ////
    
    typename POL::View val_view(&values[0], size_i, size_j);

    forall2<typename POL::Exec>( 
      RangeSegment(1,size_i), 
      RangeSegment(0,size_j), 
      [=] (Index_type i, Index_type j) {
        val_view(0,j) += val_view(i,j);
    } );


    
    ////
    //// CHECK ANSWER
    ////
    size_t nfailed = 0;
    forall2<Forall2_Policy<seq_exec, seq_exec>>(
      RangeSegment(0,size_i), 
      RangeSegment(0,size_j), 
      [&] (Index_type i, Index_type j) {
        if(i == 0){
          if(val_view(i,j) != size_i){
            ++ nfailed;
          }
        }
        else{
          if(val_view(i,j) != 1){
            ++ nfailed;
          }
        }
    } );    

    if ( nfailed ) {
    
       cout << "\n TEST FAILURE: " << nfailed << " elements failed" << endl;

    } else {
       s_ntests_passed++;
       s_ntests_passed_total++;
    }
}





// Sequentail, IJ ordering
struct Pol2dA {
  typedef Forall2_Policy<seq_exec, seq_exec,
                           Forall2_Permute<PERM_IJ>
                        > Exec; 
  typedef RAJA::View2d<int, RAJA::Layout2d<PERM_IJ>> View;  
};
#ifdef _OPENMP
// Sequentail, JI ordering
struct Pol2dB {
  typedef RAJA::Forall2_Policy<seq_exec, seq_exec, 
                                 Forall2_Permute<PERM_JI>
                              > Exec;  
  typedef RAJA::View2d<int, RAJA::Layout2d<PERM_JI>> View;  
};

// OpenMP, IJ ordering

struct Pol2dC {
  typedef RAJA::Forall2_Policy<seq_exec, omp_for_nowait_exec, 
                                 Forall2_OMP_Parallel<
                                    Forall2_Permute<PERM_IJ>
                                 >
                              > Exec;  
  typedef RAJA::View2d<int, RAJA::Layout2d<PERM_IJ>> View;  
};

// OpenMP, JI ordering
struct Pol2dD {
  typedef RAJA::Forall2_Policy<seq_exec, omp_for_nowait_exec, 
                                 Forall2_OMP_Parallel<
                                    Forall2_Permute<PERM_JI>
                                 >
                              > Exec;  
  typedef RAJA::View2d<int, RAJA::Layout2d<PERM_JI>> View;  
};

#endif // _OPENMP

void run2dTests(Index_type size_i, Index_type size_j){

  run2dTest<Pol2dA>("Pol2dA", size_i, size_j);
  run2dTest<Pol2dB>("Pol2dB", size_i, size_j);

#ifdef _OPENMP
  run2dTest<Pol2dC>("Pol2dC", size_i, size_j);
  run2dTest<Pol2dD>("Pol2dD", size_i, size_j);
#endif // _OPENMP
}


#endif

//typedef Forall2_Policy<seq_exec, seq_exec, ForallN_Permute<PERM_JI> > cudapol;

#ifdef RAJA_USE_CUDA
typedef ForallN_Policy<ExecList<cuda_exec<1>, seq_exec >,
                         //Tile<TileList<tile_fixed<2>, tile_fixed<2>>,
                           Permute<PERM_JI,
                             ForallN_Execute
                           >
                         //>
                      > npol;
#else
typedef ForallN_Policy<ExecList<seq_exec, omp_for_nowait_exec >,
                       OMP_Parallel<
                         Tile<TileList<tile_fixed<2>, tile_fixed<2>>,
                           ForallN_Execute
                         >
                       >
                      > npol;
#endif


typedef ForallN_Policy<ExecList<seq_exec, seq_exec, seq_exec>,
                         ForallN_Execute > cudapol3;


typedef ForallN_Policy<ExecList<seq_exec, seq_exec, seq_exec, seq_exec>,
                         ForallN_Execute > cudapol4;




///////////////////////////////////////////////////////////////////////////
//
// Main Program.
//
///////////////////////////////////////////////////////////////////////////

struct fcn {
  inline void RAJA_HOST_DEVICE
  operator()(Index_type i, Index_type j) const {
#ifdef __CUDA_ARCH__
    printf("(%d, %d) FROM GPU\n", (int)i, (int)j);
#else
    printf("(%d, %d) FROM CPU\n", (int)i, (int)j);
#endif
  }
};

struct fcn1 {
  inline void RAJA_HOST_DEVICE operator()(Index_type i) const{
    printf("(%d)\n", (int)i);
  }
};

struct fcn3 {
  inline void RAJA_HOST_DEVICE operator()(Index_type i, Index_type j, Index_type k) const{
    printf("(%d, %d, %d)\n", (int)i, (int)j, (int)k);
  }
};


struct fcn4 {
  inline void RAJA_HOST_DEVICE operator()(Index_type i, Index_type j, Index_type k, Index_type l) const{
    printf("(%d, %d, %d, %d)\n", (int)i, (int)j, (int)k, (int)l);
  }
};


int main(int argc, char *argv[])
{

 

///////////////////////////////////////////////////////////////////////////
//
// Run RAJA::forall nested loop tests...
//
///////////////////////////////////////////////////////////////////////////


   //run2dTests(128,1024);
   //run2dTests(37,1);
   //run2dTests(1,192);

   ///
   /// Print total number of tests passed/run.
   ///

  //cudaDeviceSynchronize();
   cout << "\n All Tests : # run / # passed = " 
             << s_ntests_passed_total << " / " 
             << s_ntests_run_total << endl;

  /* forall<cuda_exec<1> >(
       RangeSegment(0,4),
       [=] __device__ (int i){printf("%d\n", i);});
*/
/*
   typedef RAJA::View<double, RAJA::Layout<int, PERM_IJ, int, int>> View;

   double data[16];
   View v(data, 4,4);

   for(int i = 0;i < 4;++ i){
     for(int j = 0;j < 4;++ j){
       v(i,j) = i*10+j;
     }
   }
   for(int i = 0;i < 16;++ i){
     printf("data[%d]=%.0f\n", i, data[i]);
   }


   printf("IJ:\n");
   forallN<npol>(
       RangeSegment(0, 4),
       RangeSegment(0, 4),
      fcn() );

*/
/*
   printf("JI:\n");
   forallN<cudapol2>(
      RangeSegment(0, 4),
      RangeSegment(0, 4),
     fcn_obj );

*/

   printf("JIK:\n");
   forallN<cudapol3>(
      RangeSegment(0, 2),
      RangeSegment(0, 2),
      RangeSegment(0, 2),
     fcn3() );

   printf("JLIK:\n");
   forallN<cudapol4>(
      RangeSegment(0, 2),
      RangeSegment(0, 2),
      RangeSegment(0, 2),
      RangeSegment(0, 2),
     fcn4() );

//
// Clean up....
//  

   cout << "\n DONE!!! " << endl;

   return 0 ;
}

