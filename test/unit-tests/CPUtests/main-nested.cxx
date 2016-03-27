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


///////////////////////////////////////////////////////////////////////////
//
// Main Program.
//
///////////////////////////////////////////////////////////////////////////
int main(int argc, char *argv[])
{

 

///////////////////////////////////////////////////////////////////////////
//
// Run RAJA::forall reduction tests...
//
///////////////////////////////////////////////////////////////////////////


   run2dTests(128,1024);
   run2dTests(37,1);
   run2dTests(1,192);

   ///
   /// Print total number of tests passed/run.
   ///
   cout << "\n All Tests : # run / # passed = " 
             << s_ntests_passed_total << " / " 
             << s_ntests_run_total << endl;


//
// Clean up....
//  

   cout << "\n DONE!!! " << endl;

   return 0 ;
}

