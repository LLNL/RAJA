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
#include <cmath>
#include <time.h>
#include <cmath>

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
    
    typename POL::VIEW val_view(&values[0], size_i, size_j);

    forallN<typename POL::EXEC>(
      RangeSegment(1,size_i), 
      RangeSegment(0,size_j), 
      [=] (Index_type i, Index_type j) {
        val_view(0,j) += val_view(i,j);
    } );


    
    ////
    //// CHECK ANSWER
    ////
    size_t nfailed = 0;
    forallN<NestedPolicy<ExecList<seq_exec, seq_exec>>>(
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





// Sequential, IJ ordering
struct Pol2dA {
  typedef NestedPolicy<ExecList<seq_exec, seq_exec>,
                        Permute<PERM_IJ>
                      > EXEC;

  typedef RAJA::View<int, Layout<int, PERM_IJ, int, int>> VIEW;
};

// SIMD, JI ordering
struct Pol2dB {
  typedef NestedPolicy<ExecList<simd_exec, seq_exec>,
                        Permute<PERM_JI>
                      > EXEC;

  typedef RAJA::View<int, Layout<int, PERM_JI, int, int>> VIEW;
};


// SIMD, Tiled JI ordering
struct Pol2dC {
  typedef NestedPolicy<ExecList<simd_exec, seq_exec>,
                        Tile<TileList<tile_fixed<8>, tile_fixed<16>>,
                          Permute<PERM_JI>
                        >
                      > EXEC;

  typedef RAJA::View<int, Layout<int, PERM_JI, int, int>> VIEW;
};

// SIMD, Two-level tiled JI ordering
struct Pol2dD {
  typedef NestedPolicy<ExecList<simd_exec, seq_exec>,
                        Tile<TileList<tile_fixed<32>, tile_fixed<32>>,
                          Tile<TileList<tile_fixed<8>, tile_fixed<16>>,
                            Permute<PERM_JI>
                          >
                        >
                      > EXEC;

  typedef RAJA::View<int, Layout<int, PERM_JI, int, int>> VIEW;
};

#ifdef RAJA_USE_OPENMP

// OpenMP/Sequential, IJ ordering
struct Pol2dA_OMP {
  typedef NestedPolicy<ExecList<seq_exec, omp_parallel_for_exec>,
                        Permute<PERM_IJ>
                      > EXEC;

  typedef RAJA::View<int, Layout<int, PERM_IJ, int, int>> VIEW;
};

// OpenMP/SIMD, JI ordering, nowait
struct Pol2dB_OMP {
  typedef NestedPolicy<ExecList<simd_exec, omp_for_nowait_exec>,
                        OMP_Parallel<
                          Permute<PERM_JI>
                        >
                      > EXEC;

  typedef RAJA::View<int, Layout<int, PERM_JI, int, int>> VIEW;
};


// OpenMP/SIMD, Tiled JI ordering, nowait
struct Pol2dC_OMP {
  typedef NestedPolicy<ExecList<simd_exec, omp_for_nowait_exec>,
                        OMP_Parallel<
                          Tile<TileList<tile_fixed<8>, tile_fixed<16>>,
                            Permute<PERM_JI>
                          >
                        >
                      > EXEC;

  typedef RAJA::View<int, Layout<int, PERM_JI, int, int>> VIEW;
};

// OpenMP/SIMD, Two-level tiled JI ordering, nowait
struct Pol2dD_OMP {
  typedef NestedPolicy<ExecList<simd_exec, omp_for_nowait_exec>,
                        OMP_Parallel<
                          Tile<TileList<tile_fixed<32>, tile_fixed<32>>,
                            Tile<TileList<tile_fixed<8>, tile_fixed<16>>,
                              Permute<PERM_JI>
                            >
                          >
                        >
                      > EXEC;

  typedef RAJA::View<int, Layout<int, PERM_JI, int, int>> VIEW;
};


#endif

void run2dTests(Index_type size_i, Index_type size_j){
  run2dTest<Pol2dA>("Pol2dA", size_i, size_j);
  run2dTest<Pol2dB>("Pol2dB", size_i, size_j);
  run2dTest<Pol2dC>("Pol2dC", size_i, size_j);
  run2dTest<Pol2dD>("Pol2dD", size_i, size_j);

#ifdef RAJA_USE_OPENMP
  run2dTest<Pol2dA_OMP>("Pol2dA_OMP", size_i, size_j);
  run2dTest<Pol2dB_OMP>("Pol2dB_OMP", size_i, size_j);
  run2dTest<Pol2dC_OMP>("Pol2dC_OMP", size_i, size_j);
  run2dTest<Pol2dD_OMP>("Pol2dD_OMP", size_i, size_j);
#endif 

}




///////////////////////////////////////////////////////////////////////////
//
// Example LTimes kernel test routines
//
// Demonstrates a 4-nested loop, the use of complex nested policies and
// the use of strongly-typed indices
//
// This routine computes phi(m, g, z) = SUM_d {  ell(m, d)*psi(d,g,z)  }
//
///////////////////////////////////////////////////////////////////////////

RAJA_INDEX_VALUE(IMoment, "IMoment");
RAJA_INDEX_VALUE(IDirection, "IDirection");
RAJA_INDEX_VALUE(IGroup, "IGroup");
RAJA_INDEX_VALUE(IZone, "IZone");

template <typename POL>
void runLTimesTest(std::string const &policy, Index_type num_moments, Index_type num_directions, Index_type num_groups, Index_type num_zones)
{

   cout << "\n TestLTimes " << num_moments << " moments, " << num_directions << " directions, " << num_groups << " groups, and " << num_zones << " zones"
       << " with policy " << policy << "\n";


    s_ntests_run++;
    s_ntests_run_total++;


    // allocate data
    // phi is initialized to all zeros, the others are randomized
    std::vector<double> ell_data(num_moments*num_directions);
    std::vector<double> psi_data(num_directions*num_groups*num_zones);
    std::vector<double> phi_data(num_moments*num_groups*num_zones, 0.0);

    // randomize data
    for(size_t i = 0;i < ell_data.size();++i){
      ell_data[i] = drand48();
    }
    for(size_t i = 0;i < psi_data.size();++i){
      psi_data[i] = drand48();
    }


    // create views on data
    typename POL::ELL_VIEW ell(&ell_data[0], num_moments, num_directions);
    typename POL::PSI_VIEW psi(&psi_data[0], num_directions, num_groups, num_zones);
    typename POL::PHI_VIEW phi(&phi_data[0], num_moments, num_groups, num_zones);


    // get execution policy
    using EXEC = typename POL::EXEC;

    // do calculation using RAJA
    forallN<EXEC, IMoment, IDirection, IGroup, IZone>(
        RangeSegment(0, num_moments),
        RangeSegment(0, num_directions),
        RangeSegment(0, num_groups),
        RangeSegment(0, num_zones),
        [=](IMoment m, IDirection d, IGroup g, IZone z){
          phi(m,g,z) += ell(m,d) * psi(d,g,z);
        }
    );

    ////
    //// CHECK ANSWER against the hand-written sequential kernel
    ////
    size_t nfailed = 0;
    for(IZone z(0);z < num_zones;++ z){
      for(IGroup g(0);g < num_groups;++ g){
        for(IMoment m(0);m < num_moments;++ m){
          double total = 0.0;
          for(IDirection d(0);d < num_directions;++ d){
            total += ell(m,d) * psi(d,g,z);
          }

          // check answer with some reasonable tolerance
          if(fabs(total-phi(m,g,z)) > 1e-12){
            nfailed ++;
          }
        }
      }
    }


    if ( nfailed ) {

       cout << "\n TEST FAILURE: " << nfailed << " elements failed" << endl;

    } else {
       s_ntests_passed++;
       s_ntests_passed_total++;
    }
}

// Sequential
struct PolLTimesA {
  // Loops: Moments, Directions, Groups, Zones
  typedef NestedPolicy<ExecList<seq_exec, seq_exec, seq_exec, seq_exec> > EXEC;

  // psi[direction, group, zone]
  typedef RAJA::View<double, Layout<int, PERM_IJK, IDirection, IGroup, IZone>> PSI_VIEW;

  // phi[moment, group, zone]
  typedef RAJA::View<double, Layout<int, PERM_IJK, IMoment, IGroup, IZone>> PHI_VIEW;

  // ell[moment, direction]
  typedef RAJA::View<double, Layout<int, PERM_IJ, IMoment, IDirection>> ELL_VIEW;
};

// Sequential, reversed permutation
struct PolLTimesB {
  // Loops: Moments, Directions, Groups, Zones
  typedef NestedPolicy<ExecList<seq_exec, seq_exec, seq_exec, seq_exec>,
        Permute<PERM_LKJI>
      > EXEC;

  // psi[direction, group, zone]
  typedef RAJA::View<double, Layout<int, PERM_KJI, IDirection, IGroup, IZone>> PSI_VIEW;

  // phi[moment, group, zone]
  typedef RAJA::View<double, Layout<int, PERM_IJK, IMoment, IGroup, IZone>> PHI_VIEW;

  // ell[moment, direction]
  typedef RAJA::View<double, Layout<int, PERM_JI, IMoment, IDirection>> ELL_VIEW;
};

// Sequential, Tiled, another permutation
struct PolLTimesC {
  // Loops: Moments, Directions, Groups, Zones
  typedef NestedPolicy<ExecList<seq_exec, seq_exec, seq_exec, seq_exec>,
        Tile<TileList<tile_none, tile_none, tile_fixed<64>, tile_fixed<64>>,
          Permute<PERM_JKIL>
        >
      > EXEC;

  // psi[direction, group, zone]
  typedef RAJA::View<double, Layout<int, PERM_IJK, IDirection, IGroup, IZone>> PSI_VIEW;

  // phi[moment, group, zone]
  typedef RAJA::View<double, Layout<int, PERM_KJI, IMoment, IGroup, IZone>> PHI_VIEW;

  // ell[moment, direction]
  typedef RAJA::View<double, Layout<int, PERM_IJ, IMoment, IDirection>> ELL_VIEW;
};


#ifdef RAJA_USE_OPENMP


// Parallel on zones,  loop nesting: Zones, Groups, Moments, Directions
struct PolLTimesD_OMP {
  // Loops: Moments, Directions, Groups, Zones
  typedef NestedPolicy<ExecList<seq_exec, seq_exec, seq_exec, omp_for_nowait_exec>,
    OMP_Parallel<
      Permute<PERM_LKIJ>
    >
  > EXEC;

  // psi[direction, group, zone]
  typedef RAJA::View<double, Layout<int, PERM_KJI, IDirection, IGroup, IZone>> PSI_VIEW;

  // phi[moment, group, zone]
  typedef RAJA::View<double, Layout<int, PERM_KJI, IMoment, IGroup, IZone>> PHI_VIEW;

  // ell[moment, direction]
  typedef RAJA::View<double, Layout<int, PERM_IJ, IMoment, IDirection>> ELL_VIEW;
};

// Same as D, but with tiling on zones and omp collapse on groups and zones
struct PolLTimesE_OMP {
  // Loops: Moments, Directions, Groups, Zones
  typedef NestedPolicy<ExecList<seq_exec, seq_exec, omp_collapse_nowait_exec, omp_collapse_nowait_exec>,
    OMP_Parallel<
      Tile<TileList<tile_none, tile_none, tile_none, tile_fixed<16>>,
        Permute<PERM_LKIJ,
          Execute // implicit
        >
      >
    >
  > EXEC;

  // psi[direction, group, zone]
  typedef RAJA::View<double, Layout<int, PERM_KJI, IDirection, IGroup, IZone>> PSI_VIEW;

  // phi[moment, group, zone]
  typedef RAJA::View<double, Layout<int, PERM_KJI, IMoment, IGroup, IZone>> PHI_VIEW;

  // ell[moment, direction]
  typedef RAJA::View<double, Layout<int, PERM_IJ, IMoment, IDirection>> ELL_VIEW;
};


#endif



void runLTimesTests(Index_type num_moments, Index_type num_directions, Index_type num_groups, Index_type num_zones){
  runLTimesTest<PolLTimesA>("PolLTimesA", num_moments, num_directions, num_groups, num_zones);
  runLTimesTest<PolLTimesB>("PolLTimesB", num_moments, num_directions, num_groups, num_zones);
  runLTimesTest<PolLTimesC>("PolLTimesC", num_moments, num_directions, num_groups, num_zones);

#ifdef RAJA_USE_OPENMP
  runLTimesTest<PolLTimesD_OMP>("PolLTimesD_OMP", num_moments, num_directions, num_groups, num_zones);
  runLTimesTest<PolLTimesE_OMP>("PolLTimesE_OMP", num_moments, num_directions, num_groups, num_zones);
#endif
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
// Run RAJA::forall nested loop tests...
//
///////////////////////////////////////////////////////////////////////////

   // Run some 2d -> 1d reduction tests
   run2dTests(128,1024);
   run2dTests(37,1);

   // Run some LTimes example tests (directions, groups, zones)
   runLTimesTests(25, 96, 8, 32);
   runLTimesTests(100, 15, 7, 13);

   ///
   /// Print total number of tests passed/run.
   ///
   cout << "\n All Tests : # passed / # run = " 
             << s_ntests_passed_total << " / " 
             << s_ntests_run_total << endl;
                         

//
// Clean up....
//  

   cout << "\n DONE!!! " << endl;

   return 0 ;
}

