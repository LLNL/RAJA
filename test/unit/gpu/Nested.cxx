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

#include <time.h>
#include <cfloat>
#include <cstdlib>

#include <iostream>
#include <string>
#include <vector>

#include "RAJA/RAJA.hxx"

using namespace RAJA;
using namespace std;

#include "Compare.hxx"

typedef struct {
  double val;
  int idx;
} minmaxloc_t;

//
// Global variables for counting tests executed/passed.
//
unsigned s_ntests_run_total = 0;
unsigned s_ntests_passed_total = 0;

unsigned s_ntests_run = 0;
unsigned s_ntests_passed = 0;

// block_size is needed by the reduction variables to setup shared memory
// Care should be used here to cover the maximum block dimensions used by this
// test
const size_t block_size = 256;

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
void runLTimesTest(std::string const &policy,
                   Index_type num_moments,
                   Index_type num_directions,
                   Index_type num_groups,
                   Index_type num_zones)
{
  cout << "\n TestLTimes " << num_moments << " moments, " << num_directions
       << " directions, " << num_groups << " groups, and " << num_zones
       << " zones"
       << " with policy " << policy << endl;

  s_ntests_run++;
  s_ntests_run_total++;

  // allocate data
  // phi is initialized to all zeros, the others are randomized
  std::vector<double> ell_data(num_moments * num_directions);
  std::vector<double> psi_data(num_directions * num_groups * num_zones);
  std::vector<double> phi_data(num_moments * num_groups * num_zones, 0.0);

  // setup CUDA Reduction variables to be exercised
  ReduceSum<cuda_reduce<block_size>, double> pdsum(0.0);
  ReduceMin<cuda_reduce<block_size>, double> pdmin(DBL_MAX);
  ReduceMax<cuda_reduce<block_size>, double> pdmax(-DBL_MAX);
  ReduceMinLoc<cuda_reduce<block_size>, double> pdminloc(DBL_MAX, -1);
  ReduceMaxLoc<cuda_reduce<block_size>, double> pdmaxloc(-DBL_MAX, -1);

  // setup local Reduction variables as a crosscheck
  double lsum = 0.0;
  double lmin = DBL_MAX;
  double lmax = -DBL_MAX;
  minmaxloc_t lminloc = {DBL_MAX, -1};
  minmaxloc_t lmaxloc = {-DBL_MAX, -1};

  //
  // randomize data
  for (size_t i = 0; i < ell_data.size(); ++i) {
    ell_data[i] = drand48();
    //ell_data[i] = 0.0;
  }
  //ell_data[0] = 2.0;

  for (size_t i = 0; i < psi_data.size(); ++i) {
    psi_data[i] = drand48();
    //psi_data[i] = 0.0;
  }
  //psi_data[0] = 5.0;
  // create device memory
  double *d_ell, *d_phi, *d_psi;
  cudaErrchk(cudaMalloc(&d_ell, sizeof(double) * ell_data.size()));
  cudaErrchk(cudaMalloc(&d_phi, sizeof(double) * phi_data.size()));
  cudaErrchk(cudaMalloc(&d_psi, sizeof(double) * psi_data.size()));

  // Copy to device
  cudaMemcpy(d_ell,
             &ell_data[0],
             sizeof(double) * ell_data.size(),
             cudaMemcpyHostToDevice);
  cudaMemcpy(d_phi,
             &phi_data[0],
             sizeof(double) * phi_data.size(),
             cudaMemcpyHostToDevice);
  cudaMemcpy(d_psi,
             &psi_data[0],
             sizeof(double) * psi_data.size(),
             cudaMemcpyHostToDevice);

  // create views on data
  typename POL::ELL_VIEW ell(d_ell, num_moments, num_directions);
  typename POL::PSI_VIEW psi(d_psi, num_directions, num_groups, num_zones);
  typename POL::PHI_VIEW phi(d_phi, num_moments, num_groups, num_zones);

  // get execution policy
  using EXEC = typename POL::EXEC;

  // do calculation using RAJA
  forallN<EXEC, IMoment, IDirection, IGroup, IZone>(
      RangeSegment(0, num_moments),
      RangeSegment(0, num_directions),
      RangeSegment(0, num_groups),
      RangeSegment(0, num_zones),
      [=] __device__(IMoment m, IDirection d, IGroup g, IZone z) {
        // printf("%d,%d,%d,%d\n", *m, *d, *g, *z);
        double val = ell(m, d) * psi(d, g, z);
        phi(m, g, z) += val;
        pdsum += val;
        pdmin.min(val);
        pdmax.max(val);
        int index = *d + (*m * num_directions)
                    + (*g * num_directions * num_moments)
                    + (*z * num_directions * num_moments * num_groups);
        pdminloc.minloc(val, index);
        pdmaxloc.maxloc(val, index);
      });

  cudaDeviceSynchronize();
  // Copy to host the result
  cudaMemcpy(&phi_data[0],
             d_phi,
             sizeof(double) * phi_data.size(),
             cudaMemcpyDeviceToHost);

  // Free CUDA memory
  cudaFree(d_ell);
  cudaFree(d_phi);
  cudaFree(d_psi);

  ////
  //// CHECK ANSWER against the hand-written sequential kernel
  ////
  size_t nfailed = 0;

  // swap to host pointers
  ell.data = &ell_data[0];
  phi.data = &phi_data[0];
  psi.data = &psi_data[0];
  for (IZone z(0); z < num_zones; ++z) {
    for (IGroup g(0); g < num_groups; ++g) {
      for (IMoment m(0); m < num_moments; ++m) {
        double total = 0.0;
        for (IDirection d(0); d < num_directions; ++d) {
          double val = ell(m, d) * psi(d, g, z);
          total += val;
          lmin = RAJA_MIN(lmin, val);
          lmax = RAJA_MAX(lmax, val);
          int index = *d + (*m * num_directions)
                      + (*g * num_directions * num_moments)
                      + (*z * num_directions * num_moments * num_groups);
          minmaxloc_t testMinMaxLoc = {val, index};
          lminloc = RAJA_MINLOC(lminloc, testMinMaxLoc);
          lmaxloc = RAJA_MAXLOC(lmaxloc, testMinMaxLoc);
        }
        lsum += total;
        // check answer with some reasonable tolerance
        if (std::abs(total - phi(m, g, z)) > 1e-12) {
          nfailed++;
        }
      }
    }
  }
  size_t reductionsFailed = 0;
  std::string whichFailed;

  if (std::abs(lsum - double(pdsum)) > 5e-9) {
    reductionsFailed++;
    whichFailed += "[ReduceSum]";
    //printf("ReduceSum failed : EPS =  %g\n",std::abs(lsum - double(pdsum)));
  }

  if (lmin != double(pdmin)) {
    reductionsFailed++;
    whichFailed += "[ReduceMin]";
  }

  if (lmax != double(pdmax)) {
    reductionsFailed++;
    whichFailed += "[ReduceMax]";
  }

  if ((lminloc.val != double(pdminloc)) && (lminloc.idx != pdminloc.getLoc())) {
    reductionsFailed++;
    whichFailed += "[ReduceMinLoc]";
  }

  if ((lmaxloc.val != double(pdmaxloc)) && (lmaxloc.idx != pdmaxloc.getLoc())) {
    reductionsFailed++;
    whichFailed += "[ReduceMaxLoc]";
  }

  if (nfailed || reductionsFailed) {
    cout << "\n TEST FAILURE: " << nfailed << " elements failed" << endl;
    if (reductionsFailed) {
      cout << "  REDUCTIONS FAILURE: " << whichFailed << endl;
    }
  } else {
    s_ntests_passed++;
    s_ntests_passed_total++;
  }
}

// Use thread-block mappings
struct PolLTimesA_GPU {
  // Loops: Moments, Directions, Groups, Zones
  typedef NestedPolicy<ExecList<seq_exec,
                                seq_exec,
                                cuda_threadblock_x_exec<32>,
                                cuda_threadblock_y_exec<32>>>
      EXEC;

  // psi[direction, group, zone]
  typedef RAJA::View<double, Layout<int, PERM_IJK, IDirection, IGroup, IZone>>
      PSI_VIEW;

  // phi[moment, group, zone]
  typedef RAJA::View<double, Layout<int, PERM_IJK, IMoment, IGroup, IZone>>
      PHI_VIEW;

  // ell[moment, direction]
  typedef RAJA::View<double, Layout<int, PERM_IJ, IMoment, IDirection>>
      ELL_VIEW;
};

// Use thread and block mappings
struct PolLTimesB_GPU {
  // Loops: Moments, Directions, Groups, Zones
  typedef NestedPolicy<ExecList<seq_exec,
                                seq_exec,
                                cuda_thread_z_exec,
                                cuda_block_y_exec>,
                       Permute<PERM_JILK>>
      EXEC;

  // psi[direction, group, zone]
  typedef RAJA::View<double, Layout<int, PERM_IJK, IDirection, IGroup, IZone>>
      PSI_VIEW;

  // phi[moment, group, zone]
  typedef RAJA::View<double, Layout<int, PERM_IJK, IMoment, IGroup, IZone>>
      PHI_VIEW;

  // ell[moment, direction]
  typedef RAJA::View<double, Layout<int, PERM_IJ, IMoment, IDirection>>
      ELL_VIEW;
};

// Combine OMP Parallel, omp nowait, and cuda thread-block launch
struct PolLTimesC_GPU {
  // Loops: Moments, Directions, Groups, Zones
  typedef NestedPolicy<ExecList<seq_exec,
                                seq_exec,
                                omp_for_nowait_exec,
                                cuda_threadblock_y_exec<32>>,
                       OMP_Parallel<>>
      EXEC;

  // psi[direction, group, zone]
  typedef RAJA::View<double, Layout<int, PERM_IJK, IDirection, IGroup, IZone>>
      PSI_VIEW;

  // phi[moment, group, zone]
  typedef RAJA::View<double, Layout<int, PERM_IJK, IMoment, IGroup, IZone>>
      PHI_VIEW;

  // ell[moment, direction]
  typedef RAJA::View<double, Layout<int, PERM_IJ, IMoment, IDirection>>
      ELL_VIEW;
};

void runLTimesTests(Index_type num_moments,
                    Index_type num_directions,
                    Index_type num_groups,
                    Index_type num_zones)
{
  runLTimesTest<PolLTimesA_GPU>(
      "PolLTimesA_GPU", num_moments, num_directions, num_groups, num_zones);
  runLTimesTest<PolLTimesB_GPU>(
      "PolLTimesB_GPU", num_moments, num_directions, num_groups, num_zones);
  runLTimesTest<PolLTimesC_GPU>(
      "PolLTimesC_GPU", num_moments, num_directions, num_groups, num_zones);
}

void runNegativeRange()
{
  s_ntests_run++;
  s_ntests_run_total++;

  cout << "\n TestNegativeRange " << endl;

  double *data;
  double host_data[100];

  cudaMallocManaged((void **)&data, sizeof(double) * 100, cudaMemAttachGlobal);

  for (int i = 0; i < 100; ++i) {
    host_data[i] = i * 1.0;
  }

  forallN<NestedPolicy<ExecList<cuda_threadblock_y_exec<16>,
                                cuda_threadblock_x_exec<16>>>>(
      RangeSegment(-2, 8), RangeSegment(-2, 8), [=] RAJA_DEVICE(int k, int j) {
        const int idx = ((k - -2) * 10) + (j - -2);
        data[idx] = idx * 1.0;
      });

  cudaDeviceSynchronize();

  size_t nfailed = 0;

  for (int i = 0; i < 100; ++i) {
    if (host_data[i] != data[i]) {
      nfailed++;
    }
  }

  if (nfailed) {
    cout << "\n TEST FAILURE: " << nfailed << " elements failed" << endl;
  } else {
    s_ntests_passed++;
    s_ntests_passed_total++;
  }
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
  cout << "Starting GPU nested tests" << endl << endl;
  // Run some LTimes example tests (directions, groups, zones)
  runLTimesTests(2, 0, 7, 3);
  runLTimesTests(2, 3, 7, 3);
  runLTimesTests(2, 3, 32, 4);
  runLTimesTests(25, 96, 8, 32);
  runLTimesTests(100, 15, 7, 13);
  runNegativeRange();

  ///
  /// Print total number of tests passed/run.
  ///

  cout << "\n All Tests : # passed / # run = " << s_ntests_passed_total << " / "
       << s_ntests_run_total << endl;

  //
  // Clean up....
  //

  cout << "\n DONE!!! " << endl;

  return !(s_ntests_passed_total == s_ntests_run_total);
}
