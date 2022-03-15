//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-22, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <vector>

#include "RAJA/RAJA.hpp"

#include "camp/resource.hpp"

#include "memoryManager.hpp"

/*
 *  EXERCISE #3: Mesh vertex area with "colored" TypedIndexSet
 *
 *  In this exercise, you will use a RAJA TypedIndexSet containing 4 
 *  ListSegments to parallelize the mesh vertex area computation.
 *  A sum is computed at each vertex on a logically-Cartesian 2D mesh
 *  where the sum represents the vertex "area" as an average of the 4
 *  element areas surrounding the vertex. The computation is written as
 *  a loop over elements. To avoid data races, where multiple element
 *  contributions may be written to the same vertex value at the same time,
 *  the elements are partitioned into 4 subsets, where no two elements in
 *  each subset share a vertex. A ListSegment enumerates the elements in
 *  each subset. When the ListSegments are put into an TypedIndexSet, the entire
 *  computation can be executed with one RAJA::forall() statement, where
 *  you iterate over the segments sequentially and execute each segment in
 *  parallel. This exercise illustrates how RAJA can be used to enable one 
 *  to get some parallelism from such operations without fundamentally
 *  changing the way the algorithm looks in source code.
 *
 *  This file contains sequential and OpenMP variants of the vertex area
 *  computation using C-style for-loops. You will fill in RAJA versions of 
 *  these variants, plus a RAJA CUDA version if you have access to an NVIDIA 
 *  GPU and a CUDA compiler, in empty code sections indicated by comments.
 *
 *  RAJA features you will use:
 *    - `forall` loop iteration template method
 *    -  Index list segment
 *    -  TypedIndexSet segment container
 *    -  Hierarchical execution policies
 *
 * If CUDA is enabled, CUDA unified memory is used.
 */

/*
  CUDA_BLOCK_SIZE - specifies the number of threads in a CUDA thread block

                    Uncomment to use when filling in exercises.

#if defined(RAJA_ENABLE_CUDA)
const int CUDA_BLOCK_SIZE = 256;
#endif
*/

//
// Functions to check and print result.
//
void checkResult(double* a, double* aref, int n);
void printMeshData(double* v, int n, int joff);


int main(int RAJA_UNUSED_ARG(argc), char **RAJA_UNUSED_ARG(argv[]))
{

  std::cout << "\n\nExercise #3: Mesh vertex area with 'colored' TypedIndexSet...\n";

//
// 2D mesh has N^2 elements (N+1)^2 vertices.
//
  const int N = 1000;
  const int Nelem = N;
  const int Nvert = N + 1;
  double* areae = memoryManager::allocate<double>(Nelem*Nelem);
  double* areav = memoryManager::allocate<double>(Nvert*Nvert);
  double* areav_ref = memoryManager::allocate<double>(Nvert*Nvert);
  int* e2v_map = memoryManager::allocate<int>(4*Nelem*Nelem);

//
// Define mesh spacing factor 'h' and set up elem to vertex mapping array.
//
  double h = 0.1;

  for (int j = 0 ; j < Nelem ; ++j) {
    for (int i = 0 ; i < Nelem ; ++i) {
      int ielem = i + j*Nelem ;
      int imap = 4 * ielem ;
      e2v_map[imap] = ielem + j;
      e2v_map[imap+1] = ielem + j + 1;
      e2v_map[imap+2] = ielem + j + Nvert;
      e2v_map[imap+3] = ielem + j + 1 + Nvert;
    }
  }

//
// Initialize element areas so each element area 
// depends on the i,j coordinates of the element.
//
  std::memset(areae, 0, Nelem*Nelem * sizeof(double));

  for (int j = 0 ; j < Nelem ; ++j) {
    for (int i = 0 ; i < Nelem ; ++i) {
      int ielem = i + j*Nelem ;
      areae[ielem] = h*(i+1) * h*(j+1);
    }
  }

//std::cout << "\n Element areas...\n";
//printMeshData(areae, Nelem, Nelem);

//----------------------------------------------------------------------------//
// C-style sequential variant establishes reference solution to compare with.
//----------------------------------------------------------------------------//

  std::cout << "\n Running sequential C-style version of vertex sum...\n";

  std::memset(areav_ref, 0, Nvert*Nvert * sizeof(double));

  for (int ie = 0; ie < Nelem*Nelem; ++ie) {
      int* iv = &(e2v_map[4*ie]);
      areav_ref[ iv[0] ] += areae[ie] / 4.0 ;
      areav_ref[ iv[1] ] += areae[ie] / 4.0 ;
      areav_ref[ iv[2] ] += areae[ie] / 4.0 ;
      areav_ref[ iv[3] ] += areae[ie] / 4.0 ;
  }

//std::cout << "\n Vertex areas (reference)...\n";
//printMeshData(areav_ref, Nvert, jvoff);


//----------------------------------------------------------------------------//
//
// In the following, we partition the element iteration space into four
// subsets (or "colors") indicated by numbers in the figure below. 
// 
//    -----------------
//    | 2 | 3 | 2 | 3 |
//    -----------------
//    | 0 | 1 | 0 | 1 |
//    -----------------
//    | 2 | 3 | 2 | 3 |
//    -----------------
//    | 0 | 1 | 0 | 1 |   
//    -----------------
//
// Since none of the elements with the same number share a common vertex,
// we can iterate over each subset ("color") in parallel.
//
// We use RAJA ListSegments and a RAJA TypedIndexSet to define the element 
// partitioning. 
//

//
// Gather the element indices for each color in a vector.
//
  std::vector< std::vector<int> > idx(4);

  for (int j = 0 ; j < Nelem ; ++j) {
    for (int i = 0 ; i < Nelem ; ++i) {
      int ie = i + j*Nelem ;
      if ( i % 2 == 0 ) {
        if ( j % 2 == 0 ) {
          idx[0].push_back(ie);
        } else {
          idx[2].push_back(ie);  
        }
      } else {
        if ( j % 2 == 0 ) {
          idx[1].push_back(ie);
        } else {
          idx[3].push_back(ie);
        }
      }
    }
  }


//----------------------------------------------------------------------------//
// C-style OpenMP multithreading variant. Note that we use the vectors
// defined above in this variant to run each element subset in parallel. 
//----------------------------------------------------------------------------//

#if defined(RAJA_ENABLE_OPENMP)

  std::cout << "\n Running C-style OpenMP vertex sum...\n";

  std::memset(areav, 0, Nvert*Nvert * sizeof(double));

  for (int icol = 0; icol < 4; ++icol) {
     const std::vector<int>& ievec = idx[icol];
     const int len = static_cast<int>(ievec.size());

     #pragma omp parallel for  
     for (int i = 0; i < len; ++i) {
        int ie = ievec[i]; 
        int* iv = &(e2v_map[4*ie]);
        areav[ iv[0] ] += areae[ie] / 4.0 ;
        areav[ iv[1] ] += areae[ie] / 4.0 ;
        areav[ iv[2] ] += areae[ie] / 4.0 ;
        areav[ iv[3] ] += areae[ie] / 4.0 ;
     }

  }

  checkResult(areav, areav_ref, Nvert);
//std::cout << "\n Vertex areas (reference)...\n";
//printMeshData(areav_ref, Nvert, jvoff);

#endif


// The TypedIndexSet is a variadic template, where the template arguments
// are the segment types that the TypedIndexSet can hold.
//
#if defined(RAJA_ENABLE_OPENMP) || defined(RAJA_ENABLE_CUDA)
#if 0  // needed for exercises, but if-def'd out to quiet compiler warnings.
  using SegmentType = RAJA::TypedListSegment<int>;
#endif
#endif

#if defined(RAJA_ENABLE_OPENMP)

//
// Resource object used to construct list segment objects with indices
// living in host (CPU) memory.
//
  camp::resources::Resource host_res{camp::resources::Host()};

//
// Create a RAJA TypedIndexSet with four ListSegments, one for the indices of
// the elements in each subsut. This will be used in the RAJA OpenMP and CUDA
// variants of the vertex sum calculation.
#if 0  // needed for exercises, but if-def'd out to quiet compiler warnings.
  RAJA::TypedIndexSet<SegmentType> colorset;
#endif

  ///
  /// TODO...
  ///
  /// EXERCISE: Add four SegmentType objects to the coloret, one for each of 
  ///           the 'idx' arrays above. Remember to pass the 'host_res'
  ///           object to the SegmentType constructor. 
  ///


//----------------------------------------------------------------------------//
// RAJA OpenMP vertex sum calculation using TypedIndexSet (sequential iteration 
// over segments, OpenMP parallel iteration of each segment)
//----------------------------------------------------------------------------//

  std::cout << "\n Running RAJA OpenMP index set vertex sum...\n";

  std::memset(areav, 0, Nvert*Nvert * sizeof(double));

  ///
  /// TODO...
  ///
  /// EXERCISE: Implement the vertex sum kernel a RAJA::forall
  ///           method with execution policy type 
  ///
  ///            RAJA::ExecPolicy<RAJA::seq_segit, 
  ///                             RAJA::omp_parallel_for_exec>
  ///
  ///            so that the kernel iterates over the segments sequentially
  ///            and executes each segment in parallel using OpenMP.
  ///


  checkResult(areav, areav_ref, Nvert);
//std::cout << "\n Vertex volumes...\n";
//printMeshData(areav, Nvert, Nvert); 

#endif


//----------------------------------------------------------------------------//
// RAJA CUDA vertex sum calculation using TypedIndexSet (sequential iteration 
// over segments, CUDA kernel launched for each segment)
//----------------------------------------------------------------------------//

#if defined(RAJA_ENABLE_CUDA)

//
// Resource object used to construct list segment objects with indices
// living in host (CPU) memory.
//
  camp::resources::Resource cuda_res{camp::resources::Cuda()};

#if 0  // needed for exercises, but if-def'd out to quiet compiler warnings.
  RAJA::TypedIndexSet<SegmentType> cuda_colorset;
#endif

  ///
  /// TODO...
  ///
  /// EXERCISE: Add four SegmentType objects to the cuda_coloret, one for 
  ///           each of the 'idx' arrays above. Remember to pass the 'cuda_res'
  ///           object to the SegmentType constructor.
  ///


  ///
  /// TODO...
  ///
  /// EXERCISE: Implement the vertex sum kernel a RAJA::forall
  ///           method with execution policy type
  ///
  ///            RAJA::ExecPolicy<RAJA::seq_segit,
  ///                             RAJA::cuda_exec<CUDA_BLOCK_SIZE>>
  ///
  ///            so that the kernel iterates over the segments sequentially
  ///            and executes each segment in parallel as a CUDA kernel.


  std::cout << "\n Running RAJA CUDA index set vertex sum...\n";

  std::memset(areav, 0, Nvert*Nvert * sizeof(double));

  ///
  /// TODO...
  ///
  /// EXERCISE: Implement the vertex sum kernel a RAJA::forall
  ///           method with execution policy type 
  ///
  ///            RAJA::ExecPolicy<RAJA::seq_segit, 
  ///                             RAJA::cuda_exec<CUDA_BLOCK_SIZE>>
  ///
  ///            so that the kernel iterates over the segments sequentially
  ///            and executes each segment in parallel as a CUDA kernel.
  ///


  checkResult(areav, areav_ref, Nvert);
//std::cout << "\n Vertex volumes...\n";
//printMeshData(areav, Nvert, jvoff);

#endif

//----------------------------------------------------------------------------//

  // Clean up...
  memoryManager::deallocate(areae);
  memoryManager::deallocate(areav);
  memoryManager::deallocate(areav_ref);
  memoryManager::deallocate(e2v_map);

  std::cout << "\n DONE!...\n";

  return 0;
}

//
// Function to compare result to reference and print result P/F.
//
void checkResult(double* a, double* aref, int n)
{
  bool correct = true;
  for (int i = 0; i < n*n; i++) {
    if ( correct && std::abs(a[i] - aref[i]) > 10e-12 ) { correct = false; }
  }
  if ( correct ) {
    std::cout << "\n\t result -- PASS\n";
  } else {
    std::cout << "\n\t result -- FAIL\n";
  }
}

//
// Function to print mesh data with mesh indices.
//
void printMeshData(double* v, int n, int joff)
{
  std::cout << std::endl;
  for (int j = 0 ; j < n ; ++j) {
    for (int i = 0 ; i < n ; ++i) {
      int ii = i + j*joff ;
      std::cout << "v(" << i << "," << j << ") = "
                << v[ii] << std::endl;
    }
  }
  std::cout << std::endl;
}
