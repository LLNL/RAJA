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

#include "memoryManager.hpp"

#include "RAJA/RAJA.hpp"

#include "camp/resource.hpp"

/*
 *  Mesh Vertex Sum with Index Coloring Example
 *
 *  Example computes a sum at each vertex on a logically-Cartesian
 *  2D mesh. Each sum includes a contribution from each mesh element
 *  that share a vertex. In many "staggered mesh" applications, such
 *  operations are written in a way that prevents parallelization due
 *  to potential data races -- specifically, multiple loop iterates 
 *  over mesh elements writing to the same shared vertex memory location.
 *  This example illustrates how RAJA contructs can be used to enable one 
 *  to get some parallelism from such operations without fundamentally
 *  changing how the algorithm looks in source code.
 *
 *  RAJA features shown:
 *    - `forall` loop iteration template method
 *    -  Index list segment
 *    -  TypedIndexSet segment container
 *    -  Hierarchical execution policies
 *
 * If CUDA is enabled, CUDA unified memory is used.
 */

/*
  CUDA_BLOCK_SIZE - specifies the number of threads in a CUDA thread block
*/
#if defined(RAJA_ENABLE_CUDA)
const int CUDA_BLOCK_SIZE = 256;
#endif

#if defined(RAJA_ENABLE_HIP)
const int HIP_BLOCK_SIZE = 256;
#endif

//
// Functions to check and print result.
//
void checkResult(double* vol, double* volref, int n);
void printMeshData(double* v, int n, int joff);


int main(int RAJA_UNUSED_ARG(argc), char **RAJA_UNUSED_ARG(argv[]))
{

  std::cout << "\n\nRAJA mesh vertex sum example...\n";

//
// 2D mesh has N^2 "interior" vertices, (N+2)^2 total vertices and
// (N+1)^2 elements (including "ghost" elems)
//
  const int N = 1000;
  const int N_elem = N + 1;
  const int N_vert = N + 2;
  double* elemvol = memoryManager::allocate<double>(N_elem*N_elem);
  double* vertexvol = memoryManager::allocate<double>(N_vert*N_vert);
  double* vertexvol_ref = memoryManager::allocate<double>(N_vert*N_vert);
  int* elem2vert_map = memoryManager::allocate<int>(4*N_elem*N_elem);

//
// Some basic mesh parameters (offsets, mesh spacing factor 'h'),
// set up elem to vertex mapping array.
//
  int jeoff = N_elem;

  int jvoff = N_vert;

  double h = 0.1;

  for (int j = 0 ; j < N_elem ; ++j) {
    for (int i = 0 ; i < N_elem ; ++i) {
      int ielem = i + j*jeoff ;
      int imap = 4 * ielem ;
      elem2vert_map[imap] = ielem + j;
      elem2vert_map[imap+1] = ielem + j + 1;
      elem2vert_map[imap+2] = ielem + j + jvoff;
      elem2vert_map[imap+3] = ielem + j + 1 + jvoff;
    }
  }

//
// Initialize hexahedral element volumes so every element volume 
// depends on its i,j coordinates. 
//
  std::memset(elemvol, 0, N_elem*N_elem * sizeof(double));

  for (int j = 0 ; j < N_elem ; ++j) {
    for (int i = 0 ; i < N_elem ; ++i) {
      int ielem = i + j*jeoff ;
      elemvol[ielem] = h*(i+1) * h*(j+1);
    }
  }

//std::cout << "\n Element volumes...\n";
//printMeshData(elemvol, N_elem, jeoff);

//----------------------------------------------------------------------------//

  std::cout << "\n Running C-version of vertex sum...\n";

  std::memset(vertexvol_ref, 0, N_vert*N_vert * sizeof(double));

  // _cstyle_vertexsum_start
  for (int j = 0 ; j < N_elem ; ++j) {
    for (int i = 0 ; i < N_elem ; ++i) {
      int ie = i + j*jeoff ;
      int* iv = &(elem2vert_map[4*ie]);
      vertexvol_ref[ iv[0] ] += elemvol[ie] / 4.0 ;
      vertexvol_ref[ iv[1] ] += elemvol[ie] / 4.0 ;
      vertexvol_ref[ iv[2] ] += elemvol[ie] / 4.0 ;
      vertexvol_ref[ iv[3] ] += elemvol[ie] / 4.0 ;
    }
  }
  // _cstyle_vertexsum_end

//std::cout << "\n Vertex volumes (reference)...\n";
//printMeshData(vertexvol_ref, N_vert, jvoff);


//----------------------------------------------------------------------------//

  std::cout << "\n Running RAJA nested sequential version...\n";

  std::memset(vertexvol, 0, N_vert*N_vert * sizeof(double));

  // _raja_seq_vertexsum_start
  using EXEC_POL1 = 
    RAJA::KernelPolicy< 
      RAJA::statement::For<1, RAJA::seq_exec,    // j
        RAJA::statement::For<0, RAJA::seq_exec,  // i
          RAJA::statement::Lambda<0>
        > 
      > 
    >;

  RAJA::kernel<EXEC_POL1>( RAJA::make_tuple(RAJA::RangeSegment(0, N_elem),
                                            RAJA::RangeSegment(0, N_elem)),
    [=](int i, int j) {
      int ie = i + j*jeoff ;
      int* iv = &(elem2vert_map[4*ie]);
      vertexvol[ iv[0] ] += elemvol[ie] / 4.0 ;
      vertexvol[ iv[1] ] += elemvol[ie] / 4.0 ;
      vertexvol[ iv[2] ] += elemvol[ie] / 4.0 ;
      vertexvol[ iv[3] ] += elemvol[ie] / 4.0 ;
  });
  // _raja_seq_vertexsum_end

  checkResult(vertexvol, vertexvol_ref, N_vert);
//std::cout << "\n Vertex volumes...\n";
//printMeshData(vertexvol, N_vert, jvoff);

//----------------------------------------------------------------------------//

//
// Note that the C-style and RAJA versions of the vertex sum calculation
// above cannot safely execute in parallel due to potential data races;
// i.e., multiple loop iterates over mesh elements writing to the same 
// shared vertex memory location.
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
// First, gather the element indices for each color in a vector.
//
  // _colorvectors_vertexsum_start 
  std::vector<int> idx0;
  std::vector<int> idx1;
  std::vector<int> idx2;
  std::vector<int> idx3;

  for (int j = 0 ; j < N_elem ; ++j) {
    for (int i = 0 ; i < N_elem ; ++i) {
      int ie = i + j*jeoff ;
      if ( i % 2 == 0 ) {
        if ( j % 2 == 0 ) {
          idx0.push_back(ie);
        } else {
          idx2.push_back(ie);  
        }
      } else {
        if ( j % 2 == 0 ) {
          idx1.push_back(ie);
        } else {
          idx3.push_back(ie);
        }
      }
    }
  }
  // _colorvectors_vertexsum_end
 
// 
// Second, create a RAJA TypedIndexSet with four ListSegments
//
// The TypedIndexSet is a variadic template, where the template arguments
// are the segment types that the TypedIndexSet can hold. 
// 
  // _colorindexset_vertexsum_start
  using SegmentType = RAJA::TypedListSegment<int>;

  RAJA::TypedIndexSet<SegmentType> colorset;

  camp::resources::Resource host_res{camp::resources::Host()};

  colorset.push_back( SegmentType(&idx0[0], idx0.size(), host_res) ); 
  colorset.push_back( SegmentType(&idx1[0], idx1.size(), host_res) ); 
  colorset.push_back( SegmentType(&idx2[0], idx2.size(), host_res) ); 
  colorset.push_back( SegmentType(&idx3[0], idx3.size(), host_res) ); 
  // _colorindexset_vertexsum_end

//----------------------------------------------------------------------------//
 
//
// RAJA vertex volume calculation - sequential TypedIndexSet version 
// (sequential iteration over segments, 
//  sequential iteration of each segment)
//
// NOTE: we do not need i,j indices for this version since the element
//       indices are contained in the list segments
//
  std::cout << "\n Running RAJA sequential index set version...\n";

  std::memset(vertexvol, 0, N_vert*N_vert * sizeof(double));

  // _raja_seq_colorindexset_vertexsum_start
  using EXEC_POL2 = RAJA::ExecPolicy<RAJA::seq_segit, 
                                     RAJA::seq_exec>;

  RAJA::forall<EXEC_POL2>(colorset, [=](int ie) {
    int* iv = &(elem2vert_map[4*ie]);
    vertexvol[ iv[0] ] += elemvol[ie] / 4.0 ;
    vertexvol[ iv[1] ] += elemvol[ie] / 4.0 ;
    vertexvol[ iv[2] ] += elemvol[ie] / 4.0 ;
    vertexvol[ iv[3] ] += elemvol[ie] / 4.0 ;
  });
  // _raja_seq_colorindexset_vertexsum_end

  checkResult(vertexvol, vertexvol_ref, N_vert);
//std::cout << "\n Vertex volumes...\n";
//printMeshData(vertexvol, N_vert, jvoff);

//----------------------------------------------------------------------------//

#if defined(RAJA_ENABLE_OPENMP)
//
// RAJA vertex volume calculation - OpenMP TypedIndexSet version
// (sequential iteration over segments, 
//  OpenMP parallel iteration of each segment)
//
  std::cout << "\n Running RAJA OpenMP index set version...\n";

  std::memset(vertexvol, 0, N_vert*N_vert * sizeof(double));

  // _raja_openmp_colorindexset_vertexsum_start
  using EXEC_POL3 = RAJA::ExecPolicy<RAJA::seq_segit, 
                                     RAJA::omp_parallel_for_exec>;

  RAJA::forall<EXEC_POL3>(colorset, [=](int ie) {
    int* iv = &(elem2vert_map[4*ie]);
    vertexvol[ iv[0] ] += elemvol[ie] / 4.0 ;
    vertexvol[ iv[1] ] += elemvol[ie] / 4.0 ;
    vertexvol[ iv[2] ] += elemvol[ie] / 4.0 ;
    vertexvol[ iv[3] ] += elemvol[ie] / 4.0 ;
  });
  // _raja_openmp_colorindexset_vertexsum_end

  checkResult(vertexvol, vertexvol_ref, N_vert);
//std::cout << "\n Vertex volumes...\n";
//printMeshData(vertexvol, N_vert, jvoff); 
#endif

//----------------------------------------------------------------------------//

#if defined(RAJA_ENABLE_CUDA)
//
// RAJA vertex volume calculation - CUDA TypedIndexSet version
// (sequential iteration over segments, 
//  CUDA parallel execution of each segment)
//
  std::cout << "\n Running RAJA CUDA index set version...\n";

//
// We create a RAJA TypedIndexSet with four ListSegments as before,
// but now we use a CUDA resource so the segment indices live in
// device memory.
//
  RAJA::TypedIndexSet<SegmentType> colorset_cuda;

  camp::resources::Resource cuda_res{camp::resources::Cuda()};

  colorset_cuda.push_back( SegmentType(&idx0[0], idx0.size(), cuda_res) );
  colorset_cuda.push_back( SegmentType(&idx1[0], idx1.size(), cuda_res) );
  colorset_cuda.push_back( SegmentType(&idx2[0], idx2.size(), cuda_res) );
  colorset_cuda.push_back( SegmentType(&idx3[0], idx3.size(), cuda_res) );

  std::memset(vertexvol, 0, N_vert*N_vert * sizeof(double));

  // _raja_cuda_colorindexset_vertexsum_start
  using EXEC_POL4 = RAJA::ExecPolicy<RAJA::seq_segit, 
                                     RAJA::cuda_exec<CUDA_BLOCK_SIZE>>;

  RAJA::forall<EXEC_POL4>(colorset_cuda, [=] RAJA_DEVICE (int ie) {
    int* iv = &(elem2vert_map[4*ie]);
    vertexvol[ iv[0] ] += elemvol[ie] / 4.0 ;
    vertexvol[ iv[1] ] += elemvol[ie] / 4.0 ;
    vertexvol[ iv[2] ] += elemvol[ie] / 4.0 ;
    vertexvol[ iv[3] ] += elemvol[ie] / 4.0 ;
  });
  // _raja_cuda_colorindexset_vertexsum_end

  checkResult(vertexvol, vertexvol_ref, N_vert);
//std::cout << "\n Vertex volumes...\n";
//printMeshData(vertexvol, N_vert, jvoff);
#endif

//----------------------------------------------------------------------------//

#if defined(RAJA_ENABLE_HIP)
//
// RAJA vertex volume calculation - HIP IndexSet version
// (sequential iteration over segments,
//  HIP parallel execution of each segment)
//
  double* d_elemvol    = memoryManager::allocate_gpu<double>(N_elem*N_elem);
  double* d_vertexvol  = memoryManager::allocate_gpu<double>(N_vert*N_vert);
  int* d_elem2vert_map = memoryManager::allocate_gpu<int>(4*N_elem*N_elem);

  hipMemcpy(d_elemvol, elemvol, N_elem*N_elem*sizeof(double), hipMemcpyHostToDevice);
  hipMemcpy(d_elem2vert_map, elem2vert_map, 4*N_elem*N_elem*sizeof(int), hipMemcpyHostToDevice);

  std::cout << "\n Running RAJA HIP index set version...\n";

  std::memset(vertexvol, 0, N_vert*N_vert * sizeof(double));
  hipMemcpy(d_vertexvol, vertexvol, N_vert*N_vert*sizeof(double), hipMemcpyHostToDevice);

//
// We create a RAJA TypedIndexSet with four ListSegments as before,
// but now we use a Hip resource so the segment indices live in
// device memory.
//
  RAJA::TypedIndexSet<SegmentType> colorset_hip;

  camp::resources::Resource hip_res{camp::resources::Hip()};

  colorset_hip.push_back( SegmentType(&idx0[0], idx0.size(), hip_res) );
  colorset_hip.push_back( SegmentType(&idx1[0], idx1.size(), hip_res) );
  colorset_hip.push_back( SegmentType(&idx2[0], idx2.size(), hip_res) );
  colorset_hip.push_back( SegmentType(&idx3[0], idx3.size(), hip_res) );

  // _raja_hip_colorindexset_vertexsum_start
  using EXEC_POL4 = RAJA::ExecPolicy<RAJA::seq_segit,
                                     RAJA::hip_exec<HIP_BLOCK_SIZE>>;

  RAJA::forall<EXEC_POL4>(colorset_hip, [=] RAJA_DEVICE (int ie) {
    int* iv = &(d_elem2vert_map[4*ie]);
    d_vertexvol[ iv[0] ] += d_elemvol[ie] / 4.0 ;
    d_vertexvol[ iv[1] ] += d_elemvol[ie] / 4.0 ;
    d_vertexvol[ iv[2] ] += d_elemvol[ie] / 4.0 ;
    d_vertexvol[ iv[3] ] += d_elemvol[ie] / 4.0 ;
  });
  // _raja_hip_colorindexset_vertexsum_end

  hipMemcpy(vertexvol, d_vertexvol, N_vert*N_vert*sizeof(double), hipMemcpyDeviceToHost);
  checkResult(vertexvol, vertexvol_ref, N_vert);

  memoryManager::deallocate_gpu(d_elemvol);
  memoryManager::deallocate_gpu(d_vertexvol);
  memoryManager::deallocate_gpu(d_elem2vert_map);
//std::cout << "\n Vertex volumes...\n";
//printMeshData(vertexvol, N_vert, jvoff);
#endif

//----------------------------------------------------------------------------//

  // Clean up...
  memoryManager::deallocate(elemvol);
  memoryManager::deallocate(vertexvol);
  memoryManager::deallocate(vertexvol_ref);
  memoryManager::deallocate(elem2vert_map);

  std::cout << "\n DONE!...\n";

  return 0;
}

//
// Function to compare result to reference and print result P/F.
//
void checkResult(double* vol, double* volref, int n)
{
  bool match = true;
  for (int i = 0; i < n*n; i++) {
    if ( std::abs(vol[i] - volref[i]) > 10e-12 ) { match = false; }
  }
  if ( match ) {
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
