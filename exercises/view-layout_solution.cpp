//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-22, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include <cstdlib>
#include <iostream>
#include <cstring>
#include <cmath>

#include "memoryManager.hpp"

#include "RAJA/RAJA.hpp"

/*
 *  View and Layout Exercise
 *
 *  Examples illustrate the use of RAJA View and Layout types.
 *
 *  RAJA features shown:
 *    - RAJA::View
 *    - RAJA::Layout
 *    - Layout permutations 
 *    - OffsetLayout
 *    - OffsetLayout permutations 
 */

//
// Functions to check and print arrays
//
template <typename T>
void checkResult(T* C, T* Cref, int N);

template <typename T>
void printValues(T* C, int N);

int main(int RAJA_UNUSED_ARG(argc), char **RAJA_UNUSED_ARG(argv[]))
{

  std::cout << "\n\nRAJA view & layout exercises...\n";

//----------------------------------------------------------------------------//
//
// Matrix-matrix multiplication: default layout
//
//----------------------------------------------------------------------------//

  //
  // Define dimensionality (2) and size N x N of matrices.
  //
  constexpr int Matrix_DIM = 2;
  constexpr int N = 4;

  //
  // Allocate storage for matrices and initialize matrix entries
  //
  double *A = new double[ N * N ];
  double *B = new double[ N * N ];
  double *C = new double[ N * N ];
  double *Cref = new double[ N * N ];

  for (int row = 0; row < N; ++row) {
    for (int col = 0; col < N; ++col) {
      A[ col + N*row ] = row + 1;
      B[ col + N*row ] = col + 1;
      C[ col + N*row ] = 0.0;
      Cref[ col + N*row ] = 0.0;
    }
  }

//printValues<double>(A, N*N); 
//printValues<double>(B, N*N); 
//printValues<double>(C, N*N); 
//printValues<double>(Cref, N*N); 

//----------------------------------------------------------------------------//

  std::cout << "\n Running matrix multiplication reference solution...\n";

  // _cstyle_matmult_start
  for (int row = 0; row < N; ++row) {
    for (int col = 0; col < N; ++col) {
      for (int k = 0; k < N; ++k) {
        Cref[col + N*row] += A[k + N*row] * B[col + N*k];
      }
    }
  }
  // _cstyle_matmult_end

//printValues<double>(Cref, N*N);


//----------------------------------------------------------------------------//

  std::cout << "\n Running matrix multiplication w/Views...\n";

  // 
  // Define RAJA View objects to simplify access to the matrix entries.
  // 
  // Note: we use default Layout 
  //
  // _matmult_views_start
  RAJA::View< double, RAJA::Layout<Matrix_DIM, int> > Aview(A, N, N);
  RAJA::View< double, RAJA::Layout<Matrix_DIM, int> > Bview(B, N, N);
  RAJA::View< double, RAJA::Layout<Matrix_DIM, int> > Cview(C, N, N);
  // _matmult_views_end

  // _cstyle_matmult_views_start
  for (int row = 0; row < N; ++row) {
    for (int col = 0; col < N; ++col) {
      for (int k = 0; k < N; ++k) {
        Cview(row, col) += Aview(row, k) * Bview(k, col);
      }
    }
  }
  // _cstyle_matmult_views_end

  checkResult<double>(C, Cref, N*N);
//printValues<double>(C, N*N);

//
// Clean up.
//
  delete [] A;
  delete [] B;
  delete [] C;
  delete [] Cref;

//----------------------------------------------------------------------------//
//
// Default layouts use row-major data ordering
//
//----------------------------------------------------------------------------//

  //
  // Define some dimensions and allocate an array
  //
  // _default_views_init_start
  constexpr int Nx = 3;
  constexpr int Ny = 5;
  constexpr int Nz = 2;
  constexpr int Ntot  = Nx*Ny*Nz;
  int* a = new int[ Ntot ];
  int* aref = new int[ Ntot ];

  int iter{0};
  for(int i = 0; i < Ntot; ++i)
  {
    aref[iter] = i;
    ++iter;
  }
  // _default_views_init_start

//printValues<int>(ref, Ntot);

  std::cout << "\n Running default view cases...\n";

  std::cout << "\n\t Running 1D view case...\n";
 
  std::memset(a, 0, Ntot * sizeof(int));
 
  // _default_view1D_start 
  RAJA::View< int, RAJA::Layout<1, int> > view_1D(a, Ntot);

  iter = 0;
  for (int i = 0; i < Ntot; ++i) {
    view_1D(i) = iter;
    ++iter;
  }
  // _default_view1D_end 

  checkResult<int>(a, aref, Ntot);
//printValues<int>(a, Ntot);

//----------------------------------------//

  std::cout << "\n\t Running 2D view case...\n";

  std::memset(a, 0, Ntot * sizeof(int));
 
  // _default_view2D_start
  RAJA::View< int, RAJA::Layout<2, int> > view_2D(a, Nx, Ny);

  iter = 0;
  for (int i = 0; i < Nx; ++i) {
    for (int j = 0; j < Ny; ++j) {
      view_2D(i, j) = iter;
      ++iter;
    }
  }
  // _default_view2D_end

  checkResult<int>(a, aref, Nx*Ny);
//printValues<int>(a, Nx*Ny);

//----------------------------------------//

  std::cout << "\n\t Running 3D view case...\n";

  std::memset(a, 0, Ntot * sizeof(int));

  // _default_view3D_start    
  RAJA::View< int, RAJA::Layout<3, int> > view_3D(a, Nx, Ny, Nz);

  iter = 0;
  for (int i = 0; i < Nx; ++i) {
    for (int j = 0; j < Ny; ++j) {
      for (int k = 0; k < Nz; ++k) {
        view_3D(i, j, k) = iter;
        ++iter;
      }
    }
  }
  // _default_view3D_end

  checkResult<int>(a, aref, Nx*Ny*Nz);
//printValues<int>(a, Nx*Ny*Nz);

//----------------------------------------------------------------------------//
//
// Permuted layouts change the data striding order
//
//----------------------------------------------------------------------------//

  std::cout << "\n Running permuted view cases...\n";

//----------------------------------------//

  std::cout << "\n\t Running 2D default perm view case...\n";

  std::memset(a, 0, Ntot * sizeof(int));

  // _default_perm_view2D_start
  std::array<RAJA::idx_t, 2> defperm2 {{0, 1}};
  RAJA::Layout< 2, int > defperm2_layout =
    RAJA::make_permuted_layout( {{Nx, Ny}}, defperm2);
  RAJA::View< int, RAJA::Layout<2, int> > defperm_view_2D(a, defperm2_layout);

  iter = 0;
  for (int i = 0; i < Nx; ++i) {
    for (int j = 0; j < Ny; ++j) {
      defperm_view_2D(i, j) = iter;
      ++iter;
    }
  }
  // _default_perm_view2D_end

  checkResult<int>(a, aref, Nx*Ny);
//printValues<int>(a, Nx*Ny);

//----------------------------------------//

  std::cout << "\n\t Running 3D default perm view case...\n";

  std::memset(a, 0, Ntot * sizeof(int));

  // _default_perm_view3D_start
  std::array<RAJA::idx_t, 3> defperm3 {{0, 1, 2}};
  RAJA::Layout< 3, int > defperm3_layout =
    RAJA::make_permuted_layout( {{Nx, Ny, Nz}}, defperm3);
  RAJA::View< int, RAJA::Layout<3, int> > defperm_view_3D(a, defperm3_layout);

  iter = 0;
  for (int i = 0; i < Nx; ++i) {
    for (int j = 0; j < Ny; ++j) {
      for (int k = 0; k < Nz; ++k) {
        defperm_view_3D(i, j, k) = iter;
        ++iter;
      }
    }
  }
  // _default_perm_view3D_end

  checkResult<int>(a, aref, Nx*Ny*Nz);
//printValues<int>(a, Nx*Ny*Nz);

//----------------------------------------//
//----------------------------------------//

  std::cout << "\n\t Running 2D perm view case...\n";

  std::memset(a, 0, Ntot * sizeof(int));

  // _perm_view2D_start
  std::array<RAJA::idx_t, 2> perm2 {{1, 0}};
  RAJA::Layout< 2, int > perm2_layout =
    RAJA::make_permuted_layout( {{Nx, Ny}}, perm2);
  RAJA::View< int, RAJA::Layout<2, int> > perm_view_2D(a, perm2_layout);

  iter = 0;
  for (int j = 0; j < Ny; ++j) {
    for (int i = 0; i < Nx; ++i) {
      perm_view_2D(i, j) = iter;
      ++iter;
    }
  }
  // _perm_view2D_end

  checkResult<int>(a, aref, Nx*Ny);
//printValues<int>(a, Nx*Ny);

//----------------------------------------//

  std::cout << "\n\t Running 3D perma view case...\n";

  std::memset(a, 0, Ntot * sizeof(int));

  // _perma_view3D_start
  std::array<RAJA::idx_t, 3> perm3a {{2, 1, 0}};
  RAJA::Layout< 3, int > perm3a_layout =
    RAJA::make_permuted_layout( {{Nx, Ny, Nz}}, perm3a);
  RAJA::View< int, RAJA::Layout<3, int> > perm3a_view_3D(a, perm3a_layout);

  iter = 0;
  for (int k = 0; k < Nz; ++k) {
    for (int j = 0; j < Ny; ++j) {
      for (int i = 0; i < Nx; ++i) {
        perm3a_view_3D(i, j, k) = iter;
        ++iter;
      }
    }
  }
  // _perma_view3D_end

  checkResult<int>(a, aref, Nx*Ny*Nz);
//printValues<int>(a, Nx*Ny*Nz);

//----------------------------------------//

  std::cout << "\n\t Running 3D permb view case...\n";

  std::memset(a, 0, Ntot * sizeof(int));

  // _permb_view3D_start
  std::array<RAJA::idx_t, 3> perm3b {{1, 2, 0}};
  RAJA::Layout< 3, int > perm3b_layout =
    RAJA::make_permuted_layout( {{Nx, Ny, Nz}}, perm3b);
  RAJA::View< int, RAJA::Layout<3, int> > perm3b_view_3D(a, perm3b_layout);

  iter = 0;
  for (int j = 0; j < Ny; ++j) {
    for (int k = 0; k < Nz; ++k) {
      for (int i = 0; i < Nx; ++i) {
        perm3b_view_3D(i, j, k) = iter;
        ++iter;
      }
    }
  }
  // _permb_view3D_end

  checkResult<int>(a, aref, Nx*Ny*Nz);
//printValues<int>(a, Nx*Ny*Nz);

//----------------------------------------------------------------------------//
//
// Layouts: multi-dimensional indices vs. linear indicies
//
// RAJA::Layout type has methods that can be used to convert between
// multi-dimensional and linear indices. We show these below using the
// three-dimensional layouts in the examples above. Recall the Nx, Ny, Nz
// sizes defined earlier:
//
//  constexpr int Nx = 3;
//  constexpr int Ny = 5;
//  constexpr int Nz = 2;
//
//----------------------------------------------------------------------------//

  std::cout << "\n Multi-dimensional indices to linear indices...\n";

  int lin = -1;
  int i = -1; 
  int j = -1; 
  int k = -1; 

  std::cout << "\nperm3a_layout...\n" << std::endl;

  lin = perm3a_layout(1, 2, 0);
  std::cout << "\tperm3a_layout(1, 2, 0) = " << lin << std::endl;
  std::cout << "\t  Should be 7 = 1 + 2 * Nx + 0 * Nx * Ny "
            << "(since perm is {2, 1, 0})" << std::endl;
  perm3a_layout.toIndices(7, i, j, k);
  std::cout << "\tperm3a_layout.toIndices(7, i, j, k) --> (i, j, k) = "
            << "(" << i << ", " << j << ", " << k << ")\n" << std::endl;

  lin = perm3a_layout(2, 3, 1);
  std::cout << "\tperm3a_layout(2, 3, 1) = " << lin << std::endl;
  std::cout << "\t  Should be 26 = 2 + 3 * Nx + 1 * Nx * Ny "
            << "(since perm is {2, 1, 0})" << std::endl;
  perm3a_layout.toIndices(26, i, j, k);
  std::cout << "\tperm3a_layout.toIndices(26, i, j, k) --> (i, j, k) = "
            << "(" << i << ", " << j << ", " << k << ")\n" << std::endl;

  lin = perm3a_layout(0, 2, 1);
  std::cout << "\tperm3a_layout(0, 2, 1) = " << lin << std::endl;
  std::cout << "\t  Should be 21 = 0 + 2 * Nx + 1 * Nx * Ny "
            << "(since perm is {2, 1, 0})" << std::endl;
  perm3a_layout.toIndices(21, i, j, k);
  std::cout << "\tperm3a_layout.toIndices(21, i, j, k) --> (i, j, k) = "
            << "(" << i << ", " << j << ", " << k << ")\n" << std::endl;

//----------------------------------------------------------------------------//

  std::cout << "\nperm3b_layout...\n" << std::endl;

  lin = perm3b_layout(1, 2, 0);
  std::cout << "\tperm3b_layout(1, 2, 0) = " << lin << std::endl;
  std::cout << "\t  Should be 13 = 1 + 0 * Nx + 2 * Nx * Nz "
            << "(since perm is {1, 2, 0})" << std::endl;
  perm3b_layout.toIndices(13, i, j, k);
  std::cout << "\tperm3b_layout.toIndices(13, i, j, k) --> (i, j, k) = "
            << "(" << i << ", " << j << ", " << k << ")\n" << std::endl;

  lin = perm3b_layout(2, 3, 1);
  std::cout << "\tperm3b_layout(2, 3, 1) = " << lin << std::endl;
  std::cout << "\t  Should be 23 = 2 + 1 * Nx + 3 * Nx * Nz "
            << "(since perm is {1, 2, 0})" << std::endl;
  perm3b_layout.toIndices(23, i, j, k);
  std::cout << "\tperm3b_layout.toIndices(23, i, j, k) --> (i, j, k) = "
            << "(" << i << ", " << j << ", " << k << ")\n" << std::endl;

  lin = perm3b_layout(0, 2, 1);
  std::cout << "\tperm3b_layout(0, 2, 1) = " << lin << std::endl;
  std::cout << "\t  Should be 15 = 0 + 1 * Nx + 2 * Nx * Nz "
            << "(since perm is {1, 2, 0})" << std::endl;
  perm3b_layout.toIndices(15, i, j, k);
  std::cout << "\tperm3b_layout.toIndices(15, i, j, k) --> (i, j, k) = "
            << "(" << i << ", " << j << ", " << k << ")\n" << std::endl; 

  

//
// Clean up.
//
  delete [] a;
  delete [] aref;

//----------------------------------------------------------------------------//
//----------------------------------------------------------------------------//

  std::cout << "\n DONE!...\n";

  return 0;
}

//
// Function to check result and report P/F.
//
template <typename T>
void checkResult(T* C, T* Cref, int N)
{
  bool match = true;
  for (int i = 0; i < N; ++i) {
    if ( std::abs( C[i] - Cref[i] ) > 10e-12 ) {
      match = false;
    }
  }
  if ( match ) {
    std::cout << "\n\t result -- PASS\n";
  } else {
    std::cout << "\n\t result -- FAIL\n";
  }
};

template <typename T>
void printValues(T* C, int N)
{
  for (int i = 0; i < N; ++i) {
    std::cout << "array[" << i << "] = " << C[i] << std::endl;
    }
};
