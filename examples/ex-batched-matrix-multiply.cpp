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

#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iostream>

#include "RAJA/RAJA.hpp"
#include "RAJA/util/Timer.hpp"


#include "memoryManager.hpp"

/*
 *  Batched Matrix Multiply Example
 *
 *  Example carries out batched matrix multiplication
 *  for matrices of dimension 3 x 3 using two different
 *  data layouts. 
 * 
 *  Matrices are stored in arrays A, and B. Results 
 *  are stored in C. We introduce the notation A^{e}_rc
 *  to correspond to the matrix entry in the row - r, 
 *  column - c of matrix - e. Below we describe the potential
 *  layouts in the case of two matrices NMAT=2.
 *
 * Layout 1:
 * Matrix entries are grouped together so that each 
 * matrix is in a row major ordering. 
 * i.e. A = [A^{0}_{00}, A^{0}_{01}, A^{0}_{02},
 *           A^{0}_{10}, A^{0}_{11}, A^{0}_{12},
 *           A^{0}_{20}, A^{0}_{21}, A^{0}_{22},
 *           A^{1}_{00}, A^{1}_{01}, A^{1}_{02},
 *           A^{1}_{10}, A^{1}_{11}, A^{1}_{12},
 *           A^{1}_{20}, A^{1}_{21}, A^{1}_{22}];
 *
 * Layout 2:
 * Matrix entries are first ordered by matrix number,
 * then by column number, and finally by row number. 
 * i.e. A = [A^{0}_{00}, A^{1}_{00}, A^{0}_{01},
 *           A^{1}_{01}, A^{0}_{02}, A^{1}_{02},
 *           A^{0}_{10}, A^{1}_{10}, A^{0}_{11},
 *           A^{1}_{11}, A^{0}_{12}, A^{1}_{12},
 *           A^{0}_{20}, A^{1}_{20}, A^{0}_{21},
 *           A^{1}_{21}, A^{0}_{22}, A^{1}_{22}];
 *
 * Since layout 1 has the entries for a matrix
 * close in memory it simplifies vector operations. 
 * Thus we would expect improved performance on the CPU 
 * over layout 2.
 * Layout 2 is ideal for having consecutive threads
 * operate on consecutive elements. Which is favorable
 * for a GPUs. 
 *
 *  RAJA features shown:
 *    -  RAJA View
 *    -  RAJA make_permuted_layout
 *
 * If CUDA is enabled, CUDA unified memory is used.
 */

/*
  CUDA_BLOCK_SIZE - specifies the number of threads in a CUDA thread block
*/
#if defined(RAJA_ENABLE_CUDA)
const int CUDA_BLOCK_SIZE = 256;
#endif

// Dimensions of matrices
const int NCOLS = 3;
const int NROWS = 3;
const int NMAT = 120000;

// Number of iterations
const int NITER = 20;

//
// Function for comparing outputs
//
template <typename T, typename U>
void compareOutput(T C, U Cl2, int N);

int main(int RAJA_UNUSED_ARG(argc), char **RAJA_UNUSED_ARG(argv[]))
{

  std::cout << "\n\nRAJA batched matrix multiplication example...\n";
  std::cout << "Number of matrices to be multiplied: " << NMAT << " \n";
  double myMin;
  srand(time(NULL));
  using RAJA::Index_type;
  auto timer = RAJA::Timer();

  //
  // Allocate space for data in layout 1
  //
  double *A = memoryManager::allocate<double>(NCOLS * NROWS * NMAT);
  double *B = memoryManager::allocate<double>(NCOLS * NROWS * NMAT);
  double *C = memoryManager::allocate<double>(NCOLS * NROWS * NMAT);

  //
  // Layout 1
  //
  // make_permuted_layout takes the number of entries in each dimension and an
  // array indicating slowest to fastest stride
  // Indexing is equivalent to A(e,r,c) A[c + NCOLS*(r + NROWS*e)]
  auto layout =
      RAJA::make_permuted_layout({{NMAT, NROWS, NCOLS}},
                                 RAJA::as_array<RAJA::Perm<0, 1, 2> >::get());

  // RAJA::Layout is templated on dimension, argument type, and index with unit
  // stride (in this case argument 2 has unit stride)
  RAJA::View<double, RAJA::Layout<3, Index_type, 2> > Aview(A, layout);
  RAJA::View<double, RAJA::Layout<3, Index_type, 2> > Bview(B, layout);
  RAJA::View<double, RAJA::Layout<3, Index_type, 2> > Cview(C, layout);

  //
  // Allocate space for data in layout 2
  //
  double *Al2 = memoryManager::allocate<double>(NCOLS * NROWS * NMAT);
  double *Bl2 = memoryManager::allocate<double>(NCOLS * NROWS * NMAT);
  double *Cl2 = memoryManager::allocate<double>(NCOLS * NROWS * NMAT);

  //
  // Permuted layout - equivalent to indexing via
  // A[e + NELEM*(r + NROWS*c)]
  auto layout2 =
      RAJA::make_permuted_layout({{NMAT, NROWS, NCOLS}},
                                 RAJA::as_array<RAJA::Perm<1, 2, 0> >::get());
  RAJA::View<double, RAJA::Layout<3, Index_type, 0> > Al2view(Al2, layout2);
  RAJA::View<double, RAJA::Layout<3, Index_type, 0> > Bl2view(Bl2, layout2);
  RAJA::View<double, RAJA::Layout<3, Index_type, 0> > Cl2view(Cl2, layout2);

  //
  // Initialize data
  //
  for (int e = 0; e < NMAT; ++e) {
    for (int row = 0; row < NROWS; ++row) {
      for (int col = 0; col < NCOLS; ++col) {
        Aview(e, row, col) = rand() % 50 + 1;
        Bview(e, row, col) = rand() % 50 + 1;
        Al2view(e, row, col) = Aview(e, row, col);
        Bl2view(e, row, col) = Bview(e, row, col);
      }
    }
  }

  //-------------------------------------------
  // Matrix multiply with layout 1 on the CPU with loop policy
  //
  using CPUPol1 = RAJA::loop_exec;

  myMin = std::numeric_limits<double>::max();
  for (int i = 0; i < NITER; ++i) {

    timer.start();
    RAJA::forall<CPUPol1>(RAJA::RangeSegment(0, NMAT), [=](Index_type i) {

      Cview(i, 0, 0) = Aview(i, 0, 0) * Bview(i, 0, 0)
                       + Aview(i, 0, 1) * Bview(i, 1, 0)
                       + Aview(i, 0, 2) * Bview(i, 2, 0);
      Cview(i, 0, 1) = Aview(i, 0, 0) * Bview(i, 0, 1)
                       + Aview(i, 0, 1) * Bview(i, 1, 1)
                       + Aview(i, 0, 2) * Bview(i, 2, 1);
      Cview(i, 0, 2) = Aview(i, 0, 0) * Bview(i, 0, 2)
                       + Aview(i, 0, 1) * Bview(i, 1, 2)
                       + Aview(i, 0, 2) * Bview(i, 2, 2);

      Cview(i, 1, 0) = Aview(i, 1, 0) * Bview(i, 0, 0)
                       + Aview(i, 1, 1) * Bview(i, 1, 0)
                       + Aview(i, 1, 2) * Bview(i, 2, 0);
      Cview(i, 1, 1) = Aview(i, 1, 0) * Bview(i, 0, 1)
                       + Aview(i, 1, 1) * Bview(i, 1, 1)
                       + Aview(i, 1, 2) * Bview(i, 2, 1);
      Cview(i, 1, 2) = Aview(i, 1, 0) * Bview(i, 0, 2)
                       + Aview(i, 1, 1) * Bview(i, 1, 2)
                       + Aview(i, 1, 2) * Bview(i, 2, 2);

      Cview(i, 2, 0) = Aview(i, 2, 0) * Bview(i, 0, 0)
                       + Aview(i, 2, 1) * Bview(i, 1, 0)
                       + Aview(i, 2, 2) * Bview(i, 2, 0);
      Cview(i, 2, 1) = Aview(i, 2, 0) * Bview(i, 0, 1)
                       + Aview(i, 2, 1) * Bview(i, 1, 1)
                       + Aview(i, 2, 2) * Bview(i, 2, 1);
      Cview(i, 2, 2) = Aview(i, 2, 0) * Bview(i, 0, 2)
                       + Aview(i, 2, 1) * Bview(i, 1, 2)
                       + Aview(i, 2, 2) * Bview(i, 2, 2);

    });
    timer.stop();

    RAJA::Timer::ElapsedType tMin = timer.elapsed();
    if (tMin < myMin) myMin = tMin;
    timer.reset();
  }
  std::cout << "Matrix Multiplication with layout 1 on CPU with loop policy "
               "run time : "
            << myMin << " seconds" << std::endl;
  //-------------------------------------------

  //-------------------------------------------
  // Matrix multiply with layout 2 on the CPU
  //

  myMin = std::numeric_limits<double>::max();
  for (int i = 0; i < NITER; ++i) {

    timer.start();
    RAJA::forall<CPUPol1>(RAJA::RangeSegment(0, NMAT), [=](Index_type i) {

      Cl2view(i, 0, 0) = Al2view(i, 0, 0) * Bl2view(i, 0, 0)
                         + Al2view(i, 0, 1) * Bl2view(i, 1, 0)
                         + Al2view(i, 0, 2) * Bl2view(i, 2, 0);
      Cl2view(i, 0, 1) = Al2view(i, 0, 0) * Bl2view(i, 0, 1)
                         + Al2view(i, 0, 1) * Bl2view(i, 1, 1)
                         + Al2view(i, 0, 2) * Bl2view(i, 2, 1);
      Cl2view(i, 0, 2) = Al2view(i, 0, 0) * Bl2view(i, 0, 2)
                         + Al2view(i, 0, 1) * Bl2view(i, 1, 2)
                         + Al2view(i, 0, 2) * Bl2view(i, 2, 2);

      Cl2view(i, 1, 0) = Al2view(i, 1, 0) * Bl2view(i, 0, 0)
                         + Al2view(i, 1, 1) * Bl2view(i, 1, 0)
                         + Al2view(i, 1, 2) * Bl2view(i, 2, 0);
      Cl2view(i, 1, 1) = Al2view(i, 1, 0) * Bl2view(i, 0, 1)
                         + Al2view(i, 1, 1) * Bl2view(i, 1, 1)
                         + Al2view(i, 1, 2) * Bl2view(i, 2, 1);
      Cl2view(i, 1, 2) = Al2view(i, 1, 0) * Bl2view(i, 0, 2)
                         + Al2view(i, 1, 1) * Bl2view(i, 1, 2)
                         + Al2view(i, 1, 2) * Bl2view(i, 2, 2);

      Cl2view(i, 2, 0) = Al2view(i, 2, 0) * Bl2view(i, 0, 0)
                         + Al2view(i, 2, 1) * Bl2view(i, 1, 0)
                         + Al2view(i, 2, 2) * Bl2view(i, 2, 0);
      Cl2view(i, 2, 1) = Al2view(i, 2, 0) * Bl2view(i, 0, 1)
                         + Al2view(i, 2, 1) * Bl2view(i, 1, 1)
                         + Al2view(i, 2, 2) * Bl2view(i, 2, 1);
      Cl2view(i, 2, 2) = Al2view(i, 2, 0) * Bl2view(i, 0, 2)
                         + Al2view(i, 2, 1) * Bl2view(i, 1, 2)
                         + Al2view(i, 2, 2) * Bl2view(i, 2, 2);

    });
    timer.stop();

    RAJA::Timer::ElapsedType tMin = timer.elapsed();
    if (tMin < myMin) myMin = tMin;
    timer.reset();
  }
  std::cout << "Matrix Multiplication with layout 2 on CPU with loop policy "
               "run time : "
            << myMin << " seconds" << std::endl;
  //---------------------------------------------

  //
  // Compare output
  //
  compareOutput(Cview, Cl2view, NMAT);

#if defined(RAJA_ENABLE_OPENMP)
  //-------------------------------------------
  // Matrix multiply with layout 1 on the CPU with OpenMP threads
  //
  using CPUPol2 = RAJA::omp_parallel_for_exec;

  myMin = std::numeric_limits<double>::max();
  for (int i = 0; i < NITER; ++i) {

    timer.start();
    RAJA::forall<CPUPol2>(RAJA::RangeSegment(0, NMAT), [=](Index_type i) {

      Cview(i, 0, 0) = Aview(i, 0, 0) * Bview(i, 0, 0)
                       + Aview(i, 0, 1) * Bview(i, 1, 0)
                       + Aview(i, 0, 2) * Bview(i, 2, 0);
      Cview(i, 0, 1) = Aview(i, 0, 0) * Bview(i, 0, 1)
                       + Aview(i, 0, 1) * Bview(i, 1, 1)
                       + Aview(i, 0, 2) * Bview(i, 2, 1);
      Cview(i, 0, 2) = Aview(i, 0, 0) * Bview(i, 0, 2)
                       + Aview(i, 0, 1) * Bview(i, 1, 2)
                       + Aview(i, 0, 2) * Bview(i, 2, 2);

      Cview(i, 1, 0) = Aview(i, 1, 0) * Bview(i, 0, 0)
                       + Aview(i, 1, 1) * Bview(i, 1, 0)
                       + Aview(i, 1, 2) * Bview(i, 2, 0);
      Cview(i, 1, 1) = Aview(i, 1, 0) * Bview(i, 0, 1)
                       + Aview(i, 1, 1) * Bview(i, 1, 1)
                       + Aview(i, 1, 2) * Bview(i, 2, 1);
      Cview(i, 1, 2) = Aview(i, 1, 0) * Bview(i, 0, 2)
                       + Aview(i, 1, 1) * Bview(i, 1, 2)
                       + Aview(i, 1, 2) * Bview(i, 2, 2);

      Cview(i, 2, 0) = Aview(i, 2, 0) * Bview(i, 0, 0)
                       + Aview(i, 2, 1) * Bview(i, 1, 0)
                       + Aview(i, 2, 2) * Bview(i, 2, 0);
      Cview(i, 2, 1) = Aview(i, 2, 0) * Bview(i, 0, 1)
                       + Aview(i, 2, 1) * Bview(i, 1, 1)
                       + Aview(i, 2, 2) * Bview(i, 2, 1);
      Cview(i, 2, 2) = Aview(i, 2, 0) * Bview(i, 0, 2)
                       + Aview(i, 2, 1) * Bview(i, 1, 2)
                       + Aview(i, 2, 2) * Bview(i, 2, 2);
    });
    timer.stop();

    RAJA::Timer::ElapsedType tMin = timer.elapsed();
    if (tMin < myMin) myMin = tMin;
    timer.reset();
  }
  std::cout << "Matrix Multiplication with layout 1 on CPU with OpenMP parallel policy "
               "run time : "
            << myMin << " seconds" << std::endl;
  //-------------------------------------------

  //-------------------------------------------
  // Matrix multiply with layout 2 on the CPU with OpenMP parallel policy
  //
  myMin = std::numeric_limits<double>::max();
  for (int i = 0; i < NITER; ++i) {

    timer.start();
    RAJA::forall<CPUPol2>(RAJA::RangeSegment(0, NMAT), [=](Index_type i) {

      Cl2view(i, 0, 0) = Al2view(i, 0, 0) * Bl2view(i, 0, 0)
                         + Al2view(i, 0, 1) * Bl2view(i, 1, 0)
                         + Al2view(i, 0, 2) * Bl2view(i, 2, 0);
      Cl2view(i, 0, 1) = Al2view(i, 0, 0) * Bl2view(i, 0, 1)
                         + Al2view(i, 0, 1) * Bl2view(i, 1, 1)
                         + Al2view(i, 0, 2) * Bl2view(i, 2, 1);
      Cl2view(i, 0, 2) = Al2view(i, 0, 0) * Bl2view(i, 0, 2)
                         + Al2view(i, 0, 1) * Bl2view(i, 1, 2)
                         + Al2view(i, 0, 2) * Bl2view(i, 2, 2);

      Cl2view(i, 1, 0) = Al2view(i, 1, 0) * Bl2view(i, 0, 0)
                         + Al2view(i, 1, 1) * Bl2view(i, 1, 0)
                         + Al2view(i, 1, 2) * Bl2view(i, 2, 0);
      Cl2view(i, 1, 1) = Al2view(i, 1, 0) * Bl2view(i, 0, 1)
                         + Al2view(i, 1, 1) * Bl2view(i, 1, 1)
                         + Al2view(i, 1, 2) * Bl2view(i, 2, 1);
      Cl2view(i, 1, 2) = Al2view(i, 1, 0) * Bl2view(i, 0, 2)
                         + Al2view(i, 1, 1) * Bl2view(i, 1, 2)
                         + Al2view(i, 1, 2) * Bl2view(i, 2, 2);

      Cl2view(i, 2, 0) = Al2view(i, 2, 0) * Bl2view(i, 0, 0)
                         + Al2view(i, 2, 1) * Bl2view(i, 1, 0)
                         + Al2view(i, 2, 2) * Bl2view(i, 2, 0);
      Cl2view(i, 2, 1) = Al2view(i, 2, 0) * Bl2view(i, 0, 1)
                         + Al2view(i, 2, 1) * Bl2view(i, 1, 1)
                         + Al2view(i, 2, 2) * Bl2view(i, 2, 1);
      Cl2view(i, 2, 2) = Al2view(i, 2, 0) * Bl2view(i, 0, 2)
                         + Al2view(i, 2, 1) * Bl2view(i, 1, 2)
                         + Al2view(i, 2, 2) * Bl2view(i, 2, 2);

    });
    timer.stop();

    RAJA::Timer::ElapsedType tMin = timer.elapsed();
    if (tMin < myMin) myMin = tMin;
    timer.reset();
  }
  std::cout << "Matrix Multiplication with layout 2 on CPU with OpenMP parallel policy "
               "run time : "
            << myMin << " seconds" << std::endl;
  //---------------------------------------------

  //
  // Compare output
  //
  compareOutput(Cview, Cl2view, NMAT);
#endif

#if defined(RAJA_ENABLE_CUDA)
  //-------------------------------------------
  // Matrix multiply with layout 1 on the GPU
  //
  using GPUPol = RAJA::cuda_exec<CUDA_BLOCK_SIZE>;

  myMin = std::numeric_limits<double>::max();
  for (int i = 0; i < NITER; ++i) {

    timer.start();
    RAJA::forall<GPUPol>(RAJA::RangeSegment(0, NMAT),
                         [=] RAJA_DEVICE(Index_type i) {

                           Cview(i, 0, 0) = Aview(i, 0, 0) * Bview(i, 0, 0)
                                            + Aview(i, 0, 1) * Bview(i, 1, 0)
                                            + Aview(i, 0, 2) * Bview(i, 2, 0);
                           Cview(i, 0, 1) = Aview(i, 0, 0) * Bview(i, 0, 1)
                                            + Aview(i, 0, 1) * Bview(i, 1, 1)
                                            + Aview(i, 0, 2) * Bview(i, 2, 1);
                           Cview(i, 0, 2) = Aview(i, 0, 0) * Bview(i, 0, 2)
                                            + Aview(i, 0, 1) * Bview(i, 1, 2)
                                            + Aview(i, 0, 2) * Bview(i, 2, 2);

                           Cview(i, 1, 0) = Aview(i, 1, 0) * Bview(i, 0, 0)
                                            + Aview(i, 1, 1) * Bview(i, 1, 0)
                                            + Aview(i, 1, 2) * Bview(i, 2, 0);
                           Cview(i, 1, 1) = Aview(i, 1, 0) * Bview(i, 0, 1)
                                            + Aview(i, 1, 1) * Bview(i, 1, 1)
                                            + Aview(i, 1, 2) * Bview(i, 2, 1);
                           Cview(i, 1, 2) = Aview(i, 1, 0) * Bview(i, 0, 2)
                                            + Aview(i, 1, 1) * Bview(i, 1, 2)
                                            + Aview(i, 1, 2) * Bview(i, 2, 2);

                           Cview(i, 2, 0) = Aview(i, 2, 0) * Bview(i, 0, 0)
                                            + Aview(i, 2, 1) * Bview(i, 1, 0)
                                            + Aview(i, 2, 2) * Bview(i, 2, 0);
                           Cview(i, 2, 1) = Aview(i, 2, 0) * Bview(i, 0, 1)
                                            + Aview(i, 2, 1) * Bview(i, 1, 1)
                                            + Aview(i, 2, 2) * Bview(i, 2, 1);
                           Cview(i, 2, 2) = Aview(i, 2, 0) * Bview(i, 0, 2)
                                            + Aview(i, 2, 1) * Bview(i, 1, 2)
                                            + Aview(i, 2, 2) * Bview(i, 2, 2);
                         });
    timer.stop();

    RAJA::Timer::ElapsedType tMin = timer.elapsed();
    if (tMin < myMin) myMin = tMin;
    timer.reset();
  }
  std::cout << "Matrix Multiplication with layout 1 on GPU with Cuda policy run time : " << myMin
            << " seconds" << std::endl;
  //-------------------------------------------

  //-------------------------------------------
  // Matrix multiply with layout 2 on the GPU
  //
  myMin = std::numeric_limits<double>::max();
  for (int i = 0; i < NITER; ++i) {

    timer.start();
    RAJA::forall<GPUPol>(
        RAJA::RangeSegment(0, NMAT), [=] RAJA_DEVICE(Index_type i) {

          Cl2view(i, 0, 0) = Al2view(i, 0, 0) * Bl2view(i, 0, 0)
                             + Al2view(i, 0, 1) * Bl2view(i, 1, 0)
                             + Al2view(i, 0, 2) * Bl2view(i, 2, 0);
          Cl2view(i, 0, 1) = Al2view(i, 0, 0) * Bl2view(i, 0, 1)
                             + Al2view(i, 0, 1) * Bl2view(i, 1, 1)
                             + Al2view(i, 0, 2) * Bl2view(i, 2, 1);
          Cl2view(i, 0, 2) = Al2view(i, 0, 0) * Bl2view(i, 0, 2)
                             + Al2view(i, 0, 1) * Bl2view(i, 1, 2)
                             + Al2view(i, 0, 2) * Bl2view(i, 2, 2);

          Cl2view(i, 1, 0) = Al2view(i, 1, 0) * Bl2view(i, 0, 0)
                             + Al2view(i, 1, 1) * Bl2view(i, 1, 0)
                             + Al2view(i, 1, 2) * Bl2view(i, 2, 0);
          Cl2view(i, 1, 1) = Al2view(i, 1, 0) * Bl2view(i, 0, 1)
                             + Al2view(i, 1, 1) * Bl2view(i, 1, 1)
                             + Al2view(i, 1, 2) * Bl2view(i, 2, 1);
          Cl2view(i, 1, 2) = Al2view(i, 1, 0) * Bl2view(i, 0, 2)
                             + Al2view(i, 1, 1) * Bl2view(i, 1, 2)
                             + Al2view(i, 1, 2) * Bl2view(i, 2, 2);

          Cl2view(i, 2, 0) = Al2view(i, 2, 0) * Bl2view(i, 0, 0)
                             + Al2view(i, 2, 1) * Bl2view(i, 1, 0)
                             + Al2view(i, 2, 2) * Bl2view(i, 2, 0);
          Cl2view(i, 2, 1) = Al2view(i, 2, 0) * Bl2view(i, 0, 1)
                             + Al2view(i, 2, 1) * Bl2view(i, 1, 1)
                             + Al2view(i, 2, 2) * Bl2view(i, 2, 1);
          Cl2view(i, 2, 2) = Al2view(i, 2, 0) * Bl2view(i, 0, 2)
                             + Al2view(i, 2, 1) * Bl2view(i, 1, 2)
                             + Al2view(i, 2, 2) * Bl2view(i, 2, 2);

        });
    timer.stop();

    RAJA::Timer::ElapsedType tMin = timer.elapsed();
    if (tMin < myMin) myMin = tMin;
    timer.reset();
  }
  std::cout << "Matrix Multiplication with layout 2 on GPU with Cuda policy run time : " << myMin
            << " seconds" << std::endl;
  //---------------------------------------------

  //
  // Compare output
  //
  compareOutput(Cview, Cl2view, NMAT);
#endif

  //
  // Clean up.
  //
  memoryManager::deallocate(A);
  memoryManager::deallocate(B);
  memoryManager::deallocate(C);
  memoryManager::deallocate(Cl2);

  std::cout << "\n DONE!...\n";

  return 0;
}

//
// Compare output
//
template <typename T, typename U>
void compareOutput(T C, U Cl2, int N)
{

  bool status = true;
  for (int e = 0; e < N; ++e) {
    for (int row = 0; row < NROWS; ++row) {
      for (int col = 0; col < NCOLS; ++col) {
        double terr = std::abs(C(e, row, col) - Cl2(e, row, col));
        if ((terr) > 1e-8) {
          status = false;
        }
      }
    }
  }

  if (status == false) {
    std::cout << "Matrix Multiply - fail" << std::endl;
  } else {
    std::cout << "Matrix Multiply - pass" << std::endl;
  }
}
