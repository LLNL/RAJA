//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-21, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//


// This files defined the matrix element type, and matrix tests that are
// appropriate for a given element type

using MatrixElementType = double;

using TensorMatrixTypes = ::testing::Types<

#ifdef RAJA_ENABLE_CUDA
    RAJA::RectMatrixRegister<MatrixElementType, TensorMatrixLayoutType, 8,4, RAJA::cuda_warp_register>,
    RAJA::RectMatrixRegister<MatrixElementType, TensorMatrixLayoutType, 4,8, RAJA::cuda_warp_register>,
    RAJA::RectMatrixRegister<MatrixElementType, TensorMatrixLayoutType, 8,8, RAJA::cuda_warp_register>,
#endif

    // These tests use the platform default SIMD architecture
    RAJA::SquareMatrixRegister<MatrixElementType, TensorMatrixLayoutType>,

    // Try different rectangular matrices
//    RAJA::RectMatrixRegister<MatrixElementType, TensorMatrixLayoutType, 2,8>,
    RAJA::RectMatrixRegister<MatrixElementType, TensorMatrixLayoutType, 4,8>,
    RAJA::RectMatrixRegister<MatrixElementType, TensorMatrixLayoutType, 8,8>,
    RAJA::RectMatrixRegister<MatrixElementType, TensorMatrixLayoutType, 8,4>,
//    RAJA::RectMatrixRegister<MatrixElementType, TensorMatrixLayoutType, 8,2>,

    RAJA::SquareMatrixRegister<MatrixElementType, TensorMatrixLayoutType, RAJA::scalar_register>

  >;

