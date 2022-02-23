//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-22, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//


// This files defined the matrix element type, and matrix tests that are
// appropriate for a given element type

using MatrixElementType = int32_t;

using TensorMatrixTypes = ::testing::Types<

#ifdef RAJA_ENABLE_CUDA
    RAJA::RectMatrixRegister<MatrixElementType, TensorMatrixLayoutType, 8,4, RAJA::cuda_warp_register>,
    RAJA::RectMatrixRegister<MatrixElementType, TensorMatrixLayoutType, 8,8, RAJA::cuda_warp_register>,
#endif


#ifdef __AVX__
    RAJA::RectMatrixRegister<MatrixElementType, TensorMatrixLayoutType, 4,8, RAJA::avx_register>,
    RAJA::RectMatrixRegister<MatrixElementType, TensorMatrixLayoutType, 8,8, RAJA::avx_register>,
    RAJA::RectMatrixRegister<MatrixElementType, TensorMatrixLayoutType, 8,4, RAJA::avx_register>,
#endif


#ifdef __AVX2__
    RAJA::RectMatrixRegister<MatrixElementType, TensorMatrixLayoutType, 4,8, RAJA::avx2_register>,
    RAJA::RectMatrixRegister<MatrixElementType, TensorMatrixLayoutType, 8,8, RAJA::avx2_register>,
    RAJA::RectMatrixRegister<MatrixElementType, TensorMatrixLayoutType, 8,4, RAJA::avx2_register>,
#endif


#ifdef __AVX512__
    RAJA::RectMatrixRegister<MatrixElementType, TensorMatrixLayoutType, 8,16, RAJA::avx512_register>,
    RAJA::RectMatrixRegister<MatrixElementType, TensorMatrixLayoutType, 16,16, RAJA::avx512_register>,
    RAJA::RectMatrixRegister<MatrixElementType, TensorMatrixLayoutType, 16,8, RAJA::avx512_register>,
#endif


    // These tests use the platform default SIMD architecture
    RAJA::SquareMatrixRegister<MatrixElementType, TensorMatrixLayoutType>,

    // Always test the non-vectorized scalar type
    RAJA::SquareMatrixRegister<MatrixElementType, TensorMatrixLayoutType, RAJA::scalar_register>

  >;
