//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-22, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//


// This files defined the matrix element type, and matrix tests that are
// appropriate for a given element type

using MatrixElementType = int64_t;

using TensorMatrixTypes = ::testing::Types<

#ifdef RAJA_ENABLE_CUDA
    RAJA::expt::RectMatrixRegister<MatrixElementType, TensorMatrixLayoutType, 8,4, RAJA::expt::cuda_warp_register>,
    RAJA::expt::RectMatrixRegister<MatrixElementType, TensorMatrixLayoutType, 8,8, RAJA::expt::cuda_warp_register>,
#endif


#ifdef __AVX__
    RAJA::expt::RectMatrixRegister<MatrixElementType, TensorMatrixLayoutType, 2,4, RAJA::expt::avx_register>,
    RAJA::expt::RectMatrixRegister<MatrixElementType, TensorMatrixLayoutType, 2,8, RAJA::expt::avx_register>,
    RAJA::expt::RectMatrixRegister<MatrixElementType, TensorMatrixLayoutType, 4,8, RAJA::expt::avx_register>,
    RAJA::expt::RectMatrixRegister<MatrixElementType, TensorMatrixLayoutType, 8,8, RAJA::expt::avx_register>,
    RAJA::expt::RectMatrixRegister<MatrixElementType, TensorMatrixLayoutType, 8,4, RAJA::expt::avx_register>,
    RAJA::expt::RectMatrixRegister<MatrixElementType, TensorMatrixLayoutType, 4,2, RAJA::expt::avx_register>,
    RAJA::expt::RectMatrixRegister<MatrixElementType, TensorMatrixLayoutType, 8,2, RAJA::expt::avx_register>,

#endif


#ifdef __AVX2__
    RAJA::expt::RectMatrixRegister<MatrixElementType, TensorMatrixLayoutType, 2,4, RAJA::expt::avx2_register>,
    RAJA::expt::RectMatrixRegister<MatrixElementType, TensorMatrixLayoutType, 2,8, RAJA::expt::avx2_register>,
    RAJA::expt::RectMatrixRegister<MatrixElementType, TensorMatrixLayoutType, 4,8, RAJA::expt::avx2_register>,
    RAJA::expt::RectMatrixRegister<MatrixElementType, TensorMatrixLayoutType, 8,8, RAJA::expt::avx2_register>,
    RAJA::expt::RectMatrixRegister<MatrixElementType, TensorMatrixLayoutType, 8,4, RAJA::expt::avx2_register>,
    RAJA::expt::RectMatrixRegister<MatrixElementType, TensorMatrixLayoutType, 4,2, RAJA::expt::avx2_register>,
    RAJA::expt::RectMatrixRegister<MatrixElementType, TensorMatrixLayoutType, 8,2, RAJA::expt::avx2_register>,
#endif


#ifdef __AVX512__
    RAJA::expt::RectMatrixRegister<MatrixElementType, TensorMatrixLayoutType, 4,8, RAJA::expt::avx512_register>,
    RAJA::expt::RectMatrixRegister<MatrixElementType, TensorMatrixLayoutType, 4,16, RAJA::expt::avx512_register>,
    RAJA::expt::RectMatrixRegister<MatrixElementType, TensorMatrixLayoutType, 8,16, RAJA::expt::avx512_register>,
    RAJA::expt::RectMatrixRegister<MatrixElementType, TensorMatrixLayoutType, 16,16, RAJA::expt::avx512_register>,
    RAJA::expt::RectMatrixRegister<MatrixElementType, TensorMatrixLayoutType, 16,8, RAJA::expt::avx512_register>,
    RAJA::expt::RectMatrixRegister<MatrixElementType, TensorMatrixLayoutType, 16,4, RAJA::expt::avx512_register>,
    RAJA::expt::RectMatrixRegister<MatrixElementType, TensorMatrixLayoutType, 8,4, RAJA::expt::avx512_register>,
#endif


    // These tests use the platform default SIMD architecture
    RAJA::expt::SquareMatrixRegister<MatrixElementType, TensorMatrixLayoutType>,

    // Always test the non-vectorized scalar type
    RAJA::expt::SquareMatrixRegister<MatrixElementType, TensorMatrixLayoutType, RAJA::expt::scalar_register>

  >;
