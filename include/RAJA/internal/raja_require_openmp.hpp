/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   RAJA header file to enforce consistency of OpenMP between libRAJA
 *          and user code
 *
 ******************************************************************************
 */

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

#ifndef RAJA_INTERNAL_RAJA_REQUIRE_OPENMP_HPP
#define RAJA_INTERNAL_RAJA_REQUIRE_OPENMP_HPP
#include<RAJA/config.hpp>

#ifdef RAJA_ENABLE_OPENMP
#ifndef _OPENMP
#error RAJA was configured to use OpenMP, a host code is compiling it without OpenMP. This would cause a linker error. Please enable OpenMP in your code, or disable it in RAJA 
#endif // RAJA_ENABLE_OPENMP
#endif // _OPENMP

#endif // Header guard for raja_require_openmp.hpp
