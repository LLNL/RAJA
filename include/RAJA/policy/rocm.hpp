/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   Header file containing RAJA headers for NVCC ROCM execution.
 *
 *          These methods work only on platforms that support ROCM.
 *
 ******************************************************************************
 */

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-17, Lawrence Livermore National Security, LLC.
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

#ifndef RAJA_rocm_HPP
#define RAJA_rocm_HPP

#include "RAJA/config.hpp"

#if defined(RAJA_ENABLE_ROCM)

#include <hc.hpp>
#include <hc_printf.hpp>
#define hipHostMallocDefault        0x0
#include <hip/hcc_detail/hip_runtime_api.h>

/*
// now defined in hip_runtime_api.h
class dim3 {
public:
int x,y,z;
dim3(int _x, int _y, int _z):x(_x),y(_y),z(_z) {};
};
*/
#define threadIdx_x (hc_get_workitem_id(0))
#define threadIdx_y (hc_get_workitem_id(1))
#define threadIdx_z (hc_get_workitem_id(2))

#define blockIdx_x  (hc_get_group_id(0))
#define blockIdx_y  (hc_get_group_id(1))
#define blockIdx_z  (hc_get_group_id(2))

#define blockDim_x  (hc_get_group_size(0))
#define blockDim_y  (hc_get_group_size(1))
#define blockDim_z  (hc_get_group_size(2))

#define gridDim_x   (hc_get_num_groups(0))
#define gridDim_y   (hc_get_num_groups(1))
#define gridDim_z   (hc_get_num_groups(2))




#include "RAJA/policy/rocm/atomic.hpp"
#include "RAJA/policy/rocm/forall.hpp"
#include "RAJA/policy/rocm/policy.hpp"
#include "RAJA/policy/rocm/reduce.hpp"
#if defined(__HCC__)
#include "RAJA/policy/rocm/scan.hpp"
#endif

#include "RAJA/policy/rocm/forallN.hpp"
//#include "RAJA/policy/rocm/nested.hpp"
#include "RAJA/policy/rocm/kernel.hpp"

#endif  // closing endif for if defined(RAJA_ENABLE_ROCM)

#endif  // closing endif for header file include guard
