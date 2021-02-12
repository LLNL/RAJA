
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-21, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef __TEST_KERNEL_NESTED_LOOP_TYPES_HPP__
#define __TEST_KERNEL_NESTED_LOOP_TYPES_HPP__

#if defined(RAJA_ENABLE_HIP)
#define OFFLOAD_KERNEL HipKernel
#else
#define OFFLOAD_KERNEL CudaKernel
#endif

struct NULL_T {};
struct DEPTH_2 {};
struct DEPTH_2_COLLAPSE {};
struct DEPTH_3 {};
struct OFFLOAD_DEPTH_2 {};
struct OFFLOAD_DEPTH_3 {};

//
//
// Nested Loop Data Type information
//
//
template<typename EXEC_POL1, typename EXEC_POL2=void, typename EXEC_POL3=void>
struct s_NestedLoopData { 
  using type = camp::list<DEPTH_3,
                          EXEC_POL1,
                          EXEC_POL2,
                          EXEC_POL3>;
};

template<typename EXEC_POL1, typename EXEC_POL2>
struct s_NestedLoopData<EXEC_POL1, EXEC_POL2, void> { 
  using type = camp::list<DEPTH_2,
                          EXEC_POL1,
                          EXEC_POL2>;
};

template<typename EXEC_POL1>
struct s_NestedLoopData<EXEC_POL1, void, void> { 
  using type = camp::list<DEPTH_2_COLLAPSE,
                          EXEC_POL1>;
};

// Alias to clean up loop type information
template<typename... T >
using NestedLoopData = typename s_NestedLoopData<T...>::type;

template<typename EXEC_POL1, typename EXEC_POL2, typename EXEC_POL3=void>
struct s_OffloadNestedLoopData { 
  using type = camp::list<OFFLOAD_DEPTH_3,
                          EXEC_POL1,
                          EXEC_POL2,
                          EXEC_POL3>;
};

template<typename EXEC_POL1, typename EXEC_POL2>
struct s_OffloadNestedLoopData<EXEC_POL1, EXEC_POL2, void> { 
  using type = camp::list<OFFLOAD_DEPTH_2,
                          EXEC_POL1,
                          EXEC_POL2>;
};


// Alias to clean up loop type information
template<typename... T >
using OffloadNestedLoopData = typename s_OffloadNestedLoopData<T...>::type;


template<typename EXEC_POL>
using is_null_exec_pol = typename std::is_same<EXEC_POL, NULL_T>::type;

template<typename EXEC_POL>
using is_not_null_exec_pol = typename camp::concepts::negate<is_null_exec_pol<EXEC_POL>>::type;

template <typename WORKING_RES, typename EXEC_POLICY, typename... Args>
typename std::enable_if<is_null_exec_pol<EXEC_POLICY>::value>::type
KernelNestedLoopTest(Args...){}

#endif  // __TEST_KERNEL_NESTED_LOOP_TYPES_HPP__
