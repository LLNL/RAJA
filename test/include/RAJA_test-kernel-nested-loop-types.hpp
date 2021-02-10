
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-21, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef __TEST_KERNEL_NESTED_LOOP_TYPES_HPP__
#define __TEST_KERNEL_NESTED_LOOP_TYPES_HPP__

struct LAMBDA_COUNT_2 {};
struct LAMBDA_COUNT_3 {};
struct DEPTH_2 {};
struct DEPTH_2_COLLAPSE {};
struct DEPTH_3 {};
struct OFFLOAD {};

//
//
// Nested Loop Data Type information
// Might want this in a header somewhere
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

template<typename EXEC_POL1, typename EXEC_POL2>
struct s_OffloadNestedLoopData { 
  using type = camp::list<OFFLOAD,
                          EXEC_POL1,
                          EXEC_POL2>;
};

// Alias to clean up loop type information
template<typename... T >
using OffloadNestedLoopData = typename s_OffloadNestedLoopData<T...>::type;


#endif  // __TEST_KERNEL_NESTED_LOOP_TYPES_HPP__
