
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
template<typename LoopPolType, typename... Policies> 
struct NestedLoopData : camp::list<Policies...> {
  using LoopType = LoopPolType;
};


namespace detail{

  using namespace camp;

  template<typename T, typename Elements>
  struct is_in_type_list;

  template<typename T, typename Elements>
  struct NLEB_impl;

  template<typename T, typename First, typename... Rest>
  struct is_in_type_list<T, list<First, Rest...>> :
    std::conditional<
      std::is_same<  typename at<T, num<0>>::type, First  >::value,
      list<T>,
      typename is_in_type_list<T, list<Rest...>>::type > {};

  template<typename T, typename Last>
  struct is_in_type_list<T, list<Last>> :
    std::conditional< std::is_same< typename at<T, num<0>>::type , Last>::value,
      list<T>,
      list<> >{};

  template<typename POL_TYPE_LIST, typename First, typename... Rest>
  struct NLEB_impl<POL_TYPE_LIST, list<First, Rest...>> :
    join< typename NLEB_impl<POL_TYPE_LIST, list<First>>::type, 
          typename NLEB_impl<POL_TYPE_LIST, list<Rest...>>::type > {};

  template<typename POL_TYPE_LIST, typename Last>
  struct NLEB_impl<POL_TYPE_LIST, list<Last>> :
    is_in_type_list<Last, POL_TYPE_LIST > {};

} // namespace detail


template<typename POL_TYPE_LIST, typename EXEC_POL_LIST>
struct NestedLoopExecBuilder {
  using type = typename detail::NLEB_impl<POL_TYPE_LIST, EXEC_POL_LIST>::type;
};


#endif  // __TEST_KERNEL_NESTED_LOOP_TYPES_HPP__
