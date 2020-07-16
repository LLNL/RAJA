//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
#ifndef  RAJA_counter_HPP
#define  RAJA_counter_HPP

// note that these are pointers here to allow different types of memory
// to be used
extern int*            plugin_test_capture_counter_pre;
extern int*            plugin_test_capture_counter_post;
extern RAJA::Platform* plugin_test_capture_platform_active;

extern int*            plugin_test_launch_counter_pre;
extern int*            plugin_test_launch_counter_post;
extern RAJA::Platform* plugin_test_launch_platform_active;

#endif  // RAJA_counter_HPP
