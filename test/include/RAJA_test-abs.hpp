//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-21, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef __RAJA_test_abs_HPP__
#define __RAJA_test_abs_HPP__

#include "RAJA/RAJA.hpp"
#include "camp/camp.hpp"

namespace RAJA {

  template<typename T>
  std::enable_if_t< std::is_floating_point<T>::value , T>
  test_abs(T&& val) {
    return std::fabs(val);
  } 

  template<typename T>
  std::enable_if_t< std::is_integral<T>::value , T>
  test_abs(T&& val) {
    return std::abs(val);
  }

} // namespace RAJA

#endif // __RAJA_test_abs_HPP__
