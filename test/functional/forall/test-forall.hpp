//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef __TEST_FORALL_HPP__
#define __TEST_FOARLL_HPP__

#include "RAJA/RAJA.hpp"

template<typename T>
class ForallFunctionalTest: public ::testing::Test {};

using ForallTypes = ::testing::Types<RAJA::Index_type,
                                     char, 
                                     unsigned char,
                                     short,
                                     unsigned short,
                                     int, 
                                     unsigned int,
                                     long,
                                     unsigned long,
                                     long int,
                                     unsigned long int,
                                     long long,
                                     unsigned long long>;

TYPED_TEST_SUITE(ForallFunctionalTest, ForallTypes);

#endif //__TEST_FORALL_TYPES_HPP__
