/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   RAJA header file for CUDA google test convenience macros
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
// For additional details, please also read RAJA/LICENSE.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// * Redistributions of source code must retain the above copyright notice,
//   this list of conditions and the disclaimer below.
//
// * Redistributions in binary form must reproduce the above copyright notice,
//   this list of conditions and the disclaimer (as noted below) in the
//   documentation and/or other materials provided with the distribution.
//
// * Neither the name of the LLNS/LLNL nor the names of its contributors may
//   be used to endorse or promote products derived from this software without
//   specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL LAWRENCE LIVERMORE NATIONAL SECURITY,
// LLC, THE U.S. DEPARTMENT OF ENERGY OR CONTRIBUTORS BE LIABLE FOR ANY
// DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
// DAMAGES  (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS
// OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
// HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
// STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING
// IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
//
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef RAJA_gtest_HPP
#define RAJA_gtest_HPP

#include "gtest/gtest.h"

#define CUDA_TEST(X, Y)                 \
  static void cuda_test_##X##_##Y();    \
  TEST(X, Y) { cuda_test_##X##_##Y(); } \
  static void cuda_test_##X##_##Y()

#define CUDA_TEST_F(test_fixture, test_name)                  \
  static void cuda_test_f_##test_fixture##_##test_name();     \
  GTEST_TEST_(test_fixture,                                   \
              test_name,                                      \
              test_fixture,                                   \
              ::testing::internal::GetTypeId<test_fixture>()) \
  {                                                           \
    cuda_test_f_##test_fixture##_##test_name();               \
  }                                                           \
  static void cuda_test_f_##test_fixture##_##test_name()

#define CUDA_TEST_P(test_case_name, test_name)                                 \
  template <typename Invocable>                                                \
  static void gtest_cuda_##test_case_name##_##test_name(Invocable &&);         \
  class GTEST_TEST_CLASS_NAME_(test_case_name, test_name)                      \
      : public test_case_name                                                  \
  {                                                                            \
  public:                                                                      \
    GTEST_TEST_CLASS_NAME_(test_case_name, test_name)() {}                     \
    virtual void TestBody()                                                    \
    {                                                                          \
      gtest_cuda_##test_case_name##_##test_name([&] { return GetParam(); });   \
    }                                                                          \
                                                                               \
  private:                                                                     \
    static int AddToRegistry()                                                 \
    {                                                                          \
      ::testing::UnitTest::GetInstance()                                       \
          ->parameterized_test_registry()                                      \
          .GetTestCasePatternHolder<test_case_name>(                           \
              #test_case_name,                                                 \
              ::testing::internal::CodeLocation(__FILE__, __LINE__))           \
          ->AddTestPattern(                                                    \
              #test_case_name,                                                 \
              #test_name,                                                      \
              new ::testing::internal::TestMetaFactory<GTEST_TEST_CLASS_NAME_( \
                  test_case_name, test_name)>());                              \
      return 0;                                                                \
    }                                                                          \
    static int gtest_registering_dummy_ GTEST_ATTRIBUTE_UNUSED_;               \
    GTEST_DISALLOW_COPY_AND_ASSIGN_(GTEST_TEST_CLASS_NAME_(test_case_name,     \
                                                           test_name));        \
  };                                                                           \
  int GTEST_TEST_CLASS_NAME_(test_case_name,                                   \
                             test_name)::gtest_registering_dummy_ =            \
      GTEST_TEST_CLASS_NAME_(test_case_name, test_name)::AddToRegistry();      \
  template <typename Invocable>                                                \
  static void gtest_cuda_##test_case_name##_##test_name(Invocable &&GetParam)


#define CUDA_TYPED_TEST_P(CaseName, TestName)                            \
  template <typename TypeParam>                                          \
  static void cuda_typed_test_p_##CaseName##_##TestName();               \
  namespace GTEST_CASE_NAMESPACE_(CaseName)                              \
  {                                                                      \
    template <typename gtest_TypeParam_>                                 \
    class TestName : public CaseName<gtest_TypeParam_>                   \
    {                                                                    \
    private:                                                             \
      typedef CaseName<gtest_TypeParam_> TestFixture;                    \
      typedef gtest_TypeParam_ TypeParam;                                \
      virtual void TestBody()                                            \
      {                                                                  \
        cuda_typed_test_p_##CaseName##_##TestName<gtest_TypeParam_>();   \
      }                                                                  \
    };                                                                   \
    static bool gtest_##TestName##_defined_ GTEST_ATTRIBUTE_UNUSED_ =    \
        GTEST_TYPED_TEST_CASE_P_STATE_(CaseName).AddTestName(__FILE__,   \
                                                             __LINE__,   \
                                                             #CaseName,  \
                                                             #TestName); \
  }                                                                      \
  template <typename TypeParam>                                          \
  static void cuda_typed_test_p_##CaseName##_##TestName()

#endif  // closing endif for header file include guard
