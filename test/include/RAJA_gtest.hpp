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

#define CUDA_TEST_P(test_case_name, test_name)                               \
  template <typename Invocable>                                              \
  static void gtest_cuda_##test_case_name##_##test_name(Invocable &&);       \
  class GTEST_TEST_CLASS_NAME_(test_case_name, test_name)                    \
      : public test_case_name                                                \
  {                                                                          \
  public:                                                                    \
    GTEST_TEST_CLASS_NAME_(test_case_name, test_name)() {}                   \
    virtual void TestBody()                                                  \
    {                                                                        \
      gtest_cuda_##test_case_name##_##test_name([&] { return GetParam(); }); \
    }                                                                        \
                                                                             \
  private:                                                                   \
    static int AddToRegistry()                                               \
    {                                                                        \
      ::testing::UnitTest::GetInstance()                                     \
          ->parameterized_test_registry()                                    \
          .GetTestCasePatternHolder<test_case_name>(                         \
              #test_case_name,                                               \
              ::testing::internal::CodeLocation(__FILE__, __LINE__))         \
          ->AddTestPattern(                                                  \
              #test_case_name,                                               \
              #test_name,                                                    \
              new ::testing::internal::TestMetaFactory<                      \
                  GTEST_TEST_CLASS_NAME_(test_case_name, test_name)>());     \
      return 0;                                                              \
    }                                                                        \
    static int gtest_registering_dummy_ GTEST_ATTRIBUTE_UNUSED_;             \
    GTEST_DISALLOW_COPY_AND_ASSIGN_(GTEST_TEST_CLASS_NAME_(test_case_name,   \
                                                           test_name));      \
  };                                                                         \
  int GTEST_TEST_CLASS_NAME_(test_case_name,                                 \
                             test_name)::gtest_registering_dummy_ =          \
      GTEST_TEST_CLASS_NAME_(test_case_name, test_name)::AddToRegistry();    \
  template <typename Invocable>                                              \
  static void gtest_cuda_##test_case_name##_##test_name(Invocable &&GetParam)


#define CUDA_TYPED_TEST_P(CaseName, TestName)                            \
  namespace GTEST_CASE_NAMESPACE_(CaseName)                              \
  {                                                                      \
    template <typename gtest_TypeParam_>                                 \
    class TestName : public CaseName<gtest_TypeParam_>                   \
    {                                                                    \
    private:                                                             \
      typedef CaseName<gtest_TypeParam_> TestFixture;                    \
      typedef gtest_TypeParam_ TypeParam;                                \
                                                                         \
    public:                                                              \
      virtual void TestBody();                                           \
    };                                                                   \
    static bool gtest_##TestName##_defined_ GTEST_ATTRIBUTE_UNUSED_ =    \
        GTEST_TYPED_TEST_CASE_P_STATE_(CaseName).AddTestName(__FILE__,   \
                                                             __LINE__,   \
                                                             #CaseName,  \
                                                             #TestName); \
  }                                                                      \
  template <typename TypeParam>                                          \
  void GTEST_CASE_NAMESPACE_(CaseName)::TestName<TypeParam>::TestBody()

#endif  // closing endif for header file include guard
