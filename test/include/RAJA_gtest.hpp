/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   RAJA header file for GPU google test convenience macros
 *
 ******************************************************************************
 */

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-21, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef __RAJA_gtest_HPP__
#define __RAJA_gtest_HPP__

#ifdef RAJA_COMPILER_MSVC
// disable some warnings for MSVC that we can't control, because they're emitted
// by googletest headers
#pragma warning( disable : 4244 )  // Force msvc to not emit conversion warning
#pragma warning( disable : 4389 )  // Force msvc to not emit conversion warning
#endif

#include "gtest/gtest.h"

#define GPU_TEST(X, Y)                 \
  static void gpu_test_##X##_##Y();    \
  TEST(X, Y) { gpu_test_##X##_##Y(); } \
  static void gpu_test_##X##_##Y()

#define GPU_TEST_F(test_fixture, test_name)                  \
  static void gpu_test_f_##test_fixture##_##test_name();     \
  GTEST_TEST_(test_fixture,                                   \
              test_name,                                      \
              test_fixture,                                   \
              ::testing::internal::GetTypeId<test_fixture>()) \
  {                                                           \
    gpu_test_f_##test_fixture##_##test_name();               \
  }                                                           \
  static void gpu_test_f_##test_fixture##_##test_name()

#define GPU_TEST_P(test_case_name, test_name)                               \
  template <typename Invocable>                                              \
  static void gtest_gpu_##test_case_name##_##test_name(Invocable &&);       \
  class GTEST_TEST_CLASS_NAME_(test_case_name, test_name)                    \
      : public test_case_name                                                \
  {                                                                          \
  public:                                                                    \
    GTEST_TEST_CLASS_NAME_(test_case_name, test_name)() {}                   \
    virtual void TestBody()                                                  \
    {                                                                        \
      gtest_gpu_##test_case_name##_##test_name([&] { return GetParam(); }); \
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
  static void gtest_gpu_##test_case_name##_##test_name(Invocable &&GetParam)

#define GPU_TYPED_TEST_P(SuiteName, TestName)                           \
    namespace GTEST_SUITE_NAMESPACE_(SuiteName) {                       \
      template <typename gtest_TypeParam_>                              \
      class TestName : public SuiteName<gtest_TypeParam_> {             \
       private:                                                         \
        typedef SuiteName<gtest_TypeParam_> TestFixture;                \
        typedef gtest_TypeParam_ TypeParam;                             \
       public:                                                          \
        void TestBody() override;                                       \
      };                                                                \
      static bool gtest_##TestName##_defined_ GTEST_ATTRIBUTE_UNUSED_ = \
          GTEST_TYPED_TEST_SUITE_P_STATE_(SuiteName).AddTestName(       \
              __FILE__, __LINE__, GTEST_STRINGIFY_(SuiteName),          \
              GTEST_STRINGIFY_(TestName));                              \
    }                                                                   \
    template <typename gtest_TypeParam_>                                \
    void GTEST_SUITE_NAMESPACE_(                                        \
        SuiteName)::TestName<gtest_TypeParam_>::TestBody()


#ifdef RAJA_COMPILER_MSVC
#pragma warning( default : 4244 )  // reenable warning
#pragma warning( default : 4389 )  // reenable warning
#endif

#endif  // closing endif for header file include guard
