//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
#include "RAJA/RAJA.hpp"
#include "counter.hpp"

#include "gtest/gtest.h"

RAJA::Platform* plugin_test_capture_platform_active = nullptr;
int*            plugin_test_capture_counter_pre     = nullptr;
int*            plugin_test_capture_counter_post    = nullptr;

RAJA::Platform* plugin_test_launch_platform_active = nullptr;
int*            plugin_test_launch_counter_pre     = nullptr;
int*            plugin_test_launch_counter_post    = nullptr;

struct SetupPluginVars
{
  SetupPluginVars()
  {
    if (s_plugin_storage == nullptr) {
      s_plugin_storage = new PluginStorage;
      plugin_test_capture_platform_active = &s_plugin_storage->capture_platform_active;
      plugin_test_capture_counter_pre     = &s_plugin_storage->capture_counter_pre;
      plugin_test_capture_counter_post    = &s_plugin_storage->capture_counter_post;
      plugin_test_launch_platform_active = &s_plugin_storage->launch_platform_active;
      plugin_test_launch_counter_pre     = &s_plugin_storage->launch_counter_pre;
      plugin_test_launch_counter_post    = &s_plugin_storage->launch_counter_post;
    }

    s_plugin_storage->capture_platform_active = RAJA::Platform::undefined;
    s_plugin_storage->capture_counter_pre     = 0;
    s_plugin_storage->capture_counter_post    = 0;
    s_plugin_storage->launch_platform_active = RAJA::Platform::undefined;
    s_plugin_storage->launch_counter_pre     = 0;
    s_plugin_storage->launch_counter_post    = 0;
  }

  SetupPluginVars(SetupPluginVars const&) = delete;
  SetupPluginVars(SetupPluginVars &&) = delete;
  SetupPluginVars& operator=(SetupPluginVars const&) = delete;
  SetupPluginVars& operator=(SetupPluginVars &&) = delete;

  ~SetupPluginVars()
  {
    if (s_plugin_storage != nullptr) {
      delete s_plugin_storage; s_plugin_storage = nullptr;
      plugin_test_capture_platform_active = nullptr;
      plugin_test_capture_counter_pre     = nullptr;
      plugin_test_capture_counter_post    = nullptr;
      plugin_test_launch_platform_active = nullptr;
      plugin_test_launch_counter_pre     = nullptr;
      plugin_test_launch_counter_post    = nullptr;
    }
  }

private:
  struct PluginStorage
  {
    RAJA::Platform capture_platform_active;
    int            capture_counter_pre;
    int            capture_counter_post;
    RAJA::Platform launch_platform_active;
    int            launch_counter_pre;
    int            launch_counter_post;
  };

  static PluginStorage* s_plugin_storage;
};

typename SetupPluginVars::PluginStorage* SetupPluginVars::s_plugin_storage = nullptr;


struct PluginTestCallable
{
  PluginTestCallable(RAJA::Platform* capture_platform_optr,
                     int*            capture_count_pre_optr,
                     int*            capture_count_post_optr,
                     RAJA::Platform* launch_platform_optr,
                     int*            launch_count_pre_optr,
                     int*            launch_count_post_optr)
    : m_capture_platform_optr  (capture_platform_optr)
    , m_capture_count_pre_optr (capture_count_pre_optr)
    , m_capture_count_post_optr(capture_count_post_optr)
    , m_launch_platform_optr  (launch_platform_optr)
    , m_launch_count_pre_optr (launch_count_pre_optr)
    , m_launch_count_post_optr(launch_count_post_optr)

    , m_capture_platform_iptr  (plugin_test_capture_platform_active)
    , m_capture_count_pre_iptr (plugin_test_capture_counter_pre)
    , m_capture_count_post_iptr(plugin_test_capture_counter_post)
    , m_launch_platform_iptr  (plugin_test_launch_platform_active)
    , m_launch_count_pre_iptr (plugin_test_launch_counter_pre)
    , m_launch_count_post_iptr(plugin_test_launch_counter_post)
  { }

  PluginTestCallable(PluginTestCallable const& rhs)
    : m_capture_platform_optr  (rhs.m_capture_platform_optr)
    , m_capture_count_pre_optr (rhs.m_capture_count_pre_optr)
    , m_capture_count_post_optr(rhs.m_capture_count_post_optr)
    , m_launch_platform_optr  (rhs.m_launch_platform_optr)
    , m_launch_count_pre_optr (rhs.m_launch_count_pre_optr)
    , m_launch_count_post_optr(rhs.m_launch_count_post_optr)

    , m_capture_platform_iptr  (rhs.m_capture_platform_iptr)
    , m_capture_count_pre_iptr (rhs.m_capture_count_pre_iptr)
    , m_capture_count_post_iptr(rhs.m_capture_count_post_iptr)
    , m_launch_platform_iptr  (rhs.m_launch_platform_iptr)
    , m_launch_count_pre_iptr (rhs.m_launch_count_pre_iptr)
    , m_launch_count_post_iptr(rhs.m_launch_count_post_iptr)

    , m_capture_platform  (rhs.m_capture_platform)
    , m_capture_count_pre (rhs.m_capture_count_pre)
    , m_capture_count_post(rhs.m_capture_count_post)
  {
    if (m_capture_platform == RAJA::Platform::undefined &&
        *m_capture_platform_iptr != RAJA::Platform::undefined) {
      m_capture_platform   = *m_capture_platform_iptr;
      m_capture_count_pre  = *m_capture_count_pre_iptr;
      m_capture_count_post = *m_capture_count_post_iptr;
    }
  }

  PluginTestCallable(PluginTestCallable && rhs)
    : m_capture_platform_optr  (rhs.m_capture_platform_optr)
    , m_capture_count_pre_optr (rhs.m_capture_count_pre_optr)
    , m_capture_count_post_optr(rhs.m_capture_count_post_optr)
    , m_launch_platform_optr  (rhs.m_launch_platform_optr)
    , m_launch_count_pre_optr (rhs.m_launch_count_pre_optr)
    , m_launch_count_post_optr(rhs.m_launch_count_post_optr)

    , m_capture_platform_iptr  (rhs.m_capture_platform_iptr)
    , m_capture_count_pre_iptr (rhs.m_capture_count_pre_iptr)
    , m_capture_count_post_iptr(rhs.m_capture_count_post_iptr)
    , m_launch_platform_iptr  (rhs.m_launch_platform_iptr)
    , m_launch_count_pre_iptr (rhs.m_launch_count_pre_iptr)
    , m_launch_count_post_iptr(rhs.m_launch_count_post_iptr)

    , m_capture_platform  (rhs.m_capture_platform)
    , m_capture_count_pre (rhs.m_capture_count_pre)
    , m_capture_count_post(rhs.m_capture_count_post)
  {
    rhs.clear();
  }

  PluginTestCallable& operator=(PluginTestCallable const&) = default;
  PluginTestCallable& operator=(PluginTestCallable && rhs)
  {
    if (this != &rhs) {
      m_capture_platform_optr   = rhs.m_capture_platform_optr;
      m_capture_count_pre_optr  = rhs.m_capture_count_pre_optr;
      m_capture_count_post_optr = rhs.m_capture_count_post_optr;
      m_launch_platform_optr   = rhs.m_launch_platform_optr;
      m_launch_count_pre_optr  = rhs.m_launch_count_pre_optr;
      m_launch_count_post_optr = rhs.m_launch_count_post_optr;

      m_capture_platform_iptr   = rhs.m_capture_platform_iptr;
      m_capture_count_pre_iptr  = rhs.m_capture_count_pre_iptr;
      m_capture_count_post_iptr = rhs.m_capture_count_post_iptr;
      m_launch_platform_iptr   = rhs.m_launch_platform_iptr;
      m_launch_count_pre_iptr  = rhs.m_launch_count_pre_iptr;
      m_launch_count_post_iptr = rhs.m_launch_count_post_iptr;

      m_capture_platform   = rhs.m_capture_platform;
      m_capture_count_pre  = rhs.m_capture_count_pre;
      m_capture_count_post = rhs.m_capture_count_post;

      rhs.clear();
    }

    return *this;
  }

  void operator()(int i) const
  {
    m_capture_platform_optr  [i] = m_capture_platform;
    m_capture_count_pre_optr [i] = m_capture_count_pre;
    m_capture_count_post_optr[i] = m_capture_count_post;
    m_launch_platform_optr  [i] = *m_launch_platform_iptr;
    m_launch_count_pre_optr [i] = *m_launch_count_pre_iptr;
    m_launch_count_post_optr[i] = *m_launch_count_post_iptr;
  }

  void operator()(int count, int i) const
  {
    RAJA_UNUSED_VAR(count);
    operator()(i);
  }

private:
  RAJA::Platform* m_capture_platform_optr   = nullptr;
  int*            m_capture_count_pre_optr  = nullptr;
  int*            m_capture_count_post_optr = nullptr;
  RAJA::Platform* m_launch_platform_optr   = nullptr;
  int*            m_launch_count_pre_optr  = nullptr;
  int*            m_launch_count_post_optr = nullptr;

  const RAJA::Platform* m_capture_platform_iptr   = nullptr;
  const int*            m_capture_count_pre_iptr  = nullptr;
  const int*            m_capture_count_post_iptr = nullptr;
  const RAJA::Platform* m_launch_platform_iptr   = nullptr;
  const int*            m_launch_count_pre_iptr  = nullptr;
  const int*            m_launch_count_post_iptr = nullptr;

  RAJA::Platform  m_capture_platform   = RAJA::Platform::undefined;
  int             m_capture_count_pre  = -1;
  int             m_capture_count_post = -1;

  void clear()
  {
    m_capture_platform_optr   = nullptr;
    m_capture_count_pre_optr  = nullptr;
    m_capture_count_post_optr = nullptr;
    m_launch_platform_optr   = nullptr;
    m_launch_count_pre_optr  = nullptr;
    m_launch_count_post_optr = nullptr;

    m_capture_platform_iptr   = nullptr;
    m_capture_count_pre_iptr  = nullptr;
    m_capture_count_post_iptr = nullptr;
    m_launch_platform_iptr   = nullptr;
    m_launch_count_pre_iptr  = nullptr;
    m_launch_count_post_iptr = nullptr;

    m_capture_platform   = RAJA::Platform::undefined;
    m_capture_count_pre  = -1;
    m_capture_count_post = -1;
  }
};


// Check that the plugin is called with the right Platform.
// Check that the plugin is called the correct number of times,
// once before and after each kernel capture for the capture counter,
// once before and after each kernel invocation for the launch counter.

// test with basic forall
TEST(PluginTest, ForAllCounter)
{
  SetupPluginVars spv;

  RAJA::Platform* capture_platform   = new RAJA::Platform[10];
  int*            capture_count_pre  = new int[10];
  int*            capture_count_post = new int[10];

  RAJA::Platform* launch_platform   = new RAJA::Platform[10];
  int*            launch_count_pre  = new int[10];
  int*            launch_count_post = new int[10];

  for (int i = 0; i < 10; i++) {

    RAJA::forall<RAJA::seq_exec>(
      RAJA::RangeSegment(i,i+1),
      PluginTestCallable{capture_platform,
                         capture_count_pre,
                         capture_count_post,
                         launch_platform,
                         launch_count_pre,
                         launch_count_post}
    );

    ASSERT_EQ(capture_platform[i]  , RAJA::Platform::host);
    ASSERT_EQ(capture_count_pre[i] , i+1                 );
    ASSERT_EQ(capture_count_post[i], i                   );

    ASSERT_EQ(launch_platform[i]  , RAJA::Platform::host);
    ASSERT_EQ(launch_count_pre[i] , i+1                 );
    ASSERT_EQ(launch_count_post[i], i                   );
  }

  ASSERT_EQ(*plugin_test_capture_platform_active, RAJA::Platform::undefined);
  ASSERT_EQ(*plugin_test_capture_counter_pre,  10);
  ASSERT_EQ(*plugin_test_capture_counter_post, 10);

  ASSERT_EQ(*plugin_test_launch_platform_active,  RAJA::Platform::undefined);
  ASSERT_EQ(*plugin_test_launch_counter_pre,  10);
  ASSERT_EQ(*plugin_test_launch_counter_post, 10);

  delete[] capture_platform  ;
  delete[] capture_count_pre ;
  delete[] capture_count_post;
  delete[] launch_platform  ;
  delete[] launch_count_pre ;
  delete[] launch_count_post;
}

// test with basic forall_Icount
TEST(PluginTest, ForAllICountCounter)
{
  SetupPluginVars spv;

  RAJA::Platform* capture_platform   = new RAJA::Platform[10];
  int*            capture_count_pre  = new int[10];
  int*            capture_count_post = new int[10];

  RAJA::Platform* launch_platform   = new RAJA::Platform[10];
  int*            launch_count_pre  = new int[10];
  int*            launch_count_post = new int[10];

  for (int i = 0; i < 10; i++) {

    RAJA::forall_Icount<RAJA::seq_exec>(
      RAJA::RangeSegment(i,i+1), i,
      PluginTestCallable{capture_platform,
                         capture_count_pre,
                         capture_count_post,
                         launch_platform,
                         launch_count_pre,
                         launch_count_post}
    );

    ASSERT_EQ(capture_platform[i]  , RAJA::Platform::host);
    ASSERT_EQ(capture_count_pre[i] , i+1                 );
    ASSERT_EQ(capture_count_post[i], i                   );

    ASSERT_EQ(launch_platform[i]  , RAJA::Platform::host);
    ASSERT_EQ(launch_count_pre[i] , i+1                 );
    ASSERT_EQ(launch_count_post[i], i                   );
  }

  ASSERT_EQ(*plugin_test_capture_platform_active, RAJA::Platform::undefined);
  ASSERT_EQ(*plugin_test_capture_counter_pre,  10);
  ASSERT_EQ(*plugin_test_capture_counter_post, 10);

  ASSERT_EQ(*plugin_test_launch_platform_active,  RAJA::Platform::undefined);
  ASSERT_EQ(*plugin_test_launch_counter_pre,  10);
  ASSERT_EQ(*plugin_test_launch_counter_post, 10);

  delete[] capture_platform  ;
  delete[] capture_count_pre ;
  delete[] capture_count_post;
  delete[] launch_platform  ;
  delete[] launch_count_pre ;
  delete[] launch_count_post;
}

// test with IndexSet forall
TEST(PluginTest, ForAllIdxSetCounter)
{
  SetupPluginVars spv;

  RAJA::Platform* capture_platform   = new RAJA::Platform[10];
  int*            capture_count_pre  = new int[10];
  int*            capture_count_post = new int[10];

  RAJA::Platform* launch_platform   = new RAJA::Platform[10];
  int*            launch_count_pre  = new int[10];
  int*            launch_count_post = new int[10];

  for (int i = 0; i < 10; i++) {

    RAJA::TypedIndexSet< RAJA::RangeSegment > iset;

    for (int j = i; j < 10; j++) {
      iset.push_back(RAJA::RangeSegment(j, j+1));
    }

    RAJA::forall<RAJA::ExecPolicy<RAJA::seq_segit, RAJA::seq_exec>>(
      iset,
      PluginTestCallable{capture_platform,
                         capture_count_pre,
                         capture_count_post,
                         launch_platform,
                         launch_count_pre,
                         launch_count_post}
    );

    for (int j = i; j < 10; j++) {
      ASSERT_EQ(capture_platform[j]  , RAJA::Platform::host);
      ASSERT_EQ(capture_count_pre[j] , i+1                 );
      ASSERT_EQ(capture_count_post[j], i                   );

      ASSERT_EQ(launch_platform[j]  , RAJA::Platform::host);
      ASSERT_EQ(launch_count_pre[j] , i+1                 );
      ASSERT_EQ(launch_count_post[j], i                   );
    }
  }

  ASSERT_EQ(*plugin_test_capture_platform_active, RAJA::Platform::undefined);
  ASSERT_EQ(*plugin_test_capture_counter_pre,  10);
  ASSERT_EQ(*plugin_test_capture_counter_post, 10);

  ASSERT_EQ(*plugin_test_launch_platform_active,  RAJA::Platform::undefined);
  ASSERT_EQ(*plugin_test_launch_counter_pre,  10);
  ASSERT_EQ(*plugin_test_launch_counter_post, 10);

  delete[] capture_platform  ;
  delete[] capture_count_pre ;
  delete[] capture_count_post;
  delete[] launch_platform  ;
  delete[] launch_count_pre ;
  delete[] launch_count_post;
}

// test with IndexSet forall_Icount
TEST(PluginTest, ForAllIcountIdxSetCounter)
{
  SetupPluginVars spv;

  RAJA::Platform* capture_platform   = new RAJA::Platform[10];
  int*            capture_count_pre  = new int[10];
  int*            capture_count_post = new int[10];

  RAJA::Platform* launch_platform   = new RAJA::Platform[10];
  int*            launch_count_pre  = new int[10];
  int*            launch_count_post = new int[10];

  for (int i = 0; i < 10; i++) {

    RAJA::TypedIndexSet< RAJA::RangeSegment > iset;

    for (int j = i; j < 10; j++) {
      iset.push_back(RAJA::RangeSegment(j, j+1));
    }

    RAJA::forall_Icount<RAJA::ExecPolicy<RAJA::seq_segit, RAJA::seq_exec>>(
      iset,
      PluginTestCallable{capture_platform,
                         capture_count_pre,
                         capture_count_post,
                         launch_platform,
                         launch_count_pre,
                         launch_count_post}
    );

    for (int j = i; j < 10; j++) {
      ASSERT_EQ(capture_platform[j]  , RAJA::Platform::host);
      ASSERT_EQ(capture_count_pre[j] , i+1                 );
      ASSERT_EQ(capture_count_post[j], i                   );

      ASSERT_EQ(launch_platform[j]  , RAJA::Platform::host);
      ASSERT_EQ(launch_count_pre[j] , i+1                 );
      ASSERT_EQ(launch_count_post[j], i                   );
    }
  }

  ASSERT_EQ(*plugin_test_capture_platform_active, RAJA::Platform::undefined);
  ASSERT_EQ(*plugin_test_capture_counter_pre,  10);
  ASSERT_EQ(*plugin_test_capture_counter_post, 10);

  ASSERT_EQ(*plugin_test_launch_platform_active,  RAJA::Platform::undefined);
  ASSERT_EQ(*plugin_test_launch_counter_pre,  10);
  ASSERT_EQ(*plugin_test_launch_counter_post, 10);

  delete[] capture_platform  ;
  delete[] capture_count_pre ;
  delete[] capture_count_post;
  delete[] launch_platform  ;
  delete[] launch_count_pre ;
  delete[] launch_count_post;
}

// test with multi_policy forall
TEST(PluginTest, ForAllMultiPolicyCounter)
{
  SetupPluginVars spv;

  RAJA::Platform* capture_platform   = new RAJA::Platform[10];
  int*            capture_count_pre  = new int[10];
  int*            capture_count_post = new int[10];

  RAJA::Platform* launch_platform   = new RAJA::Platform[10];
  int*            launch_count_pre  = new int[10];
  int*            launch_count_post = new int[10];

  auto mp = RAJA::make_multi_policy<RAJA::seq_exec, RAJA::loop_exec>(
      [](const RAJA::RangeSegment &r) {
        if (*(r.begin()) < 5) {
          return 0;
        } else {
          return 1;
        }
      });

  for (int i = 0; i < 5; i++) {

    RAJA::forall(mp,
      RAJA::RangeSegment(i,i+1),
      PluginTestCallable{capture_platform,
                         capture_count_pre,
                         capture_count_post,
                         launch_platform,
                         launch_count_pre,
                         launch_count_post}
    );

    ASSERT_EQ(capture_platform[i]  , RAJA::Platform::host);
    ASSERT_EQ(capture_count_pre[i] , i+1                 );
    ASSERT_EQ(capture_count_post[i], i                   );

    ASSERT_EQ(launch_platform[i]  , RAJA::Platform::host);
    ASSERT_EQ(launch_count_pre[i] , i+1                 );
    ASSERT_EQ(launch_count_post[i], i                   );
  }

  ASSERT_EQ(*plugin_test_capture_platform_active, RAJA::Platform::undefined);
  ASSERT_EQ(*plugin_test_capture_counter_pre,  5);
  ASSERT_EQ(*plugin_test_capture_counter_post, 5);

  ASSERT_EQ(*plugin_test_launch_platform_active,  RAJA::Platform::undefined);
  ASSERT_EQ(*plugin_test_launch_counter_pre,  5);
  ASSERT_EQ(*plugin_test_launch_counter_post, 5);

  for (int i = 5; i < 10; i++) {

    RAJA::forall(mp,
      RAJA::RangeSegment(i,i+1),
      PluginTestCallable{capture_platform,
                         capture_count_pre,
                         capture_count_post,
                         launch_platform,
                         launch_count_pre,
                         launch_count_post}
    );

    ASSERT_EQ(capture_platform[i]  , RAJA::Platform::host);
    ASSERT_EQ(capture_count_pre[i] , i+1                 );
    ASSERT_EQ(capture_count_post[i], i                   );

    ASSERT_EQ(launch_platform[i]  , RAJA::Platform::host);
    ASSERT_EQ(launch_count_pre[i] , i+1                 );
    ASSERT_EQ(launch_count_post[i], i                   );
  }

  ASSERT_EQ(*plugin_test_capture_platform_active, RAJA::Platform::undefined);
  ASSERT_EQ(*plugin_test_capture_counter_pre,  10);
  ASSERT_EQ(*plugin_test_capture_counter_post, 10);

  ASSERT_EQ(*plugin_test_launch_platform_active,  RAJA::Platform::undefined);
  ASSERT_EQ(*plugin_test_launch_counter_pre,  10);
  ASSERT_EQ(*plugin_test_launch_counter_post, 10);

  delete[] capture_platform  ;
  delete[] capture_count_pre ;
  delete[] capture_count_post;
  delete[] launch_platform  ;
  delete[] launch_count_pre ;
  delete[] launch_count_post;
}
