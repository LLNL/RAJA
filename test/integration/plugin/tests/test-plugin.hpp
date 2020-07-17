//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

///
/// Header file containing basic functional tests for atomic operations with forall and views.
///

#ifndef __TEST_PLUGIN_HPP__
#define __TEST_PLUGIN_HPP__

#include "RAJA/RAJA.hpp"
#include "RAJA/util/macros.hpp"
#include "counter.hpp"

#include "gtest/gtest.h"


CounterData* plugin_test_data = nullptr;

camp::resources::Resource* plugin_test_resource = nullptr;

struct SetupPluginVars
{
  SetupPluginVars(camp::resources::Resource const& test_resource)
    : m_test_resource(test_resource)
  {
    // ASSERT_EQ(plugin_test_data, nullptr);
    // ASSERT_EQ(plugin_test_resource, nullptr);

    plugin_test_data = m_test_resource.allocate<CounterData>(1);
    plugin_test_resource = &m_test_resource;

    CounterData data;
    data.capture_platform_active = RAJA::Platform::undefined;
    data.capture_counter_pre     = 0;
    data.capture_counter_post    = 0;
    data.launch_platform_active = RAJA::Platform::undefined;
    data.launch_counter_pre     = 0;
    data.launch_counter_post    = 0;

    m_test_resource.memcpy(plugin_test_data, &data, sizeof(CounterData));
  }

  SetupPluginVars(SetupPluginVars const&) = delete;
  SetupPluginVars(SetupPluginVars &&) = delete;
  SetupPluginVars& operator=(SetupPluginVars const&) = delete;
  SetupPluginVars& operator=(SetupPluginVars &&) = delete;

  ~SetupPluginVars()
  {
    // ASSERT_NE(plugin_test_data, nullptr);
    // ASSERT_NE(plugin_test_resource, nullptr);

    m_test_resource.deallocate(plugin_test_data);
    plugin_test_data = nullptr;
    plugin_test_resource = nullptr;
  }

private:
  camp::resources::Resource m_test_resource;
};


struct PluginTestCallable
{
  PluginTestCallable(CounterData* data_optr)
    : m_data_optr(data_optr)
    , m_data_iptr(plugin_test_data)
  {
    clear_data();
  }

  RAJA_HOST_DEVICE PluginTestCallable(PluginTestCallable const& rhs)
    : m_data_optr(rhs.m_data_optr)
    , m_data_iptr(rhs.m_data_iptr)
    , m_data(rhs.m_data)
  {
#if !defined(RAJA_DEVICE_CODE)
    CounterData i_data;
    plugin_test_resource->memcpy(&i_data, m_data_iptr, sizeof(CounterData));

    if (m_data.capture_platform_active == RAJA::Platform::undefined &&
        i_data.capture_platform_active != RAJA::Platform::undefined) {
      m_data = i_data;
    }
#endif
  }

  RAJA_HOST_DEVICE PluginTestCallable(PluginTestCallable && rhs)
    : m_data_optr(rhs.m_data_optr)
    , m_data_iptr(rhs.m_data_iptr)
    , m_data(rhs.m_data)
  {
    rhs.clear();
  }

  RAJA_HOST_DEVICE PluginTestCallable& operator=(PluginTestCallable const& rhs)
  {
    if (this != &rhs) {
      m_data_optr = rhs.m_data_optr;
      m_data_iptr = rhs.m_data_iptr;
      m_data      = rhs.m_data;
    }
    return *this;
  }

  RAJA_HOST_DEVICE PluginTestCallable& operator=(PluginTestCallable && rhs)
  {
    if (this != &rhs) {
      m_data_optr = rhs.m_data_optr;
      m_data_iptr = rhs.m_data_iptr;
      m_data      = rhs.m_data;
      rhs.clear();
    }
    return *this;
  }

  RAJA_HOST_DEVICE void operator()(int i) const
  {
    m_data_optr[i].capture_platform_active = m_data.capture_platform_active;
    m_data_optr[i].capture_counter_pre     = m_data.capture_counter_pre;
    m_data_optr[i].capture_counter_post    = m_data.capture_counter_post;
    m_data_optr[i].launch_platform_active = m_data_iptr->launch_platform_active;
    m_data_optr[i].launch_counter_pre     = m_data_iptr->launch_counter_pre;
    m_data_optr[i].launch_counter_post    = m_data_iptr->launch_counter_post;
  }

  RAJA_HOST_DEVICE void operator()(int count, int i) const
  {
    RAJA_UNUSED_VAR(count);
    operator()(i);
  }

private:
        CounterData* m_data_optr = nullptr;
  const CounterData* m_data_iptr = nullptr;
        CounterData  m_data;


  RAJA_HOST_DEVICE void clear()
  {
    m_data_optr = nullptr;
    m_data_iptr = nullptr;
    clear_data();
  }

  RAJA_HOST_DEVICE void clear_data()
  {
    m_data.capture_platform_active = RAJA::Platform::undefined;
    m_data.capture_counter_pre     = -1;
    m_data.capture_counter_post    = -1;
    m_data.launch_platform_active = RAJA::Platform::undefined;
    m_data.launch_counter_pre     = -1;
    m_data.launch_counter_post    = -1;
  }
};


// Check that the plugin is called with the right Platform.
// Check that the plugin is called the correct number of times,
// once before and after each kernel capture for the capture counter,
// once before and after each kernel invocation for the launch counter.

// test with basic forall
template <typename ExecPolicy,
          typename WORKINGRES,
          RAJA::Platform PLATFORM>
void PluginForallTestImpl()
{
  SetupPluginVars spv(WORKINGRES{});

  CounterData* data = plugin_test_resource->allocate<CounterData>(10);

  for (int i = 0; i < 10; i++) {

    RAJA::forall<ExecPolicy>(
      RAJA::RangeSegment(i,i+1),
      PluginTestCallable{data}
    );

    CounterData loop_data;
    plugin_test_resource->memcpy(&loop_data, &data[i], sizeof(CounterData));
    ASSERT_EQ(loop_data.capture_platform_active, PLATFORM);
    ASSERT_EQ(loop_data.capture_counter_pre,     i+1);
    ASSERT_EQ(loop_data.capture_counter_post,    i);
    ASSERT_EQ(loop_data.launch_platform_active, PLATFORM);
    ASSERT_EQ(loop_data.launch_counter_pre,     i+1);
    ASSERT_EQ(loop_data.launch_counter_post,    i);
  }

  CounterData plugin_data;
  plugin_test_resource->memcpy(&plugin_data, plugin_test_data, sizeof(CounterData));
  ASSERT_EQ(plugin_data.capture_platform_active, RAJA::Platform::undefined);
  ASSERT_EQ(plugin_data.capture_counter_pre,     10);
  ASSERT_EQ(plugin_data.capture_counter_post,    10);
  ASSERT_EQ(plugin_data.launch_platform_active, RAJA::Platform::undefined);
  ASSERT_EQ(plugin_data.launch_counter_pre,     10);
  ASSERT_EQ(plugin_data.launch_counter_post,    10);

  plugin_test_resource->deallocate(data);
}

// test with basic forall_Icount
template <typename ExecPolicy,
          typename WORKINGRES,
          RAJA::Platform PLATFORM>
void PluginForAllICountTestImpl()
{
  SetupPluginVars spv(WORKINGRES{});

  CounterData* data = plugin_test_resource->allocate<CounterData>(10);

  for (int i = 0; i < 10; i++) {

    RAJA::forall_Icount<ExecPolicy>(
      RAJA::RangeSegment(i,i+1), i,
      PluginTestCallable{data}
    );

    CounterData loop_data;
    plugin_test_resource->memcpy(&loop_data, &data[i], sizeof(CounterData));
    ASSERT_EQ(loop_data.capture_platform_active, PLATFORM);
    ASSERT_EQ(loop_data.capture_counter_pre,     i+1);
    ASSERT_EQ(loop_data.capture_counter_post,    i);
    ASSERT_EQ(loop_data.launch_platform_active, PLATFORM);
    ASSERT_EQ(loop_data.launch_counter_pre,     i+1);
    ASSERT_EQ(loop_data.launch_counter_post,    i);
  }

  CounterData plugin_data;
  plugin_test_resource->memcpy(&plugin_data, plugin_test_data, sizeof(CounterData));
  ASSERT_EQ(plugin_data.capture_platform_active, RAJA::Platform::undefined);
  ASSERT_EQ(plugin_data.capture_counter_pre,     10);
  ASSERT_EQ(plugin_data.capture_counter_post,    10);
  ASSERT_EQ(plugin_data.launch_platform_active, RAJA::Platform::undefined);
  ASSERT_EQ(plugin_data.launch_counter_pre,     10);
  ASSERT_EQ(plugin_data.launch_counter_post,    10);

  plugin_test_resource->deallocate(data);
}

// test with IndexSet forall
template <typename ExecPolicy,
          typename WORKINGRES,
          RAJA::Platform PLATFORM>
void PluginForAllIdxSetTestImpl()
{
  SetupPluginVars spv(WORKINGRES{});

  CounterData* data = plugin_test_resource->allocate<CounterData>(10);

  for (int i = 0; i < 10; i++) {

    RAJA::TypedIndexSet< RAJA::RangeSegment > iset;

    for (int j = i; j < 10; j++) {
      iset.push_back(RAJA::RangeSegment(j, j+1));
    }

    RAJA::forall<RAJA::ExecPolicy<RAJA::seq_segit, ExecPolicy>>(
      iset,
      PluginTestCallable{data}
    );

    for (int j = i; j < 10; j++) {
      CounterData loop_data;
      plugin_test_resource->memcpy(&loop_data, &data[j], sizeof(CounterData));
      ASSERT_EQ(loop_data.capture_platform_active, PLATFORM);
      ASSERT_EQ(loop_data.capture_counter_pre,     i+1);
      ASSERT_EQ(loop_data.capture_counter_post,    i);
      ASSERT_EQ(loop_data.launch_platform_active, PLATFORM);
      ASSERT_EQ(loop_data.launch_counter_pre,     i+1);
      ASSERT_EQ(loop_data.launch_counter_post,    i);
    }
  }

  CounterData plugin_data;
  plugin_test_resource->memcpy(&plugin_data, plugin_test_data, sizeof(CounterData));
  ASSERT_EQ(plugin_data.capture_platform_active, RAJA::Platform::undefined);
  ASSERT_EQ(plugin_data.capture_counter_pre,     10);
  ASSERT_EQ(plugin_data.capture_counter_post,    10);
  ASSERT_EQ(plugin_data.launch_platform_active, RAJA::Platform::undefined);
  ASSERT_EQ(plugin_data.launch_counter_pre,     10);
  ASSERT_EQ(plugin_data.launch_counter_post,    10);

  plugin_test_resource->deallocate(data);
}

// test with IndexSet forall_Icount
template <typename ExecPolicy,
          typename WORKINGRES,
          RAJA::Platform PLATFORM>
void PluginForAllIcountIdxSetTestImpl()
{
  SetupPluginVars spv(WORKINGRES{});

  CounterData* data = plugin_test_resource->allocate<CounterData>(10);

  for (int i = 0; i < 10; i++) {

    RAJA::TypedIndexSet< RAJA::RangeSegment > iset;

    for (int j = i; j < 10; j++) {
      iset.push_back(RAJA::RangeSegment(j, j+1));
    }

    RAJA::forall_Icount<RAJA::ExecPolicy<RAJA::seq_segit, ExecPolicy>>(
      iset,
      PluginTestCallable{data}
    );

    for (int j = i; j < 10; j++) {
      CounterData loop_data;
      plugin_test_resource->memcpy(&loop_data, &data[j], sizeof(CounterData));
      ASSERT_EQ(loop_data.capture_platform_active, PLATFORM);
      ASSERT_EQ(loop_data.capture_counter_pre,     i+1);
      ASSERT_EQ(loop_data.capture_counter_post,    i);
      ASSERT_EQ(loop_data.launch_platform_active, PLATFORM);
      ASSERT_EQ(loop_data.launch_counter_pre,     i+1);
      ASSERT_EQ(loop_data.launch_counter_post,    i);
    }
  }

  CounterData plugin_data;
  plugin_test_resource->memcpy(&plugin_data, plugin_test_data, sizeof(CounterData));
  ASSERT_EQ(plugin_data.capture_platform_active, RAJA::Platform::undefined);
  ASSERT_EQ(plugin_data.capture_counter_pre,     10);
  ASSERT_EQ(plugin_data.capture_counter_post,    10);
  ASSERT_EQ(plugin_data.launch_platform_active, RAJA::Platform::undefined);
  ASSERT_EQ(plugin_data.launch_counter_pre,     10);
  ASSERT_EQ(plugin_data.launch_counter_post,    10);

  plugin_test_resource->deallocate(data);
}

// test with multi_policy forall
template <typename ExecPolicy,
          typename WORKINGRES,
          RAJA::Platform PLATFORM>
void PluginForAllMultiPolicyTestImpl()
{
  auto mp = RAJA::make_multi_policy<RAJA::seq_exec, ExecPolicy>(
      [](const RAJA::RangeSegment &r) {
        if (*(r.begin()) < 5) {
          return 0;
        } else {
          return 1;
        }
      });

  {
    SetupPluginVars spv(camp::resources::Host{});

    CounterData* data = plugin_test_resource->allocate<CounterData>(10);

    for (int i = 0; i < 5; i++) {

      RAJA::forall(mp,
        RAJA::RangeSegment(i,i+1),
        PluginTestCallable{data}
      );

      CounterData loop_data;
      plugin_test_resource->memcpy(&loop_data, &data[i], sizeof(CounterData));
      ASSERT_EQ(loop_data.capture_platform_active, RAJA::Platform::host);
      ASSERT_EQ(loop_data.capture_counter_pre,     i+1);
      ASSERT_EQ(loop_data.capture_counter_post,    i);
      ASSERT_EQ(loop_data.launch_platform_active, RAJA::Platform::host);
      ASSERT_EQ(loop_data.launch_counter_pre,     i+1);
      ASSERT_EQ(loop_data.launch_counter_post,    i);
    }

    CounterData plugin_data;
    plugin_test_resource->memcpy(&plugin_data, plugin_test_data, sizeof(CounterData));
    ASSERT_EQ(plugin_data.capture_platform_active, RAJA::Platform::undefined);
    ASSERT_EQ(plugin_data.capture_counter_pre,     5);
    ASSERT_EQ(plugin_data.capture_counter_post,    5);
    ASSERT_EQ(plugin_data.launch_platform_active, RAJA::Platform::undefined);
    ASSERT_EQ(plugin_data.launch_counter_pre,     5);
    ASSERT_EQ(plugin_data.launch_counter_post,    5);

    plugin_test_resource->deallocate(data);
  }

  {
    SetupPluginVars spv(WORKINGRES{});

    CounterData* data = plugin_test_resource->allocate<CounterData>(10);

    for (int i = 0; i < 5; i++) {

      RAJA::forall(mp,
        RAJA::RangeSegment(i+5,i+6),
        PluginTestCallable{data}
      );

      CounterData loop_data;
      plugin_test_resource->memcpy(&loop_data, &data[i+5], sizeof(CounterData));
      ASSERT_EQ(loop_data.capture_platform_active, PLATFORM);
      ASSERT_EQ(loop_data.capture_counter_pre,     i+1);
      ASSERT_EQ(loop_data.capture_counter_post,    i);
      ASSERT_EQ(loop_data.launch_platform_active, PLATFORM);
      ASSERT_EQ(loop_data.launch_counter_pre,     i+1);
      ASSERT_EQ(loop_data.launch_counter_post,    i);
    }

    CounterData plugin_data;
    plugin_test_resource->memcpy(&plugin_data, plugin_test_data, sizeof(CounterData));
    ASSERT_EQ(plugin_data.capture_platform_active, RAJA::Platform::undefined);
    ASSERT_EQ(plugin_data.capture_counter_pre,     5);
    ASSERT_EQ(plugin_data.capture_counter_post,    5);
    ASSERT_EQ(plugin_data.launch_platform_active, RAJA::Platform::undefined);
    ASSERT_EQ(plugin_data.launch_counter_pre,     5);
    ASSERT_EQ(plugin_data.launch_counter_post,    5);

    plugin_test_resource->deallocate(data);
  }
}

TYPED_TEST_SUITE_P(PluginTest);
template <typename T>
class PluginTest : public ::testing::Test
{
};

TYPED_TEST_P(PluginTest, PluginForall)
{
  using ExecPolicy = typename camp::at<TypeParam, camp::num<0>>::type;
  using ResType = typename camp::at<TypeParam, camp::num<1>>::type;
  using PlatformHolder = typename camp::at<TypeParam, camp::num<2>>::type;

  PluginForallTestImpl<ExecPolicy, ResType, PlatformHolder::platform>( );
}

TYPED_TEST_P(PluginTest, PluginForAllICount)
{
  using ExecPolicy = typename camp::at<TypeParam, camp::num<0>>::type;
  using ResType = typename camp::at<TypeParam, camp::num<1>>::type;
  using PlatformHolder = typename camp::at<TypeParam, camp::num<2>>::type;

  PluginForAllICountTestImpl<ExecPolicy, ResType, PlatformHolder::platform>( );
}

TYPED_TEST_P(PluginTest, PluginForAllIdxSet)
{
  using ExecPolicy = typename camp::at<TypeParam, camp::num<0>>::type;
  using ResType = typename camp::at<TypeParam, camp::num<1>>::type;
  using PlatformHolder = typename camp::at<TypeParam, camp::num<2>>::type;

  PluginForAllIdxSetTestImpl<ExecPolicy, ResType, PlatformHolder::platform>( );
}

TYPED_TEST_P(PluginTest, PluginForAllIcountIdxSet)
{
  using ExecPolicy = typename camp::at<TypeParam, camp::num<0>>::type;
  using ResType = typename camp::at<TypeParam, camp::num<1>>::type;
  using PlatformHolder = typename camp::at<TypeParam, camp::num<2>>::type;

  PluginForAllIcountIdxSetTestImpl<ExecPolicy, ResType, PlatformHolder::platform>( );
}

TYPED_TEST_P(PluginTest, PluginForAllMultiPolicy)
{
  using ExecPolicy = typename camp::at<TypeParam, camp::num<0>>::type;
  using ResType = typename camp::at<TypeParam, camp::num<1>>::type;
  using PlatformHolder = typename camp::at<TypeParam, camp::num<2>>::type;

  PluginForAllMultiPolicyTestImpl<ExecPolicy, ResType, PlatformHolder::platform>( );
}

REGISTER_TYPED_TEST_SUITE_P(PluginTest,
                            PluginForall,
                            PluginForAllICount,
                            PluginForAllIdxSet,
                            PluginForAllIcountIdxSet,
                            PluginForAllMultiPolicy);

#endif  //__TEST_PLUGIN_HPP__
