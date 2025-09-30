//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2018-25, Lawrence Livermore National Security, LLC
// and Camp project contributors. See the camp/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "camp/array.hpp"
#include "RAJA_test-base.hpp"

#include "gtest/gtest.h"

TEST(message_handler, initialize) {
  int test = 0;
  auto msg = RAJA::make_message_handler<RAJA::seq_exec>(1, [&](int val) {
    test = val;   
  });

  ASSERT_EQ(msg.test_any(), false);
  ASSERT_EQ(test, 0);
} 

TEST(message_handler, initialize_with_resource) {
  int test = 0;
  auto msg = RAJA::make_message_handler(1, camp::resources::Host(), [&](int val) {
    test = val;   
  });

  ASSERT_EQ(msg.test_any(), false);
  ASSERT_EQ(test, 0);
} 

TEST(message_handler, clear) {
  int test = 0;
  auto msg = RAJA::make_message_handler<RAJA::seq_exec>(1, [&](int val) {
    test = val;   
  });

  auto q = msg.get_queue<RAJA::mpsc_queue>();
  ASSERT_EQ(q.try_post_message(5), true);

  msg.clear();
  msg.wait_all();

  ASSERT_EQ(test, 0);
}

TEST(message_handler, try_post_message) {
  int test = 0;
  auto msg = RAJA::make_message_handler<RAJA::seq_exec>(1, [&](int val) {
    test = val;   
  });

  auto q = msg.get_queue<RAJA::mpsc_queue>();
  ASSERT_EQ(q.try_post_message(5), true);

  ASSERT_EQ(test, 0);
} 

TEST(message_handler, try_post_message_overflow) {
  int test = 0;
  auto msg = RAJA::make_message_handler<RAJA::seq_exec>(1, [&](int val) {
    test = val;   
  });

  auto q = msg.get_queue<RAJA::mpsc_queue>();
  ASSERT_EQ(q.try_post_message(5), true);
  ASSERT_EQ(q.try_post_message(7), false);

  ASSERT_EQ(test, 0);
} 

TEST(message_handler, try_post_message_overwrite) {
  int test = 0;
  auto msg = RAJA::make_message_handler<RAJA::seq_exec>(1, [&](int val) {
    test = val;   
  });

  auto q = msg.get_queue<RAJA::mpsc_queue_overwrite>();
  ASSERT_EQ(q.try_post_message(5), true);
  ASSERT_EQ(q.try_post_message(7), true);

  ASSERT_EQ(test, 0);
} 

TEST(message_handler, wait_all) {
  int test = 0;
  auto msg = RAJA::make_message_handler<RAJA::seq_exec>(1, [&](int val) {
    test = val;   
  });

  auto q = msg.get_queue<RAJA::mpsc_queue>();
  ASSERT_EQ(q.try_post_message(1), true);

  msg.wait_all();

  ASSERT_EQ(test, 1);
}

TEST(message_handler, wait_all_array) {
  camp::array<int, 3> test = {0, 0, 0};
  auto msg = RAJA::make_message_handler<RAJA::seq_exec>(1, 
    [&](camp::array<int, 3> val) {
      test[0] = val[0];   
      test[1] = val[1];
      test[2] = val[2];
    }
  );

  camp::array<int, 3> a{1,2,3};
  auto q = msg.get_queue<RAJA::mpsc_queue>();
  ASSERT_EQ(q.try_post_message(a), true);

  msg.wait_all();

  ASSERT_EQ(test[0], 1);
  ASSERT_EQ(test[1], 2);
  ASSERT_EQ(test[2], 3);
}

TEST(message_handler, wait_all_overflow) {
  int test = 0;
  auto msg = RAJA::make_message_handler<RAJA::seq_exec>(1, [&](int val) {
    test = val;   
  });

  auto q = msg.get_queue<RAJA::mpsc_queue>();
  ASSERT_EQ(q.try_post_message(1), true);
  ASSERT_EQ(q.try_post_message(2), false);

  msg.wait_all();
  ASSERT_EQ(test, 1);
}

TEST(message_handler, wait_all_overwrite) {
  int test = 0;
  auto msg = RAJA::make_message_handler<RAJA::seq_exec>(1, [&](int val) {
    test = val;   
  });

  auto q = msg.get_queue<RAJA::mpsc_queue_overwrite>();
  ASSERT_EQ(q.try_post_message(1), true);
  ASSERT_EQ(q.try_post_message(2), true);

  msg.wait_all();
  ASSERT_EQ(test, 2);
}
