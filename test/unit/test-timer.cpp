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

///
/// Source file containing tests for basic timer operation
///

#include "gtest/gtest.h"

#include "RAJA/util/Timer.hpp"

#include <iostream>
#include <sstream>
#include <string>

#include <chrono>
#include <thread>


TEST(TimerTest, No1)
{
  auto timer = RAJA::Timer();

  timer.start("test_timer");

  {
      std::stringstream sink;
      sink << "Printing 1000 stars...\n";
      for (int i = 0; i < 1000; ++i)
          sink << "*";
      sink << std::endl;
  }

  timer.stop();

  RAJA::Timer::ElapsedType elapsed = timer.elapsed();

  EXPECT_GT(elapsed, 0.0);

  // std::cout << "Printing 1000 stars took " << elapsed << " seconds.";
}


TEST(TimerTest, No2)
{
  RAJA::Timer timer;

  timer.start("test_timer");

  for (int i = 2; i > 0; --i) {
    std::cout << i << std::endl;
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }

  timer.stop();

  RAJA::Timer::ElapsedType elapsed = timer.elapsed();

  EXPECT_GT(elapsed, 0.02);
  EXPECT_LT(elapsed, 0.05);

}
