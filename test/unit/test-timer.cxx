#include "gtest/gtest.h"

#include "RAJA/Timer.hxx"

#include <string>
#include <iostream>

#include <thread>
#include <chrono>


TEST(TimerTest, No1)
{
  RAJA::Timer timer;

  timer.start("test_timer"); 

  std::cout << "Printing 1000 stars...\n";
  for (int i=0; i<1000; ++i) std::cout << "*";
  std::cout << std::endl;

  timer.stop();

  RAJA::Timer::ElapsedType elapsed = timer.elapsed();

  EXPECT_TRUE( elapsed > 0.0 );

  std::cout << "Printing 1000 stars took " << elapsed << " seconds.";
}

 
TEST(TimerTest, No2)
{ 
  RAJA::Timer timer;

  timer.start("test_timer");

  for (int i=2; i>0; --i) {
    std::cout << i << std::endl;
    std::this_thread::sleep_for(std::chrono::seconds(1));
  }

  timer.stop();

  RAJA::Timer::ElapsedType elapsed = timer.elapsed();

  EXPECT_TRUE( elapsed > 2.0 );

  std::cout << "I slept for " << elapsed << " seconds. Refreshing!";
  std::cout << std::endl; 
}
