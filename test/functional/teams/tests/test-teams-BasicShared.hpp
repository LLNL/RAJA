//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-22, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef __TEST_TEAMS_BASIC_SHARED_HPP__
#define __TEST_TEAMS_BASIC_SHARED_HPP__

#include <numeric>

template <typename WORKING_RES, typename LAUNCH_POLICY, typename TEAM_POLICY, typename THREAD_POLICY>
void TeamsBasicSharedTestImpl()
{

  int N = 1000;

  camp::resources::Resource working_res{WORKING_RES::get_default()};
  int* working_array;
  int* check_array;
  int* test_array;

  allocateForallTestData<int>(N*N,
                             working_res,
                             &working_array,
                             &check_array,
                             &test_array);
  


  //Select platform
  RAJA::expt::ExecPlace select_cpu_or_gpu;
  if (working_res.get_platform()  == camp::resources::Platform::host){
    select_cpu_or_gpu = RAJA::expt::HOST;
  }else{  
    select_cpu_or_gpu = RAJA::expt::DEVICE;
  }


  RAJA::expt::launch<LAUNCH_POLICY>(select_cpu_or_gpu,
    RAJA::expt::Grid(RAJA::expt::Teams(N), RAJA::expt::Threads(N)),
        [=] RAJA_HOST_DEVICE(RAJA::expt::LaunchContext ctx) {

          RAJA::expt::loop<TEAM_POLICY>(ctx, RAJA::RangeSegment(0, N), [&](int r) {

                // Array shared within threads of the same team
                RAJA_TEAM_SHARED int s_A[1];

                RAJA::expt::loop<THREAD_POLICY>(ctx, RAJA::RangeSegment(0, 1), [&](int c) {
                    s_A[c] = r; 
                });

                ctx.teamSync();

                //broadcast shared value to all threads and write to array
                RAJA::expt::loop<THREAD_POLICY>(ctx, RAJA::RangeSegment(0, N), [&](int c) {
                    const int idx = c + N*r;
                    working_array[idx] = s_A[0];
                });  // loop j

              });  // loop r
        });  // outer lambda



  working_res.memcpy(check_array, working_array, sizeof(int) * N*N);

  for(int r = 0; r < N; ++r) {
    for (int c = 0; c < N; c++) {
      ASSERT_EQ(r, check_array[c + r*N]);
    }
  }

  deallocateForallTestData<int>(working_res,
                               working_array,
                               check_array,
                               test_array);
}


TYPED_TEST_SUITE_P(TeamsBasicSharedTest);
template <typename T>
class TeamsBasicSharedTest : public ::testing::Test
{
};

TYPED_TEST_P(TeamsBasicSharedTest, BasicSharedTeams)
{

  using WORKING_RES = typename camp::at<TypeParam, camp::num<0>>::type;
  using LAUNCH_POLICY = typename camp::at<typename camp::at<TypeParam,camp::num<1>>::type, camp::num<0>>::type;
  using TEAM_POLICY = typename camp::at<typename camp::at<TypeParam,camp::num<1>>::type, camp::num<1>>::type;
  using THREAD_POLICY = typename camp::at<typename camp::at<TypeParam,camp::num<1>>::type, camp::num<2>>::type;

  TeamsBasicSharedTestImpl<WORKING_RES, LAUNCH_POLICY, TEAM_POLICY, THREAD_POLICY>();


}

REGISTER_TYPED_TEST_SUITE_P(TeamsBasicSharedTest,
                            BasicSharedTeams);

#endif  // __TEST_BASIC_SHARED_HPP__
