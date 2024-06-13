//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-24, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

//
// test/include headers
//
#include "RAJA_test-atomic-types.hpp"
#include "RAJA_test-atomicpol.hpp"
#include "RAJA_test-base.hpp"
#include "RAJA_test-camp.hpp"
#include "RAJA_test-forall-data.hpp"
#include "RAJA_test-forall-execpol.hpp"
#include "RAJA_test-index-types.hpp"

//
// Header for tests in ./tests directory
//
// Note: CMake adds ./tests as an include dir for these tests.
//
#include <numeric>
#include <random>

using IdxType = size_t;
constexpr int NUM_ITERS = 100;
#ifdef RAJA_ENABLE_CUDA
constexpr int STRIDE = 32;
constexpr bool STRESS_BEFORE_TEST = true;
constexpr bool NONTESTING_BLOCKS = true;
constexpr int WARP_SIZE = 32;
constexpr int BLOCK_SIZE = 128;
#elif defined(RAJA_ENABLE_HIP)
constexpr int STRIDE = 32;
constexpr bool STRESS_BEFORE_TEST = true;
constexpr bool NONTESTING_BLOCKS = true;
constexpr int WARP_SIZE = 64;
constexpr int BLOCK_SIZE = 128;
#endif

constexpr int STRESS_STRIDE = 64;
constexpr int NUM_STRESS_LINES = 2048;
constexpr int DATA_STRESS_SIZE = NUM_STRESS_LINES * STRESS_STRIDE;

constexpr int PERMUTE_PRIME_BLOCK = 17;
constexpr int PERMUTE_PRIME_GRID = 47;
constexpr int PERMUTE_STRESS = 977;

template <typename Func>
__global__ void dummy_kernel(IdxType index, Func func)
{
  func(index);
}

// Generic test driver for memory litmus tests on the GPU.
template <typename LitmusPolicy>
struct LitmusTestDriver {
public:
  using T = typename LitmusPolicy::DataType;
  struct TestData {
    int block_size;
    int grid_size;

    int stride = STRIDE;

    int pre_stress_iterations = 4;
    int stress_iterations = 12;

    // Number of blocks to run the message-passing litmus test on.
    int testing_blocks;

    // Array to shuffle block indices in a test kernel.
    IdxType* shuffle_block;

    // Barrier integers to synchronize testing threads.
    IdxType* barriers;

    int* data_stress;

    int num_stress_index;
    IdxType* stress_index;

    void allocate(camp::resources::Resource work_res,
                  int grid_size,
                  int block_size,
                  int num_testing_blocks)
    {
      this->grid_size = grid_size;
      this->block_size = block_size;

      testing_blocks = num_testing_blocks;

      shuffle_block = work_res.allocate<IdxType>(grid_size);
      barriers = work_res.allocate<IdxType>(STRIDE);
      data_stress = work_res.allocate<int>(DATA_STRESS_SIZE);

      num_stress_index = 64;
      stress_index = work_res.allocate<IdxType>(num_stress_index);
    }

    void pre_run(camp::resources::Resource work_res)
    {
      std::random_device rand_device;
      // Create a random permutation for the range [0, grid_size)
      std::vector<IdxType> shuffle_block_host(grid_size);
      {
        std::iota(shuffle_block_host.begin(), shuffle_block_host.end(), 0);
        std::shuffle(shuffle_block_host.begin(),
                     shuffle_block_host.end(),
                     std::mt19937{rand_device()});
      }
      work_res.memcpy(shuffle_block,
                      shuffle_block_host.data(),
                      sizeof(IdxType) * grid_size);

      std::vector<IdxType> stress_index_host(num_stress_index);
      {
        std::mt19937 gen{rand_device()};
        std::uniform_int_distribution<IdxType> rnd_stress_offset(0,
                                                                 STRESS_STRIDE);
        std::uniform_int_distribution<IdxType> rnd_stress_dist(
            0, NUM_STRESS_LINES - 1);
        std::generate(stress_index_host.begin(),
                      stress_index_host.end(),
                      [&]() -> IdxType {
                        return rnd_stress_dist(gen) * STRESS_STRIDE;
                      });
      }
      work_res.memcpy(stress_index,
                      stress_index_host.data(),
                      sizeof(IdxType) * num_stress_index);

      work_res.memset(barriers, 0, sizeof(IdxType) * STRIDE);
      work_res.memset(data_stress, 0, sizeof(int) * DATA_STRESS_SIZE);


#if defined(RAJA_ENABLE_CUDA)
      cudaErrchk(cudaDeviceSynchronize());
#endif

#if defined(RAJA_ENABLE_HIP)
      hipErrchk(hipDeviceSynchronize());
#endif
    }

    void deallocate(camp::resources::Resource work_res)
    {
      work_res.deallocate(shuffle_block);
      work_res.deallocate(barriers);
      work_res.deallocate(data_stress);
    }
  };

  RAJA_HOST_DEVICE LitmusTestDriver() {}

  // Run
  static void run()
  {
    int num_blocks = 0;
    {
      LitmusPolicy dummy_policy{};
      TestData dummy_test_data{};
      auto lambda = [=] RAJA_HOST_DEVICE(IdxType index) {
        LitmusTestDriver<LitmusPolicy> test_inst{};
        test_inst.test_main(index, dummy_test_data, dummy_policy);
      };
      RAJA_UNUSED_VAR(lambda);
#ifdef RAJA_ENABLE_CUDA
      cudaErrchk(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
          &num_blocks, dummy_kernel<decltype(lambda)>, BLOCK_SIZE, 0));
      num_blocks *= RAJA::cuda::device_prop().multiProcessorCount;
#endif
#ifdef RAJA_ENABLE_HIP
      hipErrchk(hipOccupancyMaxActiveBlocksPerMultiprocessor(
          &num_blocks, dummy_kernel<decltype(lambda)>, BLOCK_SIZE, 0));
      num_blocks *= RAJA::hip::device_prop().multiProcessorCount;
#endif
    }
    std::cout << "Got num_blocks = " << num_blocks
              << ", block_size = " << BLOCK_SIZE << "\n"
              << std::flush;
    if (num_blocks == 0) {
      FAIL() << "Grid size wasn't set to a valid value.\n";
    }

#ifdef RAJA_ENABLE_CUDA
    using ResourcePolicy = camp::resources::Cuda;
#endif
#ifdef RAJA_ENABLE_HIP
    using ResourcePolicy = camp::resources::Hip;
#endif
    camp::resources::Resource work_res{ResourcePolicy()};

    int num_testing_blocks = num_blocks;
    if (NONTESTING_BLOCKS) {
      num_testing_blocks = num_blocks / 4;
    }

    TestData test_data;
    test_data.allocate(work_res, num_blocks, BLOCK_SIZE, num_testing_blocks);

    LitmusPolicy litmus_test;
    litmus_test.allocate(work_res, num_testing_blocks * BLOCK_SIZE, STRIDE);

#ifdef RAJA_ENABLE_HIP
    using GPUExec = RAJA::hip_exec<BLOCK_SIZE>;
#endif

#ifdef RAJA_ENABLE_CUDA
    using GPUExec = RAJA::cuda_exec<BLOCK_SIZE>;
#endif

    for (int iter = 0; iter < NUM_ITERS; iter++) {
      test_data.pre_run(work_res);
      litmus_test.pre_run(work_res);

      RAJA::forall<GPUExec>(
          RAJA::TypedRangeSegment<IdxType>(0, num_blocks * BLOCK_SIZE),
          [=] RAJA_HOST_DEVICE(IdxType index) {
            LitmusTestDriver<LitmusPolicy> test_inst{};
            test_inst.test_main(index, test_data, litmus_test);
          });

      litmus_test.count_results(work_res);
    }

    litmus_test.verify();

    litmus_test.deallocate(work_res);
    test_data.deallocate(work_res);
  }

private:
  using NormalAtomic = RAJA::atomic_relaxed;

  RAJA_HOST_DEVICE void test_main(IdxType index,
                                  TestData param,
                                  LitmusPolicy test)
  {
    IdxType block_idx = index / param.block_size;
    IdxType thread_idx = index % param.block_size;

    // Permute the thread index, to promote scattering of memory accesses
    // within a block.
    IdxType permute_thread_idx =
        (thread_idx * PERMUTE_PRIME_BLOCK) % param.block_size;

    // Shuffle the block ID randomly according to a permutation array.
    block_idx = param.shuffle_block[block_idx];

    IdxType data_idx = block_idx * param.block_size + thread_idx;

    for (int i = 0; i < param.stride; i++) {

      // Synchronize all blocks before testing, to increase the chance of
      // interleaved requests.
      this->sync(param.grid_size, thread_idx, param.barriers[i]);

      if (block_idx < (IdxType)param.testing_blocks) {

        // Block is a testing block.
        //
        // Each block acts as a "sender" to a unique "partner" block. This is
        // done by permuting the block IDs with a function p(i) = i * k mod n,
        // where n is the number of blocks being tested, and k and n are
        // coprime.
        int partner_idx =
            (block_idx * PERMUTE_PRIME_GRID + i) % param.testing_blocks;

        // Run specified test, matching threads between the two paired blocks.
        int other_data_idx =
            partner_idx * param.block_size + permute_thread_idx;

        // Pre-stress pattern - stressing memory accesses before the test may
        // increase the rate of weak memory behaviors
        // Helps on AMD, doesn't seem to help on NVIDIA
        if (STRESS_BEFORE_TEST) {
          this->stress(block_idx, thread_idx, param, true);
        }

        test.run(data_idx, other_data_idx, i);
      } else {
        this->stress(block_idx, thread_idx, param);
      }
    }
  };

  RAJA_HOST_DEVICE void sync(int num_blocks, int thread_idx, IdxType& barrier)
  {
    if (thread_idx == 0) {
      IdxType result = RAJA::atomicAdd<NormalAtomic>(&barrier, IdxType{1});
      // Busy-wait until all blocks perform the above add.
      while (result != num_blocks) {
        result = RAJA::atomicAdd<NormalAtomic>(&barrier, IdxType{0});
      }
    }

#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_CODE__)
    __syncthreads();
#endif
  }

  RAJA_HOST_DEVICE void stress(IdxType block_idx,
                               IdxType thread_idx,
                               const TestData& param,
                               bool pre_stress = false)
  {
    int num_iters =
        (pre_stress ? param.pre_stress_iterations : param.stress_iterations);
    int volatile* stress_data = param.data_stress;
    if (thread_idx % WARP_SIZE == 0) {
      int warp_idx = thread_idx / WARP_SIZE;
      // int select_idx = block_idx * NUM_WARPS + warp_idx;
      int select_idx = block_idx;
      select_idx = (select_idx * PERMUTE_STRESS) % param.num_stress_index;
      int stress_line = param.stress_index[select_idx];
      for (int i = 0; i < num_iters; i++) {
        int data = stress_data[stress_line];
        stress_data[stress_line] = i + 1;
      }
    }
  }
};
