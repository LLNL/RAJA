#ifndef NEW_REDUCE_CUDA_REDUCE_HPP
#define NEW_REDUCE_CUDA_REDUCE_HPP


#if defined(RAJA_ENABLE_CUDA)

#include <cuda.h>
#include "RAJA/policy/cuda/MemUtils_CUDA.hpp"
#include "../util/policy.hpp"

namespace detail {
using cuda_dim_t = dim3;

// ----------------------------------------------------------------------------
//                               BLOCK REDUCE
// ----------------------------------------------------------------------------
  //! reduce values in block into thread 0
  template <typename Combiner, typename T>
  RAJA_DEVICE RAJA_INLINE T block_reduce(T val, T identity)
  {
    int numThreads = blockDim.x * blockDim.y * blockDim.z;

    int threadId = threadIdx.x + blockDim.x * threadIdx.y +
                   (blockDim.x * blockDim.y) * threadIdx.z;

    int warpId = threadId % RAJA::policy::cuda::WARP_SIZE;
    int warpNum = threadId / RAJA::policy::cuda::WARP_SIZE;

    T temp = val;

    if (numThreads % RAJA::policy::cuda::WARP_SIZE == 0) {

      // reduce each warp
      for (int i = 1; i < RAJA::policy::cuda::WARP_SIZE; i *= 2) {
        T rhs = RAJA::cuda::impl::shfl_xor_sync(temp, i);
        temp = Combiner{}(temp, rhs);
      }

    } else {

      // reduce each warp
      for (int i = 1; i < RAJA::policy::cuda::WARP_SIZE; i *= 2) {
        int srcLane = threadId ^ i;
        T rhs = RAJA::cuda::impl::shfl_sync(temp, srcLane);
        // only add from threads that exist (don't double count own value)
        if (srcLane < numThreads) {
          temp = Combiner{}(temp, rhs);
        }
      }
    }

    static_assert(RAJA::policy::cuda::MAX_WARPS <= RAJA::policy::cuda::WARP_SIZE,
                 "Max Warps must be less than or equal to Warp Size for this algorithm to work");

    // reduce per warp values
    if (numThreads > RAJA::policy::cuda::WARP_SIZE) {

      // Need to separate declaration and initialization for clang-cuda
      __shared__ unsigned char tmpsd[sizeof(RAJA::detail::SoAArray<T, RAJA::policy::cuda::MAX_WARPS>)];

      // Partial placement new: Should call new(tmpsd) here but recasting memory
      // to avoid calling constructor/destructor in shared memory.
      RAJA::detail::SoAArray<T, RAJA::policy::cuda::MAX_WARPS> * sd = reinterpret_cast<RAJA::detail::SoAArray<T, RAJA::policy::cuda::MAX_WARPS> *>(tmpsd);

      // write per warp values to shared memory
      if (warpId == 0) {
        sd->set(warpNum, temp);
      }

      __syncthreads();

      if (warpNum == 0) {

        // read per warp values
        if (warpId * RAJA::policy::cuda::WARP_SIZE < numThreads) {
          temp = sd->get(warpId);
        } else {
          temp = identity;
        }

        for (int i = 1; i < RAJA::policy::cuda::MAX_WARPS; i *= 2) {
          T rhs = RAJA::cuda::impl::shfl_xor_sync(temp, i);
          temp = Combiner{}(temp, rhs);
        }
      }

      __syncthreads();
    }

    return temp;
  }

// ----------------------------------------------------------------------------
//                               GRID REDUCE
// ----------------------------------------------------------------------------
  template <typename OP, typename T>
  RAJA_DEVICE RAJA_INLINE bool grid_reduce(Reducer<OP, T>& red) {

    int numBlocks = gridDim.x * gridDim.y * gridDim.z;
    int numThreads = blockDim.x * blockDim.y * blockDim.z;
    unsigned int wrap_around = numBlocks - 1;

    int blockId = blockIdx.x + gridDim.x * blockIdx.y +
                  (gridDim.x * gridDim.y) * blockIdx.z;

    int threadId = threadIdx.x + blockDim.x * threadIdx.y +
                   (blockDim.x * blockDim.y) * threadIdx.z;

    T temp = block_reduce<OP>(red.val, OP::identity());

    // one thread per block writes to device_mem
    bool lastBlock = false;
    if (threadId == 0) {
      red.device_mem.set(blockId, temp);
      // ensure write visible to all threadblocks
      __threadfence();
      // increment counter, (wraps back to zero if old count == wrap_around)
      unsigned int old_count = ::atomicInc(red.device_count, wrap_around);
      lastBlock = (old_count == wrap_around);
    }

    // returns non-zero value if any thread passes in a non-zero value
    lastBlock = __syncthreads_or(lastBlock);

    // last block accumulates values from device_mem
    if (lastBlock) {
      temp = OP::identity();

      for (int i = threadId; i < numBlocks; i += numThreads) {
        temp = OP{}(temp, red.device_mem.get(i));
      }

      temp = block_reduce<OP>(temp, OP::identity());

      // one thread returns value
      if (threadId == 0) {
        *(red.devicetarget) = temp;
      }
    }

    return lastBlock && threadId == 0;
  }

// ----------------------------------------------------------------------------
//                                    INIT
// ----------------------------------------------------------------------------
  template<typename EXEC_POL,
           typename OP,
           typename T>
  camp::concepts::enable_if< is_cuda_policy< EXEC_POL > >
  init(Reducer<OP, T>& red, const RAJA::cuda::detail::cudaInfo & cs)
  {
    cudaMallocManaged( (void**)(&(red.devicetarget)), sizeof(T));//, cudaHostAllocPortable );
    red.device_mem.allocate(cs.gridDim.x * cs.gridDim.y * cs.gridDim.z);
    red.device_count = RAJA::cuda::device_zeroed_mempool_type::getInstance().template malloc<unsigned int>(1);
  }

// ----------------------------------------------------------------------------
//                                   COMBINE
// ----------------------------------------------------------------------------
  template<typename EXEC_POL, typename OP, typename T>
  RAJA_HOST_DEVICE
  camp::concepts::enable_if< is_cuda_policy< EXEC_POL > >
  combine(Reducer<OP, T>& red)
  {
    bool blah = grid_reduce(red);
  }
  
// ----------------------------------------------------------------------------
//                                   RESOLVE
// ----------------------------------------------------------------------------
  template<typename EXEC_POL, typename OP, typename T>
  camp::concepts::enable_if< is_cuda_policy< EXEC_POL > >
  resolve(Reducer<OP, T>& red)
  {
    cudaDeviceSynchronize();
    cudaMemcpy(red.target, red.devicetarget, sizeof(T), cudaMemcpyDeviceToHost);
  }

} //  namespace detail
#endif

#endif //  NEW_REDUCE_CUDA_REDUCE_HPP
