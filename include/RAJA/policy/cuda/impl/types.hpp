#ifndef RAJA_policy_cuda_types_HPP
#define RAJA_policy_cuda_types_HPP

namespace RAJA
{

namespace cuda
{
#if defined(RAJA_ENABLE_CLANG_CUDA)
using dim_t = uint3;
#else
using dim_t = dim3;
#endif

/*!
 * \brief Struct that contains two CUDA dim3's that represent the number of
 * thread block and the number of blocks.
 *
 * This is passed to the execution policies to setup the kernel launch.
 */
struct Dim {
  dim_t num_threads;
  dim_t num_blocks;

  RAJA_HOST_DEVICE void print(void) const
  {
    printf("<<< (%d,%d,%d), (%d,%d,%d) >>>\n",
           num_blocks.x,
           num_blocks.y,
           num_blocks.z,
           num_threads.x,
           num_threads.y,
           num_threads.z);
  }
};

template <typename POL, typename IDX>
struct IndexPair : public POL {
  template <typename IS>
  RAJA_INLINE constexpr IndexPair(Dim &dims, IS const &is) : POL(dims, is)
  {
  }

  using INDEX = IDX;
};

/** Provides a range from 0 to N_iter - 1
 *
 */
template <typename VIEWDIM, int threads_per_block>
struct ThreadBlock {
  int distance;

  VIEWDIM view;

  template <typename Iterable>
  ThreadBlock(Dim &dims, Iterable const &i)
      : distance(std::distance(std::begin(i), std::end(i)))
  {
    setDims(dims);
  }

  __device__ inline int operator()(void)
  {
    int idx = 0 + view(blockIdx) * threads_per_block + view(threadIdx);
    if (idx >= distance) {
      idx = INT_MIN;
    }
    return idx;
  }

  void inline setDims(Dim &dims)
  {
    int n = distance;
    if (n < threads_per_block) {
      view(dims.num_threads) = n;
      view(dims.num_blocks) = 1;
    } else {
      view(dims.num_threads) = threads_per_block;

      int blocks = n / threads_per_block;
      if (n % threads_per_block) {
        ++blocks;
      }
      view(dims.num_blocks) = blocks;
    }
  }
};

template <typename VIEWDIM>
struct Thread {
  int distance;

  VIEWDIM view;

  template <typename Iterable>
  Thread(Dim &dims, Iterable const &i)
      : distance(std::distance(std::begin(i), std::end(i)))
  {
    setDims(dims);
  }

  __device__ inline int operator()(void)
  {
    int idx = view(threadIdx);
    if (idx >= distance) {
      idx = INT_MIN;
    }
    return idx;
  }

  void inline setDims(Dim &dims) { view(dims.num_threads) = distance; }
};

template <typename VIEWDIM>
struct Block {
  int distance;

  VIEWDIM view;

  template <typename Iterable>
  Block(Dim &dims, Iterable const &i)
      : distance(std::distance(std::begin(i), std::end(i)))
  {
    setDims(dims);
  }

  __device__ inline int operator()(void)
  {
    int idx = view(blockIdx);
    if (idx >= distance) {
      idx = INT_MIN;
    }
    return idx;
  }

  void inline setDims(Dim &dims) { view(dims.num_blocks) = distance; }
};


///
/////////////////////////////////////////////////////////////////////
///
/// Generalizations of CUDA dim3 x, y and z used to describe
/// sizes and indices for threads and blocks.
///
/////////////////////////////////////////////////////////////////////
///
struct Dim3x {
  RAJA_HOST_DEVICE inline unsigned int &operator()(dim_t &dim)
  {
    return dim.x;
  }

  RAJA_HOST_DEVICE inline unsigned int operator()(dim_t const &dim)
  {
    return dim.x;
  }
};
///
struct Dim3y {
  RAJA_HOST_DEVICE inline unsigned int &operator()(dim_t &dim)
  {
    return dim.y;
  }

  RAJA_HOST_DEVICE inline unsigned int operator()(dim_t const &dim)
  {
    return dim.y;
  }
};
///
struct Dim3z {
  RAJA_HOST_DEVICE inline unsigned int &operator()(dim_t &dim)
  {
    return dim.z;
  }

  RAJA_HOST_DEVICE inline unsigned int operator()(dim_t const &dim)
  {
    return dim.z;
  }
};

} // end namespace cuda

} // end namespace RAJA

#endif
