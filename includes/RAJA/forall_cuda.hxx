/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   Header file containing RAJA index set iteration template 
 *          methods for execution via CUDA kernel launch.
 *
 *          These methods should work on any platform.
 *
 * \author  Rich Hornung, Center for Applied Scientific Computing, LLNL
 * \author  Jeff Keasler, Applications, Simulations And Quality, LLNL
 *
 ******************************************************************************
 */

#ifndef RAJA_forall_cuda_HXX
#define RAJA_forall_cuda_HXX

#include "config.hxx"

#include "int_datatypes.hxx"

#include "execpolicy.hxx"

#include "fault_tolerance.hxx"


namespace RAJA {

//
//////////////////////////////////////////////////////////////////////
//
// Function templates that iterate over range index sets.
//
//////////////////////////////////////////////////////////////////////
//

/*!
 ******************************************************************************
 *
 * \brief  General CUDA kernal forall method template for index range.
 *
 ******************************************************************************
 */
template <typename LOOP_BODY>
__global__ void forall_kernel(LOOP_BODY loop_body, Index_type length)
{
  Index_type ii = blockDim.x * blockIdx.x + threadIdx.x;
  if (ii < length) {
    loop_body(ii);
  }
}


/*!
 ******************************************************************************
 *
 * \brief  Forall execution over index range via CUDA kernal launch.
 *
 ******************************************************************************
 */
template <typename LOOP_BODY>
RAJA_INLINE
void forall(cuda_exec,
            Index_type begin, Index_type end, 
            LOOP_BODY loop_body)
{

   RAJA_FT_BEGIN ;

   size_t blockSize = 256;
   size_t gridSize = (end - begin) / blockSize + 1;
   forall_kernel<<<gridSize, blockSize>>>(loop_body, end - begin);
#ifdef RAJA_SYNC
   if (cudaDeviceSynchronize() != cudaSuccess) {
      std::cerr << "\n ERROR in CUDA Call, FILE: " << __FILE__ << " line "
                << __LINE__ << std::endl;
      exit(1);
   }
#endif

   RAJA_FT_END ;
}


/*!
 ******************************************************************************
 *
 * \brief  General CUDA kernal forall_minloc method template for index range.
 *
 ******************************************************************************
 */
template <typename LOOP_BODY>
__global__ void forall_minloc_kernel(LOOP_BODY loop_body, 
                                     double *min, int *loc,
                                     Index_type length)
{
  Index_type ii = blockDim.x * blockIdx.x + threadIdx.x;
  if (ii < length) {
    loop_body(ii, min, loc);
  }
}

/*!
 ******************************************************************************
 *
 * \brief  Forall min-loc reduction over index range via CUDA kernal launch.
 *
 ******************************************************************************
 */
template <typename LOOP_BODY>
RAJA_INLINE
void forall_minloc(cuda_exec,
                   Index_type begin, Index_type end,
                   double *min, int *loc,
                   LOOP_BODY loop_body)
{

   RAJA_FT_BEGIN ;

   size_t blockSize = 256;
   size_t gridSize = (end - begin) / blockSize + 1;
   forall_minloc_kernel<<<gridSize, blockSize>>>(loop_body, 
                                                 min, loc,
                                                 end - begin);
#ifdef RAJA_SYNC
   if (cudaDeviceSynchronize() != cudaSuccess) {
      std::cerr << "\n ERROR in CUDA Call, FILE: " << __FILE__ << " line "
                << __LINE__ << std::endl;
      exit(1);
   }
#endif

   RAJA_FT_END ;
}


/*!
 ******************************************************************************
 *
 * \brief  General CUDA kernal forall method template for indirection array.
 *
 ******************************************************************************
 */
template <typename LOOP_BODY>
__global__ void forall_kernel(LOOP_BODY loop_body, 
                              const Index_type *idx, 
                              Index_type length)
{
  Index_type ii = blockDim.x * blockIdx.x + threadIdx.x;
  if (ii < length) {
    loop_body(idx[ii]);
  }
}

/*!
 ******************************************************************************
 *
 * \brief  Forall execution for indirection array via CUDA kernal launch.
 *
 ******************************************************************************
 */
template <typename LOOP_BODY>
RAJA_INLINE
void forall(cuda_exec,
            const Index_type* __restrict__ idx, const Index_type len,
            LOOP_BODY loop_body)
{
   RAJA_FT_BEGIN ;

   size_t blockSize = 256;
   size_t gridSize = len / blockSize + 1;
   forall_kernel<<<gridSize, blockSize>>>(loop_body, idx, len);
#ifdef RAJA_SYNC
   if (cudaDeviceSynchronize() != cudaSuccess) {
      std::cerr << "\n ERROR in CUDA Call, FILE: " << __FILE__ << " line "
                << __LINE__ << std::endl;
      exit(1);
   }
#endif

   RAJA_FT_END ;
}


/*!
 ******************************************************************************
 *
 * \brief  General CUDA kernal forall_minloc method template for indirection
 *         array.
 *
 ******************************************************************************
 */
template <typename LOOP_BODY>
__global__ void forall_minloc_kernel(LOOP_BODY loop_body,
                                     double *min, int *loc,
                                     const Index_type *idx, 
                                     Index_type length)
{
  Index_type ii = blockDim.x * blockIdx.x + threadIdx.x;
  if (ii < length) {
    loop_body(idx[ii], min, loc);
  }
}

/*!
 ******************************************************************************
 *
 * \brief  Forall min-loc reduction for indirection array via CUDA kernal launch.
 *
 ******************************************************************************
 */
template <typename LOOP_BODY>
RAJA_INLINE
void forall_minloc(cuda_exec,
                   const Index_type* __restrict__ idx, const Index_type len,
                   double *min, int *loc, 
                   LOOP_BODY loop_body)
{
   RAJA_FT_BEGIN ;

   size_t blockSize = 256;
   size_t gridSize = len / blockSize + 1;
   forall_minloc_kernel<<<gridSize, blockSize>>>(loop_body,
                                                 min, loc,
                                                 idx, len);
#ifdef RAJA_SYNC
   if (cudaDeviceSynchronize() != cudaSuccess) {
      std::cerr << "\n ERROR in CUDA Call, FILE: " << __FILE__ << " line "
                << __LINE__ << std::endl;
      exit(1);
   }
#endif

   RAJA_FT_END ;
}


}  // closing brace for RAJA namespace


#endif  // closing endif for header file include guard
