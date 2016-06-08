// This work was performed under the auspices of the U.S. Department of Energy by
// Lawrence Livermore National Laboratory under Contract DE-AC52-07NA27344.
//

//
// ALLOCATE/RELEASE FUNCTIONS 
//

#if defined(RAJA_ENABLE_CUDA) // CUDA managed memory allocate/release

#include <cuda.h>
#include <cuda_runtime.h>

template <typename T>
inline T *Allocate(size_t size)
{
   T *retVal ;
   cudaErrchk( cudaMallocManaged((void **)&retVal, sizeof(T)*size, cudaMemAttachGlobal) ) ;
   return retVal ;
}

template <typename EXEC_POLICY_T, typename T>
inline T *AllocateTouch(RAJA::IndexSet *is, size_t size)
{
   T *retVal ;
   cudaErrchk( cudaMallocManaged((void **)&retVal, sizeof(T)*size, cudaMemAttachGlobal) ) ;
   cudaMemset(retVal,0,sizeof(T)*size);
   return retVal ;
}

template <typename T>
inline void Release(T **ptr)
{
   if (*ptr != NULL) {
      cudaErrchk( cudaFree(*ptr) ) ;
      *ptr = NULL ;
   }
}

template <typename T>
inline void Release(T * __restrict__ *ptr)
{
   if (*ptr != NULL) {
      cudaErrchk( cudaFree(*ptr) ) ;
      *ptr = NULL ;
   }
}


#else  // Standard CPU memory allocate/release

#include <cstdlib>
#include <cstring>

template <typename T>
inline T *Allocate(size_t size)
{
   T *retVal ;
   posix_memalign((void **)&retVal, RAJA::DATA_ALIGN, sizeof(T)*size);
// memset(retVal,0,sizeof(T)*size);
   return retVal ;
}

template <typename EXEC_POLICY_T, typename T>
inline T *AllocateTouch(RAJA::IndexSet *is, size_t size)
{
   T *retVal ;
   posix_memalign((void **)&retVal, RAJA::DATA_ALIGN, sizeof(T)*size);

   /* we should specialize by policy type here */
   RAJA::forall<EXEC_POLICY_T>( *is, [=] RAJA_DEVICE (int i) {
      retVal[i] = 0 ;
   } ) ;

   return retVal ;
}

template <typename T>
inline void Release(T **ptr)
{
   if (*ptr != NULL) {
      free(*ptr) ;
      *ptr = NULL ;
   }
}

template <typename T>
inline void Release(T * __restrict__ *ptr)
{
   if (*ptr != NULL) {
      free(*ptr) ;
      *ptr = NULL ;
   }
}

#endif 


/**********************************/
/* Memory Pool                    */
/**********************************/

namespace RAJA {

template <typename VARTYPE >
struct MemoryPool {
public:
   MemoryPool()
   {
      for (int i=0; i<32; ++i) {
         lenType[i] = 0 ;
         ptr[i] = 0 ;
      }
   }

   VARTYPE *allocate(int len) {
      VARTYPE *retVal ;
      int i ;
      for (i=0; i<32; ++i) {
         if (lenType[i] == len) {
            lenType[i] = -lenType[i] ;
            retVal = ptr[i] ;
#if 0
            /* migrate smallest lengths to be first in list */
            /* since longer lengths can amortize lookup cost */
            if (i > 0) {
               if (len < abs(lenType[i-1])) {
                  lenType[i] = lenType[i-1] ;
                  ptr[i] = ptr[i-1] ;
                  lenType[i-1] = -len ;
                  ptr[i-1] = retVal ;
               }
            }
#endif
            break ;
         }
         else if (lenType[i] == 0) {
            lenType[i] = -len ;
            ptr[i] = Allocate<VARTYPE>(len) ;
            retVal = ptr[i] ;
            break ;
         }
      }
      if (i == 32) {
         retVal = 0 ;  /* past max available pointers */
      }
      return retVal ;
   }

   bool release(VARTYPE **oldPtr) {
      int i ;
      bool success = true ;
      for (i=0; i<32; ++i) {
         if (ptr[i] == *oldPtr) {
            lenType[i] = -lenType[i] ;
            *oldPtr = 0 ;
            break ;
         }
      }
      if (i == 32) {
         success = false ; /* error -- not found */
      }
      return success ;
   }

   bool release(VARTYPE * __restrict__ *oldPtr) {
      int i ;
      bool success = true ;
      for (i=0; i<32; ++i) {
         if (ptr[i] == *oldPtr) {
            lenType[i] = -lenType[i] ;
            *oldPtr = 0 ;
            break ;
         }
      }
      if (i == 32) {
         success = false ; /* error -- not found */
      }
      return success ; 
   }

   VARTYPE *ptr[32] ; 
   int lenType[32] ;
} ;

}

