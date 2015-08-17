// This work was performed under the auspices of the U.S. Department of Energy by
// Lawrence Livermore National Laboratory under Contract DE-AC52-07NA27344.
//

//
// ALLOCATE/RELEASE FUNCTIONS 
//

#if defined(RAJA_USE_CUDA) // CUDA managed memory allocate/release

template <typename T>
inline T *Allocate(size_t size)
{
   T *retVal ;
   if (cudaMallocManaged((void **)&retVal, sizeof(T)*size, cudaMemAttachGlobal) != cudaSuccess) {
      std::cerr << "\n ERROR in CUDA Call, FILE: " << __FILE__ << " line "
                << __LINE__ << std::endl;
     exit(1);
   }
   return retVal ;
}

template <typename EXEC_POLICY_T, typename T>
inline T *AllocateTouch(LULESH_INDEXSET *is, size_t size)
{
   T *retVal ;
   if (cudaMallocManaged((void **)&retVal, sizeof(T)*size, cudaMemAttachGlobal) != cudaSuccess) {
      std::cerr << "\n ERROR in CUDA Call, FILE: " << __FILE__ << " line "
                << __LINE__ << std::endl;
      exit(1);
   }
   return retVal ;
}

template <typename T>
inline void Release(T **ptr)
{
   if (*ptr != NULL) {
      if (cudaFree(*ptr) != cudaSuccess) {
        std::cerr << "\n ERROR in CUDA Call, FILE: " << __FILE__ << " line "
                  << __LINE__ << std::endl;
        exit(1);
      }
      *ptr = NULL ;
   }
}


#else  // Standard CPU memory allocate/release

template <typename T>
inline T *Allocate(size_t size)
{
   T *retVal ;
   posix_memalign((void **)&retVal, RAJA::DATA_ALIGN, sizeof(T)*size);
   return retVal ;
}

template <typename EXEC_POLICY_T, typename T>
inline T *AllocateTouch(LULESH_INDEXSET *is, size_t size)
{
   T *retVal ;
   posix_memalign((void **)&retVal, RAJA::DATA_ALIGN, sizeof(T)*size);

   /* we should specialize by policy type here */
   RAJA::forall<EXEC_POLICY_T>( *is, [&] (int i) {
      retVal[i] = 0 ;
   } ) ;

   return retVal ;
}

inline void Release(Real_p ptr)
{
   if (ptr != NULL) {
      free(ptr) ;
      ptr = NULL ;
   }
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
