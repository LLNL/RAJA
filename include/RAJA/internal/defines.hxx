#ifndef RAJA_INTERNAL_DEFINES_HXX
#define RAJA_INTERNAL_DEFINES_HXX

//
// Macros for decorating host/device functions for CUDA kernels.
// We need a better solution than this as it is a pain to manage
// this stuff in an application.
//
#if defined(RAJA_ENABLE_CUDA)

#define RAJA_HOST_DEVICE __host__ __device__
#define RAJA_DEVICE __device__
#if defined(_WIN32)  // windows is non-compliant, yay
#define RAJA_SUPPRESS_HD_WARN __pragma(nv_exec_check_disable)
#else
#define RAJA_SUPPRESS_HD_WARN _Pragma("nv_exec_check_disable")
#endif

#else

#define RAJA_HOST_DEVICE
#define RAJA_DEVICE
#define RAJA_SUPPRESS_HD_WARN
#endif

#endif /* RAJA_INTERNAL_DEFINES_HXX */
