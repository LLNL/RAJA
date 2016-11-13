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

/*!
 *******************************************************************************
 * \def RAJA_NOT_USED(x)
 *
 * \brief Macro for silencing compiler warnings in methods with unused
 *  arguments.
 *
 * \note The intent is to use this macro in the function signature. For example:
 *
 * \code
 *
 *  void my_function(int x, int RAJA_NOT_USED(y))
 *  {
 *    // my implementation
 *  }
 *
 * \endcode
 *******************************************************************************
 */
#define RAJA_NOT_USED(x)

/*!
 *******************************************************************************
 * \def RAJA_DEBUG_VAR(x)
 *
 * \brief Macro used to silence compiler warnings about variables
 *        that are defined but not used.
 *
 * \note The intent is to use this macro for variables that are only used
 *       for debugging purposes (e.g. in debug assertions). For example:
 *
 * \code
 *
 *  double myVar = ...
 *  RAJA_DEBUG_VAR(myVar); 
 *  
 *                       
 *  cassert(myVar > 0)  // variable used in an assertion
 *
 * \endcode
 *******************************************************************************
 */
#define ATK_DEBUG_VAR(_x)   static_cast<void>(_x)

/*!
 * \def RAJA_STRINGIFY_HELPER(x)
 *
 * Helper for RAJA_STRINGIFY_MACRO
 */
#define RAJA_STRINGIFY_HELPER(x) #x

/*!
 * \def RAJA_STRINGIFY_MACRO(x)
 * 
 * Used in static_assert macros to print values of defines
 */
#define RAJA_STRINGIFY_MACRO(x) RAJA_STRINGIFY_HELPER(x)

/*!
 * \def RAJA_DIVIDE_CEILING_INT(dividend, divisor)
 *
 * Macro to find ceiling (dividend / divisor) for integer types
 */
#define RAJA_DIVIDE_CEILING_INT(dividend, divisor) \
 ( ( (dividend) + (divisor) - 1 ) / (divisor) )

#endif /* RAJA_INTERNAL_DEFINES_HXX */
