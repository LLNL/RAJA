#if defined(RAJA_USE_ATOMIC_TWO)

#if defined(RAJA_USE_CUDA_ATOMIC)
#error \
    "More than one atomic specialization has been included. This is very wrong"
#endif

#define RAJA_USE_CUDA_ATOMIC

/*!
 ******************************************************************************
 *
 * \brief Alternative atomic min and max update methods used to gather current
 *        min or max reduction value.
 *
 *        These appear to be more robust than the ones above, not sure why.
 *
 ******************************************************************************
 */
template <>
__device__ inline double min(double *address, double value)
{
  double temp = *(reinterpret_cast<double volatile *>(address));
  if (temp > value) {
    unsigned long long *address_as_ull =
        reinterpret_cast<unsigned long long *>(address);

    unsigned long long assumed;
    unsigned long long oldval = __double_as_longlong(temp);
    do {
      assumed = oldval;
      oldval = atomicCAS(address_as_ull,
                         assumed,
                         __double_as_longlong(
                             RAJA_MIN(__longlong_as_double(assumed), value)));
    } while (assumed != oldval);
    temp = __longlong_as_double(oldval);
  }
  return temp;
}
///
template <>
__device__ inline float min(float *address, float value)
{
  float temp = *(reinterpret_cast<float volatile *>(address));
  if (temp > value) {
    int *address_as_i = (int *)address;
    int assumed;
    int oldval = __float_as_int(temp);
    do {
      assumed = oldval;
      oldval =
          atomicCAS(address_as_i,
                    assumed,
                    __float_as_int(RAJA_MIN(__int_as_float(assumed), value)));
    } while (assumed != oldval);
    temp = __int_as_float(oldval);
  }
  return temp;
}
///
template <>
__device__ inline double max(double *address, double value)
{
  double temp = *(reinterpret_cast<double volatile *>(address));
  if (temp < value) {
    unsigned long long *address_as_ull =
        reinterpret_cast<unsigned long long *>(address);

    unsigned long long assumed;
    unsigned long long oldval = __double_as_longlong(temp);
    do {
      assumed = oldval;
      oldval = atomicCAS(address_as_ull,
                         assumed,
                         __double_as_longlong(
                             RAJA_MAX(__longlong_as_double(assumed), value)));
    } while (assumed != oldval);
    temp = __longlong_as_double(oldval);
  }
  return temp;
}
///
template <>
__device__ inline float max(float *address, float value)
{
  float temp = *(reinterpret_cast<float volatile *>(address));
  if (temp < value) {
    int *address_as_i = (int *)address;
    int assumed;
    int oldval = __float_as_int(temp);
    do {
      assumed = oldval;
      oldval =
          atomicCAS(address_as_i,
                    assumed,
                    __float_as_int(RAJA_MAX(__int_as_float(assumed), value)));
    } while (assumed != oldval);
    temp = __int_as_float(oldval);
  }
  return temp;
}

#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 350
// don't specialize for 64-bit min/max if they exist
#else
///
template <>
__device__ inline unsigned long long int min(unsigned long long int *address,
                                             unsigned long long int value)
{
  unsigned long long int temp =
      *(reinterpret_cast<unsigned long long int volatile *>(address));
  if (temp > value) {
    unsigned long long int assumed;
    unsigned long long int oldval = temp;
    do {
      assumed = oldval;
      oldval = atomicCAS(address, assumed, RAJA_MIN(assumed, value));
    } while (assumed != oldval);
    temp = oldval;
  }
  return temp;
}
///
template <>
__device__ inline unsigned long long int max(unsigned long long int *address,
                                             unsigned long long int value)
{
  unsigned long long int temp =
      *(reinterpret_cast<unsigned long long int volatile *>(address));
  if (temp < value) {
    unsigned long long int assumed;
    unsigned long long int oldval = temp;
    do {
      assumed = oldval;
      oldval = atomicCAS(address, assumed, RAJA_MAX(assumed, value));
    } while (assumed != oldval);
    temp = oldval;
  }
  return temp;
}
#endif

#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 600
// don't specialize for doubles if they exist
#else
/*!
 ******************************************************************************
 *
 * \brief Atomic add update methods used to accumulate to memory locations.
 *
 ******************************************************************************
 */
template <>
__device__ inline double add(double *address, double value)
{
  unsigned long long int *address_as_ull = (unsigned long long int *)address;
  unsigned long long int oldval = *address_as_ull, assumed;

  do {
    assumed = oldval;
    oldval =
        atomicCAS(address_as_ull,
                  assumed,
                  __double_as_longlong(__longlong_as_double(oldval) + value));
  } while (assumed != oldval);
  return __longlong_as_double(oldval);
}

#endif

#endif
