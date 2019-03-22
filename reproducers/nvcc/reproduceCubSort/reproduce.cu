//
// When compiling with relocatable device code with cuda 9 (.0, .1, .2)
// we encounter dlink "uses too much shared data" errors like this.
// @E@nvlink error   : Entry function '_ZN3cub30DeviceRadixSortDownsweepKernelINS_21DeviceRadixSortPolicyIiNS_8NullTypeEiE9Policy700ELb0ELb0EiS2_iEEvPKT2_PS5_PKT3_PS9_PT4_SD_iiNS_13GridEvenShareISD_EE' uses too much shared data (0xc080 bytes, 0xc000 max)
//

// function using 1 byte of shared memory
__device__ __forceinline__
void share_1B()
{
  __shared__ char s[1];
  s[0] = 0;
}

// These other_1 and other_2 global functions call a device function that uses
// shared memory.
// This shared memory usage appears to be summed into the shared memory usage
// of certain global functions
// This does not occur if only one of other_1 or other_2 is present.
__global__
void other_1()
{
  share_1B();
}

__global__
void other_2()
{
  share_1B();
}

// This device function uses the maximum amount of shared memory
__device__ __forceinline__
void share_maxB()
{
  __shared__ char s[0xc000];
  s[0] = 0;
}

// This global function calls a function that uses the maximum amount of
// shared memory allowed.
// however the device linker detects too much shared memory usage by this
// global function when compiled using relocatable device code
// and the above other_1 and other_2 global functions exist.
__global__
void reproduce()
{
  share_maxB();
}
