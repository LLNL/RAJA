
#include "RAJA/RAJA.hpp"
#include "cub/device/device_radix_sort.cuh"

// This type is used to generate enough shared memory use to cause an error.
// In reality the extra shared memory may be composed of contributions from
// multiple sources or from a single source using a large type.
struct MinT
{
   int data[7];
   __host__ __device__
   bool operator<(MinT const& rhs) const
   {
      for (int i = 0; i < 3; ++i) {
         if (data[i] < rhs.data[i]) return true;
         else if (rhs.data[i] < data[i]) return false;
      }
      return false;
   }
};

// We use RAJA reducers which call functions that use shared memory
// in multiple kernels.
// Note that these would normally be spread across multiple different
// compilation units.
__global__
void other_1(RAJA::ReduceMin<RAJA::cuda_reduce, MinT> r)
{
  auto rc = r; // shared memory used in RAJA::ReduceMin destructor
}

__global__
void other_2(RAJA::ReduceMin<RAJA::cuda_reduce, MinT> r)
{
  auto rc = r; // shared memory used in RAJA::ReduceMin destructor
}

// We see the error in a global function called in
// ::cub::DeviceRadixSort::SortKeys.
void reproduce(int* din, int* dout, int n)
{
  void*    d_temp_storage = NULL;
  size_t   temp_storage_bytes = 0;
   ::cub::DeviceRadixSort::SortKeys(d_temp_storage, temp_storage_bytes,
         din, dout, n);
}
