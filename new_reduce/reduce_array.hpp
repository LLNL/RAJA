#ifndef NEW_REDUCE_ARRAY_HPP
#define NEW_REDUCE_ARRAY_HPP

#include "util/valloc.hpp"

#if defined(RAJA_ENABLE_CUDA)
#define DEVICE cuda
#elif defined(RAJA_ENABLE_HIP)
#define DEVICE hip
#endif

namespace detail
{

  //
  //
  // Basic ReducerArray
  //
  //
  template <typename Op, typename T>
  struct ReducerArray {
    using op = Op;
    using val_type = T*;

    RAJA_HOST_DEVICE ReducerArray() {
      //printf("Default Ctor %p\n", this);
    }

    ReducerArray(T *target_in, size_t size) :
        target(target_in),
        size(size),
        val_array(new T[size])
    {
      //printf("Ctor %p\n", this); 
      for (size_t i = 0; i < size; i++){
        val_array[i] = Op::identity();
      }
    }

    ReducerArray(const ReducerArray& rhs) :
        target(rhs.target), 
        size(rhs.size)//,
        //val_array(rhs.size ? new T[size] : nullptr)
    {
      //printf("Copy %p, %p\n", this, &rhs);
      //if(val_array) std::copy(rhs.val_array, rhs.val_array+size, val_array);
      if (rhs.val_array){
        val_array = new T[size];
        for (int i = 0; i < size; i++) {
          val_array[i] = rhs.val_array[i];
        }
      }
    }

    T *target = nullptr;
    size_t size;
    T *val_array;
    //RAJA_HOST_DEVICE ~ReducerArray() {printf("Dtor %p\n", this);}

//#if defined(RAJA_ENABLE_CUDA) || defined(RAJA_ENABLE_HIP)
//    // Device related attributes.
//    T * devicetarget = nullptr;
//    RAJA::detail::SoAPtr<T, RAJA::DEVICE::device_mempool_type> device_mem;
//    unsigned int * device_count = nullptr;
//#endif

    static constexpr size_t num_lambda_args = 1;
    auto get_lambda_arg_tup() { return camp::make_tuple(&val_array); }
  };

} // namespace detail

//#include "sequential/reduce.hpp"
#include "openmp/reduce_array.hpp"
//#include "omp-target/reduce.hpp"
//#include "cuda/reduce.hpp"
//#include "hip/reduce.hpp"

template <template <typename, typename, typename> class Op, typename T>
auto constexpr ReduceArray(T *target, size_t size)
{
  return detail::ReducerArray<Op<T, T, T>, T>(target, size);
}

//template <typename T>
//auto constexpr ReduceArrayLoc(ValLocMin<T> *target) 
//{
//  using R = ValLocMin<T>;
//  return detail::ReducerArray<RAJA::operators::minimum<R,R,R>, R>(target);
//}
//template <typename T>
//auto constexpr ReduceLoc(ValLocMax<T> *target)
//{
//  using R = ValLocMax<T>;
//  return detail::ReducerArray<RAJA::operators::maximum<R,R,R>, R>(target);
//}

#endif //  NEW_REDUCE_ARRAY_HPP
