//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-24, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

// RAJA/RAJA.hpp cannot be included here because the include logic will
// default all atomic implementations to a desul backend.
#include "RAJA/policy/loop/policy.hpp"
#include "RAJA/policy/openmp/atomic.hpp"
#include "RAJA/RAJA.hpp"
#include "RAJA/policy/openmp/policy.hpp"

#include <type_traits>
#include <iostream>
#include <sstream>

/// This helper template is used to deduce if device memory allocations are necessary
/// inside the body of the benchmark, using the type of the execution policy.
template<typename>
struct IsGPU : public std::false_type {};

/// Conditional compilation for CUDA benchmarks.
#if defined RAJA_ENABLE_CUDA
#include "RAJA/policy/cuda.hpp"
#include "RAJA/policy/cuda/atomic.hpp"

template<int M>
struct IsGPU<RAJA::cuda_exec<M>> : public std::true_type {};

template<int BLOCK_SZ>
struct ExecPolicyGPU {
    using policy = RAJA::cuda_exec<BLOCK_SZ>;
    static std::string PolicyName() {
        std::stringstream ss;
        ss << "CUDA execution with block size " << BLOCK_SZ;
        return ss.str();
    }
};

struct GPUAtomic {
    using policy = RAJA::policy::cuda::cuda_atomic;
};

template<typename AtomicType>
void AllocateAtomicDevice(AtomicType** atomic, int array_length) {
    cudaErrchk(cudaMalloc((void **)atomic, array_length * sizeof(AtomicType)));
    cudaMemset(*atomic, 0, array_length * sizeof(AtomicType));
}

template<typename AtomicType>
void DeallocateDeviceAtomic(AtomicType* atomic) {
    cudaErrchk(cudaFree((void *)atomic));
}

#elif defined RAJA_ENABLE_HIP
#include "RAJA/policy/hip.hpp"
#include "RAJA/policy/hip/atomic.hpp"

template<int M>
struct IsGPU<RAJA::hip_exec<M>> : public std::true_type {};

template<int BLOCK_SZ>
struct ExecPolicyGPU {
    using policy = RAJA::hip_exec<BLOCK_SZ>;
    static std::string PolicyName() {
        std::stringstream ss;
        ss << "HIP execution with block size " << BLOCK_SZ;
        return ss.str();
    }
};

struct GPUAtomic {
    using policy = RAJA::policy::hip::hip_atomic;
};

template<typename AtomicType>
void AllocateAtomicDevice(AtomicType** atomic, int array_length) {
    hipMalloc((void **)atomic, len_array * sizeof(AtomicType));
    hipMemset(*atomic, 0, len_array * sizeof(AtomicType));
}

template<typename AtomicType>
void DeallocateDeviceAtomic(AtomicType* atomic) {
    hipFree((void *)atomic);
}

#endif

#include "desul/atomics.hpp"
#include "RAJA/util/Timer.hpp"

#define N 1000000000
#define INDENT "  "
using raja_default_desul_order = desul::MemoryOrderRelaxed;
using raja_default_desul_scope = desul::MemoryScopeDevice;

// Desul atomics have a different signature than RAJA's built in ops. The following code provides some
// helper function templates so that they can be called using the same signature in timing code.

// Struct holding Desul atomic signature typedef
template<typename AtomicType>
struct DesulAtomicSignature {
    using type = AtomicType(*)(AtomicType*, const AtomicType, raja_default_desul_order, raja_default_desul_scope);
};

// Struct holding RAJA atomic signature typedef
template<typename AtomicType>
struct RajaAtomicSignature {
    using type = AtomicType(*)(AtomicType*, const AtomicType);
};

/// RAJA::atomicAdd is overloaded and has an ambiguous type so it can't be passed as a template parameter.
/// The following wrappers disambiguate the call and provide a signature comaptible with the DESUL
/// wrapper.
template<typename Policy>
struct AtomicAdd {
    template<typename AtomicType>
    RAJA_HOST_DEVICE AtomicType operator()(AtomicType* acc, const AtomicType val) const {
        return RAJA::atomicAdd(Policy {}, acc, val);
    }
};

template<typename Policy>
struct AtomicMax {
    template<typename AtomicType>
    RAJA_HOST_DEVICE AtomicType operator()(AtomicType* acc, const AtomicType val) const {
        return RAJA::atomicMax(Policy {}, acc, val);
    }
};

/// ExecPolicy wrapper for OpenMP
struct ExecPolicyOMP {
    using policy = RAJA::omp_for_exec;
    static std::string PolicyName() {
        std::stringstream ss;
        ss << "OpenMP execution";
        return ss.str();
    }
};

/// Functor wrapping the desul implementation.  Wrapping the desul call ensure an identical signature with
/// RAJA's implementations.  Wrapping the call in an functor allows simple type deduction for printing
/// from within the benchmark.
template<typename T, typename DesulAtomicSignature<T>::type atomic_impl>
struct atomicWrapperDesul {
    /// Call operator overload template that allows invoking DESUL atomic with a (int*)(T*, T) signature
    RAJA_DEVICE T operator()(T * acc, T value) const {
        return atomic_impl(acc, value, raja_default_desul_order{},
                        raja_default_desul_scope{});
    }
};

template<typename T>
class IsDesul : public std::false_type {};

template<typename T, typename DesulAtomicSignature<T>::type atomic_impl>
class IsDesul<atomicWrapperDesul<T, atomic_impl>> : public std::true_type {};

template<typename AtomicImplType>
std::string GetImplName (const AtomicImplType& impl) {
    if (IsDesul<AtomicImplType>::value) {
        return "Desul atomic";
    } else {
        return "RAJA atomic";
    }
}

/// Helper to deduce if atomic implementation is using desul under the hood
//bool isDesul(typename RajaAtomicSignature<AtomicType>::signature atomic_impl) {
//    bool is_desul =
//        atomic_impl == atomicWrapperDesul<AtomicType, typename GPUAtomic::policy, desul::atomic_fetch_add> ||
//        atomic_impl == atomicWrapperDesul<AtomicType, typename GPUAtomic::policy, desul::atomic_fetch_sub> ||
//        atomic_impl == atomicWrapperDesul<AtomicType, typename GPUAtomic::policy, desul::atomic_fetch_>
//
//}

template <class ExecPolicy, typename AtomicType, bool test_array, typename AtomicImplType>
void TimeAtomicOp( AtomicImplType atomic_impl, uint64_t num_iterations = 2, int array_size = 100) {
    RAJA::Timer timer;

    for (int i = 0; i < num_iterations; ++i) {
        AtomicType* device_value = nullptr;
        int len_array = test_array ? array_size : 1;
        if (IsGPU<typename ExecPolicy::policy>::value) {
            AllocateAtomicDevice(&device_value, len_array);
        } else {
            device_value = new AtomicType [len_array];
        }
        timer.start();
        RAJA::forall<typename ExecPolicy::policy>(RAJA::RangeSegment(0, N),
        [=] RAJA_HOST_DEVICE(int tid)  {
            if (test_array) {
                atomic_impl(&(device_value[tid % array_size]), AtomicType(1));
            } else {
                atomic_impl(device_value, AtomicType(1));
            }
        });

        timer.stop();
        if (IsGPU<typename ExecPolicy::policy>::value) {
            DeallocateDeviceAtomic(device_value);
        } else {
            delete device_value;
        }

    }

    double t = timer.elapsed();
    std::cout << INDENT << INDENT << t << "s" << INDENT;
    std::cout << GetImplName(atomic_impl) << ", ";
    std::cout << "Number of atomics under contention " << array_size << ", ";
    std::cout << num_iterations * N << " many atomic operations" << ", ";
    std::cout << ExecPolicy::PolicyName();
    std::cout << std::endl;
}

int main () {
    // GPU benchmarks
    std::cout << "Executing CUDA benchmarks" << std::endl;
    std::cout << INDENT << "Executing atomic add integer benchmarks" << std::endl;
    TimeAtomicOp<ExecPolicyGPU<64>, int,  true>(AtomicAdd<typename GPUAtomic::policy>{}, 4);
    TimeAtomicOp<ExecPolicyGPU<64>, int, true>(atomicWrapperDesul<int, desul::atomic_fetch_add>{}, 4);
    TimeAtomicOp<ExecPolicyGPU<128>, int, true>(AtomicAdd<typename GPUAtomic::policy>{}, 4);
    TimeAtomicOp<ExecPolicyGPU<128>, int, true>(atomicWrapperDesul<int, desul::atomic_fetch_add>{}, 4);
    TimeAtomicOp<ExecPolicyGPU<256>, int, true>(AtomicAdd<typename GPUAtomic::policy> {}, 4);
    TimeAtomicOp<ExecPolicyGPU<256>, int, true>(atomicWrapperDesul<int, desul::atomic_fetch_add>{}, 4);

    TimeAtomicOp<ExecPolicyGPU<128>, int, true>(AtomicAdd<typename GPUAtomic::policy>{}, 4, 10);
    TimeAtomicOp<ExecPolicyGPU<128>, int, true>(atomicWrapperDesul<int, desul::atomic_fetch_add> {}, 4, 10);
    TimeAtomicOp<ExecPolicyGPU<256>, int, true>(AtomicAdd<typename GPUAtomic::policy> {}, 4, 10);
    TimeAtomicOp<ExecPolicyGPU<256>, int, true>(atomicWrapperDesul<int, desul::atomic_fetch_add> {}, 4, 10);

    std::cout << INDENT << "Executing atomic add double benchmarks" << std::endl;

    TimeAtomicOp<ExecPolicyGPU<128>, double, false>(AtomicAdd<typename GPUAtomic::policy> {});
    TimeAtomicOp<ExecPolicyGPU<128>, double, false>(atomicWrapperDesul<double, desul::atomic_fetch_add> {});
    TimeAtomicOp<ExecPolicyGPU<256>, double, false>(AtomicAdd<typename GPUAtomic::policy> {});
    TimeAtomicOp<ExecPolicyGPU<256>, double, false>(atomicWrapperDesul<double, desul::atomic_fetch_add> {});

    std::cout << INDENT << "Executing atomic max benchmarks" << std::endl;

    TimeAtomicOp<ExecPolicyGPU<128>, int, false>(AtomicMax<GPUAtomic::policy>{});
    TimeAtomicOp<ExecPolicyGPU<128>, int, false>(atomicWrapperDesul<int, desul::atomic_fetch_max> {});
    TimeAtomicOp<ExecPolicyGPU<256>, int, false>(AtomicMax<GPUAtomic::policy> {});
    TimeAtomicOp<ExecPolicyGPU<256>, int, false>(atomicWrapperDesul<int, desul::atomic_fetch_max> {});

    // OpenMP benchmarks
    std::cout << "Executing OpenMP benchmarks" << std::endl;
    std::cout << INDENT << "Executing atomic add benchmarks" << std::endl;
    TimeAtomicOp<ExecPolicyOMP, int, false>(AtomicAdd<RAJA::policy::omp::omp_atomic> {});
    TimeAtomicOp<ExecPolicyOMP, int, false>(atomicWrapperDesul<int, desul::atomic_fetch_add> {});

    return 0;
}
