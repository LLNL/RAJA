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
#if defined RAJA_ENABLE_CUDA
#include "RAJA/policy/cuda.hpp"
#include "RAJA/policy/cuda/atomic.hpp"
#elif defined RAJA_ENABLE_HIP
#include "RAJA/policy/hip.hpp"
#include "RAJA/policy/hip/atomic.hpp"
#endif

#include "desul/atomics.hpp"
#include "RAJA/util/Timer.hpp"

#include <type_traits>
#include <iostream>

#define N 1000000000
using raja_default_desul_order = desul::MemoryOrderRelaxed;
using raja_default_desul_scope = desul::MemoryScopeDevice;

template<int BLOCK_SZ>
struct ExecPolicyGPU {
#if defined RAJA_ENABLE_CUDA
    using policy = RAJA::cuda_exec<BLOCK_SZ>;
#elif defined RAJA_ENABLE_HIP
    using policy = RAJA::hip_exec<BLOCK_SZ>;
#endif
};

struct GPUAtomic {
    #if defined RAJA_ENABLE_CUDA
        using policy = RAJA::policy::cuda::cuda_atomic;
    #elif defined RAJA_ENABLE_HIP
        using policy = RAJA::policy::hip::hip_atomic;
    #endif
};

// Desul atomics have a different signature than RAJA's built in ops. The following code provides some
// helper function templates so that they can be called using the same signature in timing code.

// Struct holding Desul atomic signature typedef
template<typename AtomicType>
struct DesulAtomicSignature {
    using signature = AtomicType(*)(AtomicType*, const AtomicType, raja_default_desul_order, raja_default_desul_scope);
};

// Struct holding RAJA atomic signature typedef
template<typename AtomicType>
struct RajaAtomicSignature {
    using signature = AtomicType(*)(AtomicType*, const AtomicType);
};

/// RAJA::atomicAdd is overloaded and has an ambiguous type so it can't be passed as a template parameter.
/// The following wrappers disambiguate the call and provide a signature comaptible with the DESUL
/// wrapper.
template<typename AtomicType>
RAJA_HOST_DEVICE AtomicType GPUAtomicAdd(AtomicType* acc, const AtomicType val) {
    return RAJA::atomicAdd(GPUAtomic::policy {}, acc, val);
}

template<typename AtomicType>
RAJA_HOST_DEVICE AtomicType GPUAtomicMax(AtomicType* acc, const AtomicType val) {
    return RAJA::atomicMax(GPUAtomic::policy {}, acc, val);
}

template<typename AtomicType>
RAJA_HOST_DEVICE AtomicType OpenMPAtomicAdd(AtomicType* acc, const AtomicType val) {
    return RAJA::atomicAdd(RAJA::policy::omp::omp_atomic{}, acc, val);
}

/// Function template that allows invoking DESUL atomic with a (int*)(T*, T) signature
template<typename T, typename Policy, typename DesulAtomicSignature<T>::signature AtomicImpl>
RAJA_HOST_DEVICE T atomicWrapperDesul(T * acc, T value) {

    return AtomicImpl(acc, value, raja_default_desul_order{},
                    raja_default_desul_scope{});
}

template<typename>
struct IsGPU : public std::false_type {};

/// These helper templates are used to deduce if device memory allocations are necessary
/// inside the body of the benchmark, using the type of the execution policy.
#if defined RAJA_ENABLE_CUDA
template<int M>
struct IsGPU<RAJA::cuda_exec<M>> : public std::true_type {};
#elif defined RAJA_ENABLE_HIP
template<int M>
struct IsGPU<RAJA::hip_exec<M>> : public std::true_type {};
#endif

template <class ExecPolicy, typename AtomicType, typename RajaAtomicSignature<AtomicType>::signature AtomicImpl, bool test_array = false>
void TimeAtomicOp(const std::string& test_name, int num_iterations = 2, int array_size = 100) {
    std::cout << "EXECUTING " << test_name << "; ";
    RAJA::Timer timer;

    for (int i = 0; i < num_iterations; ++i) {
        AtomicType* device_value = nullptr;
        int len_array = test_array ? array_size : 1;
        if (IsGPU<ExecPolicy>::value) {
#if defined(RAJA_ENABLE_CUDA)
            cudaErrchk(cudaMalloc((void **)&device_value, len_array * sizeof(AtomicType)));
            cudaMemset(device_value, 0, len_array * sizeof(AtomicType));
#elif defined(RAJA_ENABLE_HIP)
            hipMalloc((void **)&device_value, len_array * sizeof(AtomicType));
            hipMemset(device_value, 0, len_array * sizeof(AtomicType));
#endif
        }
        timer.start();
        RAJA::forall<ExecPolicy>(RAJA::RangeSegment(0, N),
        [=] RAJA_HOST_DEVICE(int tid)  {
            if (test_array) {
                AtomicImpl(&(device_value[tid % array_size]), 1);
            } else {
                AtomicImpl(device_value, 1);
            }
        });

        timer.stop();
        if (IsGPU<ExecPolicy>::value) {
#if defined(RAJA_ENABLE_CUDA)
            cudaErrchk(cudaFree((void *)device_value));
#elif defined(RAJA_ENABLE_HIP)
            hipFree((void *)device_value);
#endif
        }

    }

    double t = timer.elapsed();
    std::cout << "ELAPSED TIME = " << t << std::endl;
}

int main () {
    // GPU benchmarks
    TimeAtomicOp<ExecPolicyGPU<32>::policy, int, GPUAtomicAdd<int>, true>("Benchmark array contention.  CUDA Block size 32, RAJA builtin atomic", 4);
    TimeAtomicOp<ExecPolicyGPU<32>::policy, int, atomicWrapperDesul<int, typename GPUAtomic::policy, desul::atomic_fetch_add>, true>("Benchmark array contention.  CUDA Block size 32, DESUL atomic", 4);
    TimeAtomicOp<ExecPolicyGPU<64>::policy, int, GPUAtomicAdd<int>, true>("Benchmark array contention. CUDA Block size 64, RAJA builtin atomic", 4);
    TimeAtomicOp<ExecPolicyGPU<64>::policy, int, atomicWrapperDesul<int, typename GPUAtomic::policy, desul::atomic_fetch_add>, true>("Benchmark array contention. CUDA Block size 64, DESUL atomic", 4);
    TimeAtomicOp<ExecPolicyGPU<128>::policy, int, GPUAtomicAdd<int>, true>("Benchmark array contention.  CUDA Block size 128, RAJA builtin atomic", 4);
    TimeAtomicOp<ExecPolicyGPU<128>::policy, int, atomicWrapperDesul<int, typename GPUAtomic::policy, desul::atomic_fetch_add>, true>("Benchmark array contention.  CUDA Block size 128, DESUL atomic", 4);
    TimeAtomicOp<ExecPolicyGPU<256>::policy, int, GPUAtomicAdd<int>, true>("Benchmark array contention. CUDA Block size 256, RAJA builtin atomic", 4);
    TimeAtomicOp<ExecPolicyGPU<256>::policy, int, atomicWrapperDesul<int, typename GPUAtomic::policy, desul::atomic_fetch_add>, true>("Benchmark array contention. CUDA Block size 256, DESUL atomic", 4);

    TimeAtomicOp<ExecPolicyGPU<128>::policy, int, GPUAtomicAdd<int>, true>("Benchmark array contention.  CUDA Block size 128, RAJA builtin atomic", 2, 10);
    TimeAtomicOp<ExecPolicyGPU<128>::policy, int, atomicWrapperDesul<int, typename GPUAtomic::policy, desul::atomic_fetch_add>, true>("Benchmark array contention.  CUDA Block size 128, DESUL atomic", 2, 10);
    TimeAtomicOp<ExecPolicyGPU<256>::policy, int, GPUAtomicAdd<int>, true>("Benchmark array contention. CUDA Block size 256, RAJA builtin atomic", 2, 10);
    TimeAtomicOp<ExecPolicyGPU<256>::policy, int, atomicWrapperDesul<int, typename GPUAtomic::policy, desul::atomic_fetch_add>, true>("Benchmark array contention. CUDA Block size 256, DESUL atomic", 2, 10);

    TimeAtomicOp<ExecPolicyGPU<128>::policy, double, GPUAtomicAdd<double>>("CUDA Block size 128, RAJA builtin atomic");
    TimeAtomicOp<ExecPolicyGPU<128>::policy, double, atomicWrapperDesul<double, typename GPUAtomic::policy, desul::atomic_fetch_add>>("CUDA Block size 128, DESUL atomic");
    TimeAtomicOp<ExecPolicyGPU<256>::policy, double, GPUAtomicAdd<double>>("CUDA Block size 256, RAJA builtin atomic");
    TimeAtomicOp<ExecPolicyGPU<256>::policy, double, atomicWrapperDesul<double, typename GPUAtomic::policy, desul::atomic_fetch_add>>("CUDA Block size 256, DESUL atomic");

    TimeAtomicOp<ExecPolicyGPU<128>::policy, int, GPUAtomicMax<int>>("CUDA Block size 128, RAJA builtin atomic max");
    TimeAtomicOp<ExecPolicyGPU<128>::policy, int, atomicWrapperDesul<int, typename GPUAtomic::policy, desul::atomic_fetch_max>>("CUDA Block size 128, DESUL atomic max");
    TimeAtomicOp<ExecPolicyGPU<256>::policy, int, GPUAtomicMax<int>>("CUDA Block size 256, RAJA builtin atomic max");
    TimeAtomicOp<ExecPolicyGPU<256>::policy, int, atomicWrapperDesul<int, typename GPUAtomic::policy, desul::atomic_fetch_max>>("CUDA Block size 256, DESUL atomic max");
    // OpenMP benchmarks
    //TimeAtomicOp<RAJA::omp_for_exec, int, OpenMPAtomicAdd<int>>("OpenMP, int, RAJA builtin atomic");
    //TimeAtomicOp<RAJA::omp_for_exec, int, atomicWrapperDesul<int, RAJA::policy::omp::omp_atomic, desul::atomic_fetch_add>>("OpenMP, desul atomic");

    return 0;
}
