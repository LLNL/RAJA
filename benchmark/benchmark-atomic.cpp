//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-24, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// This file is intended to provide an interface for comparing the performance
// of RAJA's atomic implementations with Desul's atomic implementations.  In order
// to accomplish this without requiring two separate build system configurations
// this file directly includes "desul/atomics.hpp" and directly calls desul namespace
// atomics.  This is different from how a typical RAJA user would call a desul atomic.

#include "RAJA/RAJA.hpp"
#include "RAJA/util/for_each.hpp"
#include "RAJA/util/Timer.hpp"
#if defined (RAJA_ENABLE_OPENMP)
#include "RAJA/policy/openmp/atomic.hpp"
#include "RAJA/policy/openmp/policy.hpp"
#endif
#include "desul/atomics.hpp"

#include <type_traits>
#include <iostream>
#include <sstream>
#include <string>


/// Conditional compilation for CUDA benchmarks.
#if defined (RAJA_ENABLE_CUDA)
#include "RAJA/policy/cuda.hpp"
#include "RAJA/policy/cuda/atomic.hpp"

template<int BLOCK_SZ>
struct ExecPolicyGPU {
    using policy = RAJA::cuda_exec<BLOCK_SZ, false>;
    static std::string PolicyName() {
        std::stringstream ss;
        ss << "CUDA execution with block size " << BLOCK_SZ;
        return ss.str();
    }
};

struct GPUAtomic {
    using policy = RAJA::policy::cuda::cuda_atomic;
};

#elif defined (RAJA_ENABLE_HIP)
#include "RAJA/policy/hip.hpp"
#include "RAJA/policy/hip/atomic.hpp"

template<int M>
struct IsGPU<RAJA::hip_exec<M>> : public std::true_type {};

template<int BLOCK_SZ>
struct ExecPolicyGPU {
    using policy = RAJA::hip_exec<BLOCK_SZ, false>;
    static std::string PolicyName() {
        std::stringstream ss;
        ss << "HIP execution with block size " << BLOCK_SZ;
        return ss.str();
    }
};

struct GPUAtomic {
    using policy = RAJA::policy::hip::hip_atomic;
};

#endif

#define BLOCK_SZ 256
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

/// RAJA::atomicAdd and other RAJA namespace atomic calls are overloaded and have an ambiguous type
/// so they can't be passed as a template parameter.
/// The following macro disambiguates the call and provide a signature comaptible with the DESUL
/// wrapper. AtomicOperation must be a valid RAJA namespace atomic operation, like atomicAdd,
/// atomicMax, etc.
#define DECLARE_ATOMIC_WRAPPER(AtomicFunctorName, AtomicOperation)                          \
template<typename Policy>                                                                   \
struct AtomicFunctorName {                                                                  \
    const char* name = #AtomicOperation ;                                                   \
    template<typename AtomicType>                                                           \
    RAJA_HOST_DEVICE AtomicType operator()(AtomicType* acc, const AtomicType val) const {   \
        return RAJA::AtomicOperation(Policy {}, acc, val);                                  \
    }                                                                                       \
};                                                                                          \

DECLARE_ATOMIC_WRAPPER(AtomicAdd, atomicAdd)
DECLARE_ATOMIC_WRAPPER(AtomicSub, atomicSub)
DECLARE_ATOMIC_WRAPPER(AtomicMax, atomicMax)
DECLARE_ATOMIC_WRAPPER(AtomicMin, atomicMin)
DECLARE_ATOMIC_WRAPPER(AtomicInc, atomicInc)
DECLARE_ATOMIC_WRAPPER(AtomicDec, atomicDec)
DECLARE_ATOMIC_WRAPPER(AtomicAnd, atomicAnd)
DECLARE_ATOMIC_WRAPPER(AtomicOr, atomicOr)
DECLARE_ATOMIC_WRAPPER(AtomicXor, atomicXor)

/// ExecPolicy wrapper for OpenMP
struct ExecPolicyOMP {
    using policy = RAJA::omp_parallel_for_exec;;
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
    RAJA_HOST_DEVICE T operator()(T * acc, T value) const {
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

template <class ExecPolicy, typename AtomicType, bool test_array, typename AtomicImplType>
void TimeAtomicOp( AtomicImplType atomic_impl, uint64_t N, uint64_t num_iterations = 4, int array_size = 100, bool print_to_output = true) {
    RAJA::Timer timer;

    // Allocate memory
    AtomicType* device_value = nullptr;
    int len_array = test_array ? array_size : 1;
    camp::resources::Resource resource {RAJA::resources::get_resource<typename ExecPolicy::policy>::type::get_default()};
    device_value = resource.allocate<AtomicType>(len_array);

    timer.start();
    if (test_array) {
        for (int i = 0; i < num_iterations; ++i) {
            RAJA::forall<typename ExecPolicy::policy>(RAJA::RangeSegment(0, N),
            [=] RAJA_HOST_DEVICE(int tid)  {
                atomic_impl(&(device_value[tid % array_size]), AtomicType(1));
            });
        }
    } else {
        for (int i = 0; i < num_iterations; ++i) {
            RAJA::forall<typename ExecPolicy::policy>(RAJA::RangeSegment(0, N),
            [=] RAJA_HOST_DEVICE(int tid)  {
                atomic_impl(device_value, AtomicType(1));
            });
        }
    }

    resource.wait();
    timer.stop();
    resource.deallocate(device_value);

    double t = timer.elapsed();
    if (print_to_output) {
        std::cout << INDENT << INDENT << t << "s" << INDENT;
        std::cout << GetImplName(atomic_impl) << ", ";
        std::cout << "Number of atomics under contention " << array_size << ", ";
        std::cout << num_iterations * N << " many atomic operations" << ", ";
        std::cout << ExecPolicy::PolicyName();
        std::cout << std::endl;
    }
}

/// Holder struct for all atomic operations
template<typename AtomicDataType, typename Policy>
struct atomic_ops {
    using type = camp::list<std::pair<atomicWrapperDesul<AtomicDataType, desul::atomic_fetch_add>, AtomicAdd<Policy>>,
                                                std::pair<atomicWrapperDesul<AtomicDataType, desul::atomic_fetch_sub>, AtomicSub<Policy>>,
                                                std::pair<atomicWrapperDesul<AtomicDataType, desul::atomic_fetch_min>, AtomicMin<Policy>>,
                                                std::pair<atomicWrapperDesul<AtomicDataType, desul::atomic_fetch_max>, AtomicMax<Policy>>,
                                                // These two operations are unary and require some retooling to measure using the same testing loop.
                                                //std::pair<atomicWrapperDesul<AtomicDataType, desul::atomic_fetch_inc>, AtomicInc<Policy>>,
                                                //std::pair<atomicWrapperDesul<AtomicDataType, desul::atomic_fetch_dec>, AtomicDec<Policy>>,
                                                std::pair<atomicWrapperDesul<AtomicDataType, desul::atomic_fetch_and>, AtomicAnd<Policy>>,
                                                std::pair<atomicWrapperDesul<AtomicDataType, desul::atomic_fetch_or>, AtomicOr<Policy>>,
                                                std::pair<atomicWrapperDesul<AtomicDataType, desul::atomic_fetch_xor>, AtomicXor<Policy>>>;
};

int main (int argc, char* argv[]) {
    if (argc > 2) {
        RAJA_ABORT_OR_THROW("Usage: ./benchmark-atomic.exe <N> where N is the optional size of the benchmark loop");
    }
    uint64_t N = 1000000000;
    if (argc == 2) {
        N = std::stoi(argv[1]);
    }

    // Perform an untimed initialization of both desul and RAJA atomics.
    TimeAtomicOp<ExecPolicyGPU<BLOCK_SZ>, int, true>(AtomicAdd<typename GPUAtomic::policy>{}, N, 10, 1000, false);
    TimeAtomicOp<ExecPolicyGPU<BLOCK_SZ>, int, true>(atomicWrapperDesul<int, desul::atomic_fetch_add>{}, N, 10, 1000, false);
    // GPU benchmarks
    std::cout << "Executing GPU benchmarks" << std::endl;
    RAJA::for_each_type(atomic_ops<int, typename GPUAtomic::policy>::type{}, [&](auto type_pair) {
        auto desul_functor = type_pair.first;
        auto raja_functor = type_pair.second;
        std::cout << INDENT << "Executing " << raja_functor.name << " integer benchmarks" << std::endl;
        TimeAtomicOp<ExecPolicyGPU<BLOCK_SZ>, int, true>(desul_functor, N, 100, 10000);
        TimeAtomicOp<ExecPolicyGPU<BLOCK_SZ>, int, true>(raja_functor, N, 100, 10000);
        TimeAtomicOp<ExecPolicyGPU<BLOCK_SZ>, int, true>(desul_functor, N, 10, 1000);
        TimeAtomicOp<ExecPolicyGPU<BLOCK_SZ>, int, true>(raja_functor, N, 10, 1000);
        TimeAtomicOp<ExecPolicyGPU<BLOCK_SZ>, int, true>(desul_functor, N, 4, 10);
        TimeAtomicOp<ExecPolicyGPU<BLOCK_SZ>, int, true>(raja_functor, N, 4, 10);
    });
    std::cout << INDENT << "Executing atomic add double benchmarks" << std::endl;

    TimeAtomicOp<ExecPolicyGPU<BLOCK_SZ>, double, false>(AtomicAdd<typename GPUAtomic::policy> {}, N);
    TimeAtomicOp<ExecPolicyGPU<BLOCK_SZ>, double, false>(atomicWrapperDesul<double, desul::atomic_fetch_add> {}, N);

    // OpenMP benchmarks
    std::cout << "Executing OpenMP benchmarks" << std::endl;
    RAJA::for_each_type(atomic_ops<int, RAJA::policy::omp::omp_atomic>::type{}, [&](auto type_pair) {
        auto desul_functor = type_pair.first;
        auto raja_functor = type_pair.second;
        std::cout << INDENT << "Executing " << raja_functor.name << " integer benchmarks" << std::endl;
        TimeAtomicOp<ExecPolicyOMP, int, true>(desul_functor, N, 100, 10000);
        TimeAtomicOp<ExecPolicyOMP, int, true>(raja_functor, N, 100, 10000);
        TimeAtomicOp<ExecPolicyOMP, int, true>(desul_functor, N, 10, 1000);
        TimeAtomicOp<ExecPolicyOMP, int, true>(raja_functor, N, 10, 1000);
        TimeAtomicOp<ExecPolicyOMP, int, true>(desul_functor, N, 4, 10);
        TimeAtomicOp<ExecPolicyOMP, int, true>(raja_functor, N, 4, 10);
    });


    return 0;
}
