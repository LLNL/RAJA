/*
 * NOTICE
 *
 * This work was produced at the Lawrence Livermore National Laboratory (LLNL)
 * under contract no. DE-AC-52-07NA27344 (Contract 44) between the U.S.
 * Department of Energy (DOE) and Lawrence Livermore National Security, LLC
 * (LLNS) for the operation of LLNL. The rights of the Federal Government are
 * reserved under Contract 44.
 *
 * DISCLAIMER
 *
 * This work was prepared as an account of work sponsored by an agency of the
 * United States Government. Neither the United States Government nor Lawrence
 * Livermore National Security, LLC nor any of their employees, makes any
 * warranty, express or implied, or assumes any liability or responsibility
 * for the accuracy, completeness, or usefulness of any information, apparatus,
 * product, or process disclosed, or represents that its use would not infringe
 * privately-owned rights. Reference herein to any specific commercial products,
 * process, or service by trade name, trademark, manufacturer or otherwise does
 * not necessarily constitute or imply its endorsement, recommendation, or
 * favoring by the United States Government or Lawrence Livermore National
 * Security, LLC. The views and opinions of authors expressed herein do not
 * necessarily state or reflect those of the United States Government or
 * Lawrence Livermore National Security, LLC, and shall not be used for
 * advertising or product endorsement purposes.
 *
 * NOTIFICATION OF COMMERCIAL USE
 *
 * Commercialization of this product is prohibited without notifying the
 * Department of Energy (DOE) or Lawrence Livermore National Security.
 */

#ifndef KERNEL_SWEEP_POLICY_H__
#define KERNEL_SWEEP_POLICY_H__

#include<Kripke.h>
  
typedef RAJA::IndexSet::ExecPolicy<RAJA::seq_segit, RAJA::simd_exec> sweep_seq_exec;
#ifdef RAJA_ENABLE_OPENMP
typedef RAJA::IndexSet::ExecPolicy<RAJA::seq_segit, RAJA::omp_parallel_for_exec> sweep_omp_exec;
#else
using sweep_omp_exec = sweep_seq_exec;
#endif

template<typename T>
struct SweepPolicy{}; // d, g, z

#ifdef RAJA_COMPILER_ICC
template<>
struct SweepPolicy<NEST_DGZ_T> : RAJA::NestedPolicy<
                                    RAJA::ExecList<RAJA::seq_exec, 
                                                   RAJA::seq_exec, 
                                                   sweep_seq_exec>
                                  >
{};

#else
template<>
struct SweepPolicy<NEST_DGZ_T> : RAJA::NestedPolicy<
                                    RAJA::ExecList<kripke_omp_collapse_nowait_exec, 
                                                   kripke_omp_collapse_nowait_exec, 
                                                   sweep_seq_exec>,
                                    kripke_OMP_Parallel<RAJA::Execute>
                                 >
{};
#endif
template<>
struct SweepPolicy<NEST_DZG_T> : RAJA::NestedPolicy<
                                    RAJA::ExecList<kripke_omp_for_nowait_exec, 
                                                   RAJA::seq_exec, 
                                                   sweep_seq_exec>, 
                                    RAJA::Permute<RAJA::PERM_IKJ>
                                 >
{};

template<>
struct SweepPolicy<NEST_GDZ_T> : RAJA::NestedPolicy<
                                    RAJA::ExecList<kripke_omp_for_nowait_exec, 
                                                   kripke_omp_for_nowait_exec, 
                                                   sweep_seq_exec>, 
                                    RAJA::Permute<RAJA::PERM_JIK>
                                 >
{};

template<>
struct SweepPolicy<NEST_GZD_T> : RAJA::NestedPolicy<
                                    RAJA::ExecList<RAJA::seq_exec, 
                                                   kripke_omp_for_nowait_exec, 
                                                   sweep_seq_exec>, 
                                    RAJA::Permute<RAJA::PERM_JKI>
                                 >
{};

template<>
struct SweepPolicy<NEST_ZDG_T> : RAJA::NestedPolicy<
                                    RAJA::ExecList<RAJA::seq_exec, 
                                                   RAJA::seq_exec, 
                                                   sweep_omp_exec>, 
                                    RAJA::Permute<RAJA::PERM_KIJ>
                                 >
{};

template<>
struct SweepPolicy<NEST_ZGD_T> : RAJA::NestedPolicy<
                                    RAJA::ExecList<RAJA::seq_exec, 
                                                   RAJA::seq_exec, 
                                                   sweep_omp_exec>, 
                                    RAJA::Permute<RAJA::PERM_KJI>
                                 >
{};


#endif
