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

#ifndef KERNEL_SCATTERING_POLICY_H__
#define KERNEL_SCATTERING_POLICY_H__

#include <Kripke.h>

template <typename T>
struct ScatteringPolicy {};  // nm, g, gp, mat

#ifdef RAJA_COMPILER_ICC
template <>
struct ScatteringPolicy<NEST_DGZ_T>
    : RAJA::NestedPolicy<RAJA::ExecList<RAJA::seq_exec,
                                        RAJA::seq_exec,
                                        RAJA::seq_exec,
                                        RAJA::simd_exec> > {};

#else
template <>
struct ScatteringPolicy<NEST_DGZ_T>
    : RAJA::
          NestedPolicy<RAJA::ExecList<RAJA::seq_exec,
                                      RAJA::seq_exec,
                                      kripke_omp_for_nowait_exec,
                                      RAJA::simd_exec>,
                       kripke_OMP_Parallel<RAJA::Permute<RAJA::PERM_IJKL> > > {
};
#endif

template <>
struct ScatteringPolicy<NEST_DZG_T>
    : RAJA::NestedPolicy<RAJA::ExecList<kripke_omp_for_nowait_exec,
                                        RAJA::seq_exec,
                                        RAJA::seq_exec,
                                        RAJA::seq_exec>,
                         RAJA::Permute<RAJA::PERM_ILJK> > {};

template <>
struct ScatteringPolicy<NEST_GDZ_T>
    : RAJA::
          NestedPolicy<RAJA::ExecList<RAJA::seq_exec,
                                      RAJA::seq_exec,
                                      kripke_omp_for_nowait_exec,
                                      RAJA::seq_exec>,
                       kripke_OMP_Parallel<RAJA::Permute<RAJA::PERM_JKIL> > > {
};

template <>
struct ScatteringPolicy<NEST_GZD_T>
    : RAJA::NestedPolicy<RAJA::ExecList<RAJA::seq_exec,
                                        RAJA::seq_exec,
                                        kripke_omp_for_nowait_exec,
                                        RAJA::seq_exec>,
                         RAJA::Permute<RAJA::PERM_JKLI> > {};

template <>
struct ScatteringPolicy<NEST_ZDG_T>
    : RAJA::NestedPolicy<RAJA::ExecList<RAJA::seq_exec,
                                        RAJA::seq_exec,
                                        RAJA::seq_exec,
                                        kripke_omp_for_nowait_exec>,
                         RAJA::Permute<RAJA::PERM_LIJK> > {};

template <>
struct ScatteringPolicy<NEST_ZGD_T>
    : RAJA::NestedPolicy<RAJA::ExecList<RAJA::seq_exec,
                                        RAJA::seq_exec,
                                        RAJA::seq_exec,
                                        kripke_omp_for_nowait_exec>,
                         RAJA::Permute<RAJA::PERM_LJKI> > {};

#endif
