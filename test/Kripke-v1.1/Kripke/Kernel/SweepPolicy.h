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
#include<RAJA/Layout.hxx>
#include<RAJA/Forall.hxx>
  

template<typename T>
struct SweepPolicy{}; // d, g, z

template<>
struct SweepPolicy<NEST_DGZ_T> : RAJA::Forall3_Policy<omp_nowait, omp_nowait, sweep_seq_pol, 
                                                      RAJA::Forall3_OMP_Parallel<RAJA::Forall3_Permute<RAJA::PERM_IJK>>>
{};

template<>
struct SweepPolicy<NEST_DZG_T> : RAJA::Forall3_Policy<omp_pol, seq_pol, sweep_seq_pol, RAJA::Forall3_Permute<RAJA::PERM_IKJ>>
{};

template<>
struct SweepPolicy<NEST_GDZ_T> : RAJA::Forall3_Policy<omp_pol, omp_pol, sweep_seq_pol, RAJA::Forall3_Permute<RAJA::PERM_JIK>>
{};

template<>
struct SweepPolicy<NEST_GZD_T> : RAJA::Forall3_Policy<seq_pol, omp_pol, sweep_seq_pol, RAJA::Forall3_Permute<RAJA::PERM_JKI>>
{};

template<>
struct SweepPolicy<NEST_ZDG_T> : RAJA::Forall3_Policy<seq_pol, seq_pol, sweep_omp_pol, RAJA::Forall3_Permute<RAJA::PERM_KIJ>>
{};

template<>
struct SweepPolicy<NEST_ZGD_T> : RAJA::Forall3_Policy<seq_pol, seq_pol, sweep_omp_pol, RAJA::Forall3_Permute<RAJA::PERM_KJI>>
{};


#endif
