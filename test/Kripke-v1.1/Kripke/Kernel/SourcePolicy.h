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

#ifndef KERNEL_SOURCE_POLICY_H__
#define KERNEL_SOURCE_POLICY_H__

#include<Kripke.h>
#include<RAJA/Layout.hxx>
#include<RAJA/Forall.hxx>


template<typename T>
struct SourcePolicy : RAJA::Forall2_Policy<> {}; // g,mix

template<>
struct SourcePolicy<NEST_DGZ_T> :  RAJA::Forall2_Policy<omp_nowait, seq_pol,
                      RAJA::Forall2_OMP_Parallel<
                      RAJA::Forall2_Permute<RAJA::PERM_IJ>
									  >
									>
{};

template<>
struct SourcePolicy<NEST_DZG_T> : RAJA::Forall2_Policy<omp_pol, seq_pol, RAJA::Forall2_Permute<RAJA::PERM_JI> >
{};

template<>
struct SourcePolicy<NEST_GDZ_T> : RAJA::Forall2_Policy<seq_pol, seq_pol, RAJA::Forall2_Permute<RAJA::PERM_IJ> >
{};

template<>
struct SourcePolicy<NEST_GZD_T> : RAJA::Forall2_Policy<omp_pol, seq_pol, RAJA::Forall2_Permute<RAJA::PERM_IJ> >
{};

template<>
struct SourcePolicy<NEST_ZDG_T> : RAJA::Forall2_Policy<omp_pol, seq_pol, RAJA::Forall2_Permute<RAJA::PERM_JI> >
{};

template<>
struct SourcePolicy<NEST_ZGD_T> : RAJA::Forall2_Policy<omp_pol, seq_pol, RAJA::Forall2_Permute<RAJA::PERM_JI> >
{};

#endif
