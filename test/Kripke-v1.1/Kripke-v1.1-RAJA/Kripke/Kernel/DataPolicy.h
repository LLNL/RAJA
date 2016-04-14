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

#ifndef KERNEL_VARIABLE_POLICY_H__
#define KERNEL_VARIABLE_POLICY_H__

#include<Kripke.h>
#include<Kripke/Directions.h>
#include<Kripke/DView.h>
#include<RAJA/Layout.hxx>
#include<RAJA/Forall.hxx>
#include<RAJA/IndexValue.hxx>


/*
 * Define strongly-typed indices used in Kripke
 */
RAJA_INDEX_VALUE(IMaterial, "IMaterial");     // Material ID
RAJA_INDEX_VALUE(ILegendre, "ILegendre");     // Legendre expansion coefficient
RAJA_INDEX_VALUE(IMoment, "IMoment");       // Spherical harmonic moment
RAJA_INDEX_VALUE(IDirection, "IDirection");    // Local direction
RAJA_INDEX_VALUE(IGlobalGroup, "IGlobalGroup");  // Global energy group
RAJA_INDEX_VALUE(IGroup, "IGroup");        // Local energy group
RAJA_INDEX_VALUE(IZone, "IZone");         // Cannonical zone number
RAJA_INDEX_VALUE(IZoneIdx, "IZoneIdx");      // Mapped zone index (sequential in hyperplane)
RAJA_INDEX_VALUE(IMix, "IMix");          // Mixed element slot
RAJA_INDEX_VALUE(IZoneI, "IZoneI");        // zone on the I boundary face
RAJA_INDEX_VALUE(IZoneJ, "IZoneJ");        // zone on the K boundary face
RAJA_INDEX_VALUE(IZoneK, "IZoneK");        // zone on the K boundary face



/**
 * Layout policies that don't change with nesting.
 */
struct FixedLayoutPolicy {
  typedef DLayout2d<RAJA::PERM_JI, IDirection, IMoment> Layout_Ell;
  typedef DLayout2d<RAJA::PERM_IJ, IDirection, IMoment> Layout_EllPlus;

  typedef DLayout3d<RAJA::PERM_KJI, IZoneI, IZoneJ, IZoneK, IZone> TLayout_Zone;
};


/**
 * Layout policies tied directly to nesting.
 */
template<typename T>
struct NestingPolicy{};

template<>
struct NestingPolicy<NEST_DGZ_T> : public FixedLayoutPolicy {
  typedef DLayout3d<RAJA::PERM_IJK, IDirection, IGroup, IZone>    Layout_Psi;
  typedef DLayout3d<RAJA::PERM_IJK, IMoment, IGlobalGroup, IZone> Layout_Phi;
  typedef DLayout4d<RAJA::PERM_IJKL, ILegendre, IGlobalGroup, IGlobalGroup, IMaterial> Layout_SigS;
  typedef DLayout2d<RAJA::PERM_IJ, IGroup, IZone> Layout_SigT;
  
  typedef DLayout4d<RAJA::PERM_IJLK, IDirection, IGroup, IZoneJ, IZoneK> Layout_FaceI;
  typedef DLayout4d<RAJA::PERM_IJLK, IDirection, IGroup, IZoneI, IZoneK> Layout_FaceJ;
  typedef DLayout4d<RAJA::PERM_IJLK, IDirection, IGroup, IZoneI, IZoneJ> Layout_FaceK;
};

template<>
struct NestingPolicy<NEST_DZG_T> : public FixedLayoutPolicy {
  typedef DLayout3d<RAJA::PERM_IKJ, IDirection, IGroup, IZone>    Layout_Psi;
  typedef DLayout3d<RAJA::PERM_IKJ, IMoment, IGlobalGroup, IZone> Layout_Phi;
  typedef DLayout4d<RAJA::PERM_ILJK, ILegendre, IGlobalGroup, IGlobalGroup, IMaterial> Layout_SigS;
  typedef DLayout2d<RAJA::PERM_JI, IGroup, IZone> Layout_SigT;

  typedef DLayout4d<RAJA::PERM_ILKJ, IDirection, IGroup, IZoneJ, IZoneK> Layout_FaceI;
  typedef DLayout4d<RAJA::PERM_ILKJ, IDirection, IGroup, IZoneI, IZoneK> Layout_FaceJ;
  typedef DLayout4d<RAJA::PERM_ILKJ, IDirection, IGroup, IZoneI, IZoneJ> Layout_FaceK;
};

template<>
struct NestingPolicy<NEST_GDZ_T> : public FixedLayoutPolicy {
  typedef DLayout3d<RAJA::PERM_JIK, IDirection, IGroup, IZone>    Layout_Psi;
  typedef DLayout3d<RAJA::PERM_JIK, IMoment, IGlobalGroup, IZone> Layout_Phi;
  typedef DLayout4d<RAJA::PERM_JKIL, ILegendre, IGlobalGroup, IGlobalGroup, IMaterial> Layout_SigS;
  typedef DLayout2d<RAJA::PERM_IJ, IGroup, IZone> Layout_SigT;

  typedef DLayout4d<RAJA::PERM_JILK, IDirection, IGroup, IZoneJ, IZoneK> Layout_FaceI;
  typedef DLayout4d<RAJA::PERM_JILK, IDirection, IGroup, IZoneI, IZoneK> Layout_FaceJ;
  typedef DLayout4d<RAJA::PERM_JILK, IDirection, IGroup, IZoneI, IZoneJ> Layout_FaceK;
};

template<>
struct NestingPolicy<NEST_GZD_T> : public FixedLayoutPolicy {
  typedef DLayout3d<RAJA::PERM_JKI, IDirection, IGroup, IZone>    Layout_Psi;
  typedef DLayout3d<RAJA::PERM_JKI, IMoment, IGlobalGroup, IZone> Layout_Phi;
  typedef DLayout4d<RAJA::PERM_JKLI, ILegendre, IGlobalGroup, IGlobalGroup, IMaterial> Layout_SigS;
  typedef DLayout2d<RAJA::PERM_IJ, IGroup, IZone> Layout_SigT;

  typedef DLayout4d<RAJA::PERM_JLKI, IDirection, IGroup, IZoneJ, IZoneK> Layout_FaceI;
  typedef DLayout4d<RAJA::PERM_JLKI, IDirection, IGroup, IZoneI, IZoneK> Layout_FaceJ;
  typedef DLayout4d<RAJA::PERM_JLKI, IDirection, IGroup, IZoneI, IZoneJ> Layout_FaceK;
};

template<>
struct NestingPolicy<NEST_ZDG_T> : public FixedLayoutPolicy {
  typedef DLayout3d<RAJA::PERM_KIJ, IDirection, IGroup, IZone>    Layout_Psi;
  typedef DLayout3d<RAJA::PERM_KIJ, IMoment, IGlobalGroup, IZone> Layout_Phi;
  typedef DLayout4d<RAJA::PERM_LIJK, ILegendre, IGlobalGroup, IGlobalGroup, IMaterial> Layout_SigS;
  typedef DLayout2d<RAJA::PERM_JI, IGroup, IZone> Layout_SigT;

  typedef DLayout4d<RAJA::PERM_LKIJ, IDirection, IGroup, IZoneJ, IZoneK> Layout_FaceI;
  typedef DLayout4d<RAJA::PERM_LKIJ, IDirection, IGroup, IZoneI, IZoneK> Layout_FaceJ;
  typedef DLayout4d<RAJA::PERM_LKIJ, IDirection, IGroup, IZoneI, IZoneJ> Layout_FaceK;
};

template<>
struct NestingPolicy<NEST_ZGD_T> : public FixedLayoutPolicy {
  typedef DLayout3d<RAJA::PERM_KJI, IDirection, IGroup, IZone>    Layout_Psi;
  typedef DLayout3d<RAJA::PERM_KJI, IMoment, IGlobalGroup, IZone> Layout_Phi;
  typedef DLayout4d<RAJA::PERM_LJKI, ILegendre, IGlobalGroup, IGlobalGroup, IMaterial> Layout_SigS;
  typedef DLayout2d<RAJA::PERM_JI, IGroup, IZone> Layout_SigT;

  typedef DLayout4d<RAJA::PERM_LKJI, IDirection, IGroup, IZoneJ, IZoneK> Layout_FaceI;
  typedef DLayout4d<RAJA::PERM_LKJI, IDirection, IGroup, IZoneI, IZoneK> Layout_FaceJ;
  typedef DLayout4d<RAJA::PERM_LKJI, IDirection, IGroup, IZoneI, IZoneJ> Layout_FaceK;
};


/**
 * Views that have fixed policies
 */
struct FixedViewPolicy {
  typedef DView1d<double, DLayout1d<RAJA::PERM_I, IZoneI> >View_dx;
  typedef DView1d<double, DLayout1d<RAJA::PERM_I, IZoneJ> > View_dy;
  typedef DView1d<double, DLayout1d<RAJA::PERM_I, IZoneK> > View_dz;
  typedef DView1d<Directions, DLayout1d<RAJA::PERM_I, IDirection> > View_Directions;
  
  typedef DView1d<IZoneI, DLayout1d<RAJA::PERM_I, IZoneIdx> > View_IdxToI;
  typedef DView1d<IZoneJ, DLayout1d<RAJA::PERM_I, IZoneIdx> > View_IdxToJ;
  typedef DView1d<IZoneK, DLayout1d<RAJA::PERM_I, IZoneIdx> > View_IdxToK;

  typedef DView1d<IZone, DLayout1d<RAJA::PERM_I, IMix> > View_MixedToZones;
  typedef DView1d<IMaterial, DLayout1d<RAJA::PERM_I, IMix> > View_MixedToMaterial;
  typedef DView1d<double, DLayout1d<RAJA::PERM_I, IMix> > View_MixedToFraction;
  typedef DView1d<ILegendre, DLayout1d<RAJA::PERM_I, IMoment> > View_MomentToCoeff;
};

/**
 * Views with policies that vary between nestings.
 */
template<typename T>
struct ViewPolicy : public FixedViewPolicy {
  // Discrete and Moment Unknowns
  typedef DView3d<double, typename T::Layout_Psi> View_Psi;
  typedef DView3d<double, typename T::Layout_Phi> View_Phi;

  // Spatial domain face indices
  typedef DView4d<double, typename T::Layout_FaceI> View_FaceI;
  typedef DView4d<double, typename T::Layout_FaceJ> View_FaceJ;
  typedef DView4d<double, typename T::Layout_FaceK> View_FaceK;

  // L and L+ matrices
  typedef DView2d<double, typename T::Layout_Ell> View_Ell;
  typedef DView2d<double, typename T::Layout_EllPlus> View_EllPlus;

  // Data tables
  typedef DView4d<double, typename T::Layout_SigS> View_SigS;
  typedef DView2d<double, typename T::Layout_SigT> View_SigT;
};


/**
 * Combined Policies for Layouts, Views.
 *
 * A convenience class: makes it easier to include in application.
 */
struct FixedDataPolicy {
  static const int memory_alignment = 64;
};

template<typename T>
struct DataPolicy : public FixedDataPolicy, public NestingPolicy<T>, public ViewPolicy<NestingPolicy<T> >
{
};

#endif
