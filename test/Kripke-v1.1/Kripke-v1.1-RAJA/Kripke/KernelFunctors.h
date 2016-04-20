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
 
#ifndef KRIPKE_KERNEL_FUNCTORS__
#define KRIPKE_KERNEL_FUNCTORS__


template<typename PHI, typename ELL, typename PSI>
struct LTimesFcn {

  int group0;
  PHI phi;
  ELL ell;
  PSI psi;

  RAJA_INLINE
  RAJA_HOST_DEVICE
  LTimesFcn(PHI const &phi_, ELL const &ell_, PSI const &psi_, int g0) : 
    phi(phi_), ell(ell_), psi(psi_), group0(g0)
  {}

  RAJA_INLINE
  RAJA_HOST_DEVICE 
  void operator()(IMoment nm, IDirection d, IGroup g, IZone z) const {

    IGlobalGroup g_global( (*g) + group0);

    phi(nm, g_global, z) += ell(d, nm) * psi(d, g, z);
  }

};



template<typename PSI, typename ELL_PLUS, typename PHI>
struct LPlusTimesFcn {

  int      group0;
  PSI      rhs;
  ELL_PLUS ell_plus;
  PHI      phi_out;

  RAJA_INLINE
  RAJA_HOST_DEVICE
  LPlusTimesFcn(PSI const &rhs_, ELL_PLUS const &ell_plus_, PHI const &phi_out_, int g0) :
    rhs(rhs_), ell_plus(ell_plus_), phi_out(phi_out_), group0(g0)
  {}

  RAJA_INLINE
  RAJA_HOST_DEVICE 
  void operator()(IMoment nm, IDirection d, IGroup g, IZone z) const {

    IGlobalGroup g_global( (*g) + group0);

    rhs(d, g, z) += ell_plus(d, nm) * phi_out(nm, g_global, z);
  }

};



template<typename PHI, typename SIGS, typename MZ, typename MM, typename MF, typename MC>
struct ScatteringFcn {

  PHI  phi;
  PHI  phi_out;
  SIGS sigs;

  MZ   mixed_to_zones;
  MM   mixed_material;
  MF   mixed_fraction;
  MC   moment_to_coeff;

  RAJA_INLINE
  RAJA_HOST_DEVICE
  ScatteringFcn(PHI phi_, PHI phi_out_, SIGS sigs_, MZ mz, MM mm, MF mf, MC mc) :
    phi(phi_), phi_out(phi_out_), sigs(sigs_),
    mixed_to_zones(mz),
    mixed_material(mm),
    mixed_fraction(mf),
    moment_to_coeff(mc)
  {}

  RAJA_INLINE
  RAJA_HOST_DEVICE
  void operator()(IMoment nm, IGlobalGroup g, IGlobalGroup gp, IMix mix) const {

    ILegendre n = moment_to_coeff(nm);
    IZone zone = mixed_to_zones(mix);
    IMaterial material = mixed_material(mix);
    double fraction = mixed_fraction(mix);

    phi_out(nm, gp, zone) +=
      sigs(n, g, gp, material) * phi(nm, g, zone) * fraction;
  }

};




template<typename PHI, typename MZ, typename MM, typename MF>
struct SourceFcn {

  PHI  phi_out;

  MZ   mixed_to_zones;
  MM   mixed_material;
  MF   mixed_fraction;

  RAJA_INLINE
  RAJA_HOST_DEVICE
  SourceFcn(PHI phi_out_, MZ mz, MM mm, MF mf) :
    phi_out(phi_out_), 
    mixed_to_zones(mz),
    mixed_material(mm),
    mixed_fraction(mf)
  {}

  RAJA_INLINE
  RAJA_HOST_DEVICE
  void operator()(IGlobalGroup g, IMix mix) const {
    IZone zone = mixed_to_zones(mix);
    IMaterial material = mixed_material(mix);
    double fraction = mixed_fraction(mix);

    if(*material == 0){
      phi_out(IMoment(0), g, zone) += 1.0 * fraction;
    }
  }
  
};

template<typename DIR, typename PSI, typename SIGT, typename DX, typename DY, typename DZ,
  typename ZONE_LAYOUT, typename FACEI, typename FACEJ, typename FACEK,
  typename IDXI, typename IDXJ, typename IDXK>
struct SweepFcn {

  DIR direction;
  PSI rhs;
  PSI psi;
  SIGT sigt;
  DX dx;
  DY dy;
  DZ dz;
  ZONE_LAYOUT zone_layout;
  FACEI face_lf;
  FACEJ face_fr;
  FACEK face_bo;
  IDXI idx_to_i;
  IDXJ idx_to_j;
  IDXK idx_to_k;

  RAJA_INLINE
  RAJA_HOST_DEVICE
  SweepFcn(DIR direction_, PSI rhs_, PSI psi_, SIGT sigt_, DX dx_, DY dy_, DZ dz_,
      ZONE_LAYOUT zone_layout_, FACEI face_lf_, FACEJ face_fr_, FACEK face_bo_,
      IDXI idx_to_i_, IDXJ idx_to_j_, IDXK idx_to_k_) :
    direction(direction_),
    rhs(rhs_),
    psi(psi_),
    sigt(sigt_),
    dx(dx_),
    dy(dy_),
    dz(dz_),
    zone_layout(zone_layout_),
    face_lf(face_lf_),
    face_fr(face_fr_),
    face_bo(face_bo_),
    idx_to_i(idx_to_i_),
    idx_to_j(idx_to_j_),
    idx_to_k(idx_to_k_)
  {}

  RAJA_INLINE
  RAJA_HOST_DEVICE
  void operator()(IDirection d, IGroup g, IZoneIdx zone_idx) const {
    IZoneI i = idx_to_i(zone_idx);
    IZoneJ j = idx_to_j(zone_idx);
    IZoneK k = idx_to_k(zone_idx);

    double const xcos_dxi = 2.0 * direction(d).xcos / dx(i+1);
    double const ycos_dyj = 2.0 * direction(d).ycos / dy(j+1);
    double const zcos_dzk = 2.0 * direction(d).zcos / dz(k+1);

    IZone z = zone_layout(i,j,k);

    // Calculate new zonal flux
    double const psi_d_g_z = (
          rhs(d,g,z)
        + face_lf(d,g,j,k) * xcos_dxi
        + face_fr(d,g,i,k) * ycos_dyj
        + face_bo(d,g,i,j) * zcos_dzk)
        / (xcos_dxi + ycos_dyj + zcos_dzk + sigt(g,z) );

    psi(d,g,z) = psi_d_g_z;

    // Apply diamond-difference relationships
    face_lf(d,g,j,k) = 2.0 * psi_d_g_z - face_lf(d,g,j,k);
    face_fr(d,g,i,k) = 2.0 * psi_d_g_z - face_fr(d,g,i,k);
    face_bo(d,g,i,j) = 2.0 * psi_d_g_z - face_bo(d,g,i,j);
  }
};


template<typename REDUCE, typename DIR, typename PSI, typename VOL>
struct ParticleEditFcn {

  REDUCE &part_reduce;
  DIR direction;
  PSI psi;
  VOL volume;

  RAJA_INLINE
  RAJA_HOST_DEVICE
  ParticleEditFcn(REDUCE &reduce_, DIR &direction_, PSI psi_, VOL vol_) :
    part_reduce(reduce_),
    direction(direction_),
    psi(psi_),
    volume(vol_)
  {}


#pragma nv_exec_check_disable
  RAJA_INLINE
  RAJA_HOST_DEVICE
  void operator()(IDirection d, IGroup g, IZone z) const {
    part_reduce += direction(d).w * psi(d,g,z) * volume(z);              
  }
  
};

#endif
