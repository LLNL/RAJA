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

#include<Kripke/Kernel.h>
#include<Kripke/Grid.h>
#include<Kripke/SubTVec.h>

#include<RAJA/RAJA.hxx>

#include<Kripke/Kernel/Kernel_3d_GDZ.h>
#include<Kripke/Kernel/Kernel_3d_DGZ.h>
#include<Kripke/Kernel/Kernel_3d_ZDG.h>
#include<Kripke/Kernel/Kernel_3d_DZG.h>
#include<Kripke/Kernel/Kernel_3d_ZGD.h>
#include<Kripke/Kernel/Kernel_3d_GZD.h>

#include<Kripke/Kernel/DataPolicy.h>


/**
 * Factory to create a kernel object for the specified nesting
 */
Kernel *createKernel(Nesting_Order nest, int num_dims){
  if(num_dims == 3){
    switch(nest){
    case NEST_GDZ:
      return new Kernel_3d_GDZ();
    case NEST_DGZ:
      return new Kernel_3d_DGZ();
    case NEST_ZDG:
      return new Kernel_3d_ZDG();
    case NEST_DZG:
      return new Kernel_3d_DZG();
    case NEST_ZGD:
      return new Kernel_3d_ZGD();
    case NEST_GZD:
      return new Kernel_3d_GZD();
    }
  }

  KripkeAbort("Unknown nesting order %d or number of dimensions %d\n", (int)nest, num_dims);
  return NULL;
}


Kernel::Kernel(Nesting_Order nest) :
  nesting_order(nest)
{}

Kernel::~Kernel(){
}



#include<Kripke/Kernel/LTimesPolicy.h>

namespace {

template<typename nest_type>
RAJA_INLINE
void kernel_LTimes(Grid_Data &domain) {

  typedef DataPolicy<nest_type> POL;
  // Zero Phi
  FORALL_ZONESETS(seq_pol, domain, sdom_id, sdom)
    sdom.phi->clear(0.0);
  END_FORALL

  // Loop over Subdomains
  FORALL_SUBDOMAINS(seq_pol, domain, sdom_id, sdom)

    // Get dimensioning
    int group0 = sdom.group0;

    // Get pointers
    typename POL::View_Psi psi(domain, sdom_id, sdom.psi->ptr());
    typename POL::View_Phi phi(domain, sdom_id, sdom.phi->ptr());
    typename POL::View_Ell ell(domain, sdom_id, sdom.ell->ptr());

    dForallN<LTimesPolicy<nest_type>, IMoment, IDirection, IGroup, IZone>(
      domain, sdom_id,
      RAJA_LAMBDA (IMoment nm, IDirection d, IGroup g, IZone z){

        IGlobalGroup g_global( (*g) + group0);

        phi(nm, g_global, z) += ell(d, nm) * psi(d, g, z);
      });

  END_FORALL
}

} // anon namespace

void Kernel::LTimes(Grid_Data *domain) {
  switch(nesting_order){
    case NEST_DGZ: kernel_LTimes<NEST_DGZ_T>(*domain); break;
    case NEST_DZG: kernel_LTimes<NEST_DZG_T>(*domain); break;
    case NEST_GDZ: kernel_LTimes<NEST_GDZ_T>(*domain); break;
    case NEST_GZD: kernel_LTimes<NEST_GZD_T>(*domain); break;
    case NEST_ZDG: kernel_LTimes<NEST_ZDG_T>(*domain); break;
    case NEST_ZGD: kernel_LTimes<NEST_ZGD_T>(*domain); break;
  }
}

#include<Kripke/Kernel/LPlusTimesPolicy.h>

namespace {
template<typename nest_type>
RAJA_INLINE
void kernel_LPlusTimes(Grid_Data &domain) {

  typedef DataPolicy<nest_type> POL;

  // Loop over Subdomains
  FORALL_SUBDOMAINS(seq_pol, domain, sdom_id, sdom)
    sdom.rhs->clear(0.0);
  END_FORALL

  FORALL_SUBDOMAINS(seq_pol, domain, sdom_id, sdom)

    // Get dimensioning
    int group0 = sdom.group0;

    // Get pointers
    typename POL::View_Psi     rhs     (domain, sdom_id, sdom.rhs->ptr());
    typename POL::View_Phi     phi_out (domain, sdom_id, sdom.phi_out->ptr());
    typename POL::View_EllPlus ell_plus(domain, sdom_id, sdom.ell_plus->ptr());

    dForallN<LPlusTimesPolicy<nest_type>, IMoment, IDirection, IGroup, IZone>(
      domain, sdom_id,
      RAJA_LAMBDA (IMoment nm, IDirection d, IGroup g, IZone z){

        IGlobalGroup g_global( (*g) + group0);

        rhs(d, g, z) += ell_plus(d, nm) * phi_out(nm, g_global, z);
      });

  END_FORALL
  
}

} // anon namespace

void Kernel::LPlusTimes(Grid_Data *domain) {
  switch(nesting_order){
    case NEST_DGZ: kernel_LPlusTimes<NEST_DGZ_T>(*domain); break;
    case NEST_DZG: kernel_LPlusTimes<NEST_DZG_T>(*domain); break;
    case NEST_GDZ: kernel_LPlusTimes<NEST_GDZ_T>(*domain); break;
    case NEST_GZD: kernel_LPlusTimes<NEST_GZD_T>(*domain); break;
    case NEST_ZDG: kernel_LPlusTimes<NEST_ZDG_T>(*domain); break;
    case NEST_ZGD: kernel_LPlusTimes<NEST_ZGD_T>(*domain); break;
  }
}

/**
  Compute scattering source term phi_out from flux moments in phi.
  phi_out(gp,z,nm) = sum_g { sigs(g, n, gp) * phi(g,z,nm) }
*/
#include<Kripke/Kernel/ScatteringPolicy.h>
namespace {
template<typename nest_type>
RAJA_INLINE
void kernel_scattering(Grid_Data &domain) {
  
  typedef DataPolicy<nest_type> POL;

  // Zero out source terms
  FORALL_ZONESETS(seq_pol, domain, sdom_id, sdom)
    sdom.phi_out->clear(0.0);
  END_FORALL

  // Loop over zoneset subdomains
  FORALL_ZONESETS(seq_pol, domain, sdom_id, sdom)

    typename POL::View_Phi     phi    (domain, sdom_id, sdom.phi->ptr());
    typename POL::View_Phi     phi_out(domain, sdom_id, sdom.phi_out->ptr());
    typename POL::View_SigS    sigs   (domain, sdom_id, domain.sigs->ptr());

    typename POL::View_MixedToZones    mixed_to_zones (domain, sdom_id, (IZone*)&sdom.mixed_to_zones[0]);
    typename POL::View_MixedToMaterial mixed_material (domain, sdom_id, (IMaterial*)&sdom.mixed_material[0]);
    typename POL::View_MixedToFraction mixed_fraction (domain, sdom_id, &sdom.mixed_fraction[0]);
    typename POL::View_MomentToCoeff   moment_to_coeff(domain, sdom_id, (ILegendre*)&domain.moment_to_coeff[0]);

    dForallN<ScatteringPolicy<nest_type>, IMoment, IGlobalGroup, IGlobalGroup, IMix>(
      domain, sdom_id,
      RAJA_LAMBDA (IMoment nm, IGlobalGroup g, IGlobalGroup gp, IMix mix){
      
        ILegendre n = moment_to_coeff(nm);
        IZone zone = mixed_to_zones(mix);
        IMaterial material = mixed_material(mix);
        double fraction = mixed_fraction(mix);

        phi_out(nm, gp, zone) +=
          sigs(n, g, gp, material) * phi(nm, g, zone) * fraction;

      });  // forall
        
  END_FORALL // zonesets

}

} // anon namespace

void Kernel::scattering(Grid_Data *domain) {
  switch(nesting_order){
    case NEST_DGZ: kernel_scattering<NEST_DGZ_T>(*domain); break;
    case NEST_DZG: kernel_scattering<NEST_DZG_T>(*domain); break;
    case NEST_GDZ: kernel_scattering<NEST_GDZ_T>(*domain); break;
    case NEST_GZD: kernel_scattering<NEST_GZD_T>(*domain); break;
    case NEST_ZDG: kernel_scattering<NEST_ZDG_T>(*domain); break;
    case NEST_ZGD: kernel_scattering<NEST_ZGD_T>(*domain); break;
  }
}
  
/**
 * Add an isotropic source, with flux of 1, to every zone with Region 1
 * (or material 0).
 *
 * Since it's isotropic, we're just adding this to nm=0.
 */
#include<Kripke/Kernel/SourcePolicy.h>
namespace {
template<typename nest_type>
RAJA_INLINE
void kernel_source(Grid_Data &domain) {
  typedef DataPolicy<nest_type> POL;

  // Loop over zoneset subdomains
  FORALL_ZONESETS(seq_pol, domain, sdom_id, sdom)
    typename POL::View_Phi             phi_out       (domain, sdom_id, sdom.phi_out->ptr());
    typename POL::View_MixedToZones    mixed_to_zones(domain, sdom_id, (IZone*)&sdom.mixed_to_zones[0]);
    typename POL::View_MixedToMaterial mixed_material(domain, sdom_id, (IMaterial*)&sdom.mixed_material[0]);
    typename POL::View_MixedToFraction mixed_fraction(domain, sdom_id, &sdom.mixed_fraction[0]);

    dForallN<SourcePolicy<nest_type>, IGlobalGroup, IMix >(
      domain, sdom_id,
      RAJA_LAMBDA (IGlobalGroup g, IMix mix){
        IZone zone = mixed_to_zones(mix);
        IMaterial material = mixed_material(mix);
        double fraction = mixed_fraction(mix);

        if(*material == 0){
          phi_out(IMoment(0), g, zone) += 1.0 * fraction;
        }
    }); // forall

  END_FORALL
}

} // anon namespace



void Kernel::source(Grid_Data *domain) {
  switch(nesting_order){
    case NEST_DGZ: kernel_source<NEST_DGZ_T>(*domain); break;
    case NEST_DZG: kernel_source<NEST_DZG_T>(*domain); break;
    case NEST_GDZ: kernel_source<NEST_GDZ_T>(*domain); break;
    case NEST_GZD: kernel_source<NEST_GZD_T>(*domain); break;
    case NEST_ZDG: kernel_source<NEST_ZDG_T>(*domain); break;
    case NEST_ZGD: kernel_source<NEST_ZGD_T>(*domain); break;
  }
}


#include<Kripke/Kernel/SweepPolicy.h>

namespace {
template<typename nest_type>
RAJA_INLINE
void kernel_sweep(Grid_Data &domain, int sdom_id) {
  typedef DataPolicy<nest_type> POL;

  Subdomain *sdom = &domain.subdomains[sdom_id];

  typename POL::View_Directions direction(domain, sdom_id, sdom->directions);

  typename POL::View_Psi     rhs (domain, sdom_id, sdom->rhs->ptr());
  typename POL::View_Psi     psi (domain, sdom_id, sdom->psi->ptr());
  typename POL::View_SigT    sigt(domain, sdom_id, sdom->sigt->ptr());

  typename POL::View_dx      dx(domain, sdom_id, &sdom->deltas[0][0]);
  typename POL::View_dy      dy(domain, sdom_id, &sdom->deltas[1][0]);
  typename POL::View_dz      dz(domain, sdom_id, &sdom->deltas[2][0]);

  typename POL::TLayout_Zone zone_layout(domain, sdom_id);

  typename POL::View_FaceI face_lf(domain, sdom_id, sdom->plane_data[0]->ptr());
  typename POL::View_FaceJ face_fr(domain, sdom_id, sdom->plane_data[1]->ptr());
  typename POL::View_FaceK face_bo(domain, sdom_id, sdom->plane_data[2]->ptr());

  // All directions have same id,jd,kd, since these are all one Direction Set
  // So pull that information out now
  Grid_Sweep_Block const &extent = sdom->sweep_block;
  typename POL::View_IdxToI  idx_to_i(domain, sdom_id, (IZoneI*)&extent.idx_to_i[0]);
  typename POL::View_IdxToJ  idx_to_j(domain, sdom_id, (IZoneJ*)&extent.idx_to_j[0]);
  typename POL::View_IdxToK  idx_to_k(domain, sdom_id, (IZoneK*)&extent.idx_to_k[0]);

  RAJA::forallN<SweepPolicy<nest_type>, IDirection, IGroup, IZoneIdx>(
    domain.indexRange<IDirection>(sdom_id),
    domain.indexRange<IGroup>(sdom_id),
    extent.indexset_sweep,
    RAJA_LAMBDA (IDirection d, IGroup g, IZoneIdx zone_idx){

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
    }); // forall3

}

} // anon namespace

void Kernel::sweep(Grid_Data *domain, int sdom_id) {
  switch(nesting_order){
    case NEST_DGZ: kernel_sweep<NEST_DGZ_T>(*domain, sdom_id); break;
    case NEST_DZG: kernel_sweep<NEST_DZG_T>(*domain, sdom_id); break;
    case NEST_GDZ: kernel_sweep<NEST_GDZ_T>(*domain, sdom_id); break;
    case NEST_GZD: kernel_sweep<NEST_GZD_T>(*domain, sdom_id); break;
    case NEST_ZDG: kernel_sweep<NEST_ZDG_T>(*domain, sdom_id); break;
    case NEST_ZGD: kernel_sweep<NEST_ZGD_T>(*domain, sdom_id); break;
  }
}
