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
#include<RAJA/View.hxx>
#include<RAJA/Forall.hxx>

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

  MPI_Abort(MPI_COMM_WORLD, 1);
  return NULL;
}


Kernel::Kernel(Nesting_Order nest) :
  nesting_order(nest)
{}

Kernel::~Kernel(){
}



#include<Kripke/Kernel/LTimesPolicy.h>
void Kernel::LTimes(Grid_Data *domain) {

  BEGIN_POLICY(nesting_order, nest_type)
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
      typename POL::View_Psi psi(sdom.psi->ptr(), domain, sdom_id);
      typename POL::View_Phi phi(sdom.phi->ptr(), domain, sdom_id);
      typename POL::View_Ell ell(sdom.ell->ptr(), domain, sdom_id);

      dForall4<LTimesPolicy<nest_type>, IMoment, IDirection, IGroup, IZone>(
        domain, sdom_id, 
        RAJA_LAMBDA (auto nm, auto d, auto g, auto z){
  
          IGlobalGroup g_global( (*g) + group0);
          
          phi(nm, g_global, z) += ell(d, nm) * psi(d, g, z);
        });

    END_FORALL
  END_POLICY
}

#include<Kripke/Kernel/LPlusTimesPolicy.h>
void Kernel::LPlusTimes(Grid_Data *domain) {

  BEGIN_POLICY(nesting_order, nest_type)
    typedef DataPolicy<nest_type> POL;

    // Loop over Subdomains
    FORALL_SUBDOMAINS(seq_pol, domain, sdom_id, sdom)
      sdom.rhs->clear(0.0);
    END_FORALL

    FORALL_SUBDOMAINS(seq_pol, domain, sdom_id, sdom)

      // Get dimensioning
      int group0 = sdom.group0;

      // Get pointers
      typename POL::View_Psi     rhs(sdom.rhs->ptr(), domain, sdom_id);
      typename POL::View_Phi     phi_out(sdom.phi_out->ptr(), domain, sdom_id);
      typename POL::View_EllPlus ell_plus(sdom.ell_plus->ptr(), domain, sdom_id);
      
      dForall4<LPlusTimesPolicy<nest_type>, IMoment, IDirection, IGroup, IZone>(
        domain, sdom_id, 
        RAJA_LAMBDA (auto nm, auto d, auto g, auto z){
  
          IGlobalGroup g_global( (*g) + group0);
  
          rhs(d, g, z) += ell_plus(d, nm) * phi_out(nm, g_global, z);  
        });

    END_FORALL
  END_POLICY
}


/**
  Compute scattering source term phi_out from flux moments in phi.
  phi_out(gp,z,nm) = sum_g { sigs(g, n, gp) * phi(g,z,nm) }
*/
#include<Kripke/Kernel/ScatteringPolicy.h>
void Kernel::scattering(Grid_Data *domain){
  
  BEGIN_POLICY(nesting_order, nest_type)
    typedef DataPolicy<nest_type> POL;

    // Zero out source terms
    FORALL_ZONESETS(seq_pol, domain, sdom_id, sdom)
      sdom.phi_out->clear(0.0);
    END_FORALL

    // Loop over zoneset subdomains
    FORALL_ZONESETS(seq_pol, domain, sdom_id, sdom)

      typename POL::View_Phi     phi(sdom.phi->ptr(), domain, sdom_id);
      typename POL::View_Phi     phi_out(sdom.phi_out->ptr(), domain, sdom_id);
      typename POL::View_SigS    sigs(domain->sigs->ptr(), domain, sdom_id);

      typename POL::View_MixedToZones mixed_to_zones((IZone*)&sdom.mixed_to_zones[0], domain, sdom_id);
      typename POL::View_MixedToMaterial mixed_material((IMaterial*)&sdom.mixed_material[0], domain, sdom_id);
      typename POL::View_MixedToFraction mixed_fraction(&sdom.mixed_fraction[0], domain, sdom_id);
      typename POL::View_MomentToCoeff moment_to_coeff((ILegendre*)&domain->moment_to_coeff[0], domain, sdom_id);
      
      dForall4<ScatteringPolicy<nest_type>, IMoment, IGlobalGroup, IGlobalGroup, IMix>(
        domain, sdom_id,
        RAJA_LAMBDA (auto nm, auto g, auto gp, auto mix){
        
          ILegendre n = moment_to_coeff(nm);
          IZone zone = mixed_to_zones(mix);
          IMaterial material = mixed_material(mix);
          double fraction = mixed_fraction(mix);

          phi_out(nm, gp, zone) += 
            sigs(n, g, gp, material) * phi(nm, g, zone) * fraction;                     
                             
        });  // forall
          
    END_FORALL // zonesets

  END_POLICY
}

  
/**
 * Add an isotropic source, with flux of 1, to every zone with Region 1
 * (or material 0).
 *
 * Since it's isotropic, we're just adding this to nm=0.
 */
#include<Kripke/Kernel/SourcePolicy.h>
void Kernel::source(Grid_Data *domain){

  BEGIN_POLICY(nesting_order, nest_type)
    typedef DataPolicy<nest_type> POL;

    // Loop over zoneset subdomains
    FORALL_ZONESETS(seq_pol, domain, sdom_id, sdom)
      typename POL::View_Phi     phi_out(sdom.phi_out->ptr(), domain, sdom_id);
      typename POL::View_MixedToZones mixed_to_zones((IZone*)&sdom.mixed_to_zones[0], domain, sdom_id);
      typename POL::View_MixedToMaterial mixed_material((IMaterial*)&sdom.mixed_material[0], domain, sdom_id);
      typename POL::View_MixedToFraction mixed_fraction(&sdom.mixed_fraction[0], domain, sdom_id);

      dForall2<SourcePolicy<nest_type>, IGlobalGroup, IMix >(
        domain, sdom_id,
        RAJA_LAMBDA (auto g, auto mix){
          IZone zone = mixed_to_zones(mix);
          IMaterial material = mixed_material(mix);
          double fraction = mixed_fraction(mix);

          if(*material == 0){
            phi_out(IMoment(0), g, zone) += 1.0 * fraction;
          }
      }); // forall

    END_FORALL
  END_POLICY
}


#include<Kripke/Kernel/SweepPolicy.h>
void Kernel::sweep(Grid_Data *domain, int sdom_id) {
  BEGIN_POLICY(nesting_order, nest_type)
    typedef DataPolicy<nest_type> POL;

    Subdomain *sdom = &domain->subdomains[sdom_id];

    typename POL::View_Directions direction(sdom->directions, domain, sdom_id);

    typename POL::View_Psi     rhs(sdom->rhs->ptr(), domain, sdom_id);
    typename POL::View_Psi     psi(sdom->psi->ptr(), domain, sdom_id);
    typename POL::View_SigT    sigt(sdom->sigt->ptr(), domain, sdom_id);

    typename POL::View_dx      dx(&sdom->deltas[0][0], domain, sdom_id);
    typename POL::View_dy      dy(&sdom->deltas[1][0], domain, sdom_id);
    typename POL::View_dz      dz(&sdom->deltas[2][0], domain, sdom_id);
    
    typename POL::TLayout_Zone zone_layout(domain, sdom_id);
    
    typename POL::View_FaceI face_lf(sdom->plane_data[0]->ptr(), domain, sdom_id);
    typename POL::View_FaceJ face_fr(sdom->plane_data[1]->ptr(), domain, sdom_id);
    typename POL::View_FaceK face_bo(sdom->plane_data[2]->ptr(), domain, sdom_id);

    // All directions have same id,jd,kd, since these are all one Direction Set
    // So pull that information out now
    Grid_Sweep_Block const &extent = sdom->sweep_block;    
    typename POL::View_IdxToI  idx_to_i((IZoneI*)&extent.idx_to_i[0], domain, sdom_id);
    typename POL::View_IdxToJ  idx_to_j((IZoneJ*)&extent.idx_to_j[0], domain, sdom_id);
    typename POL::View_IdxToK  idx_to_k((IZoneK*)&extent.idx_to_k[0], domain, sdom_id);

    RAJA::forall3<SweepPolicy<nest_type>, IDirection, IGroup, IZoneIdx>(
      domain->indexRange<IDirection>(sdom_id),
      domain->indexRange<IGroup>(sdom_id),
      extent.indexset_sweep,
      RAJA_LAMBDA (auto d, auto g, auto zone_idx){
        
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
  END_POLICY
}

