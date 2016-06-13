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

#include<Kripke/Kernel/LTimesPolicy.h>
#include<Kripke/Kernel/LPlusTimesPolicy.h>
#include<Kripke/Kernel/ScatteringPolicy.h>
#include<Kripke/Kernel/SourcePolicy.h>
#include<Kripke/Kernel/SweepPolicy.h>
#include<Kripke/Kernel/ParticleEditPolicy.h>

// For now, if CUDA is being used, then we are using functors for the kernels 
// instead of lambdas.  CUDA8's __host__ __device__ lambdas might fix this
// restriction
#ifdef RAJA_ENABLE_CUDA
#define KRIPKE_USE_FUNCTORS
#else

// Uncomment the next line to force the use of functors
//#define KRIPKE_USE_FUNCTORS

#endif


#ifdef KRIPKE_USE_FUNCTORS
#include<Kripke/KernelFunctors.h>
#endif

/*
 This function provides a mapping from the runtime Nesting_Order variable to a
 compile-time type (ie NEST_DZG_T, etc.), passing that type as the first 
 argument to the callable "kernel".
*/
template<typename KERNEL, typename ... ARGS>
RAJA_INLINE
void callKernelWithPolicy(Nesting_Order nesting_order, KERNEL kernel, ARGS & ... args){
  switch(nesting_order){
    case NEST_DGZ: kernel(NEST_DGZ_T(), args...); break;
#ifndef RAJA_COMPILER_ICC
    case NEST_DZG: kernel(NEST_DZG_T(), args...); break;
    case NEST_GDZ: kernel(NEST_GDZ_T(), args...); break;
    case NEST_GZD: kernel(NEST_GZD_T(), args...); break;
    case NEST_ZDG: kernel(NEST_ZDG_T(), args...); break;
    case NEST_ZGD: kernel(NEST_ZGD_T(), args...); break;
#else
    default: KripkeAbort("All nesting orders except DGZ are currently disabled with the Intel compilers\n");
#endif
  }
}


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



struct Kernel_LTimes{

  template<typename nest_type>
  RAJA_INLINE
  void operator()(nest_type, Grid_Data &domain) const {

    typedef DataPolicy<nest_type> POL;
    
    using PHI = typename POL::View_Phi;
    using PSI = typename POL::View_Psi;
    using ELL = typename POL::View_Ell; 
    
    // Zero Phi
    FORALL_ZONESETS(seq_pol, domain, sdom_id, sdom)
      sdom.phi->clear(0.0);
    END_FORALL

    // Loop over Subdomains
    FORALL_SUBDOMAINS(seq_pol, domain, sdom_id, sdom)

      // Get dimensioning
      int group0 = sdom.group0;

      // Get pointers
      PSI psi(domain, sdom_id, sdom.psi->ptr());
      PHI phi(domain, sdom_id, sdom.phi->ptr());
      ELL ell(domain, sdom_id, sdom.ell->ptr());

#ifdef KRIPKE_USE_FUNCTORS
      dForallN<LTimesPolicy<nest_type>, IMoment, IDirection, IGroup, IZone>(
        domain, sdom_id, 
        LTimesFcn<PHI, ELL, PSI>(phi, ell, psi, group0)
      );
#else
      dForallN<LTimesPolicy<nest_type>, IMoment, IDirection, IGroup, IZone>(
        domain, sdom_id,
        RAJA_LAMBDA (IMoment nm, IDirection d, IGroup g, IZone z){

          IGlobalGroup g_global( (*g) + group0);

          phi(nm, g_global, z) += ell(d, nm) * psi(d, g, z);
        }
      );
#endif

    END_FORALL
  }
};


void Kernel::LTimes(Grid_Data *domain) {
  callKernelWithPolicy(nesting_order, Kernel_LTimes(), *domain);
}



struct Kernel_LPlusTimes {
  template<typename nest_type>
  RAJA_INLINE
  void operator()(nest_type, Grid_Data &domain) const {

    typedef DataPolicy<nest_type> POL;

    using PHI      = typename POL::View_Phi;
    using PSI      = typename POL::View_Psi;
    using ELL_PLUS = typename POL::View_EllPlus;

    // Zero Phi
    FORALL_SUBDOMAINS(seq_pol, domain, sdom_id, sdom)
      sdom.rhs->clear(0.0);
    END_FORALL

    // Loop over Subdomains
    FORALL_SUBDOMAINS(seq_pol, domain, sdom_id, sdom)

      // Get dimensioning
      int group0 = sdom.group0;

      // Get pointers
      PSI      rhs     (domain, sdom_id, sdom.rhs->ptr());
      PHI      phi_out (domain, sdom_id, sdom.phi_out->ptr());
      ELL_PLUS ell_plus(domain, sdom_id, sdom.ell_plus->ptr());

#ifdef KRIPKE_USE_FUNCTORS
      dForallN<LPlusTimesPolicy<nest_type>, IMoment, IDirection, IGroup, IZone>(
        domain, sdom_id,
        LPlusTimesFcn<PSI, ELL_PLUS, PHI>(rhs, ell_plus, phi_out, group0)
      );
#else
      dForallN<LPlusTimesPolicy<nest_type>, IMoment, IDirection, IGroup, IZone>(
        domain, sdom_id,
        RAJA_LAMBDA (IMoment nm, IDirection d, IGroup g, IZone z){

          IGlobalGroup g_global( (*g) + group0);

          rhs(d, g, z) += ell_plus(d, nm) * phi_out(nm, g_global, z);
        }
      );
#endif

    END_FORALL
  }
};

void Kernel::LPlusTimes(Grid_Data *domain) {
  callKernelWithPolicy(nesting_order, Kernel_LPlusTimes(), *domain);
}







/**
  Compute scattering source term phi_out from flux moments in phi.
  phi_out(gp,z,nm) = sum_g { sigs(g, n, gp) * phi(g,z,nm) }
*/
struct Kernel_Scattering{
  template<typename nest_type>
  RAJA_INLINE
  void operator()(nest_type, Grid_Data &domain) const {
    
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

#ifdef KRIPKE_USE_FUNCTORS
      dForallN<ScatteringPolicy<nest_type>, IMoment, IGlobalGroup, IGlobalGroup, IMix>(
        domain, sdom_id,
        ScatteringFcn<typename POL::View_Phi,
                      typename POL::View_SigS,
                      typename POL::View_MixedToZones,
                      typename POL::View_MixedToMaterial,
                      typename POL::View_MixedToFraction,
                      typename POL::View_MomentToCoeff>
                    (phi, phi_out, sigs, mixed_to_zones, mixed_material, mixed_fraction, moment_to_coeff)
      );
#else
      dForallN<ScatteringPolicy<nest_type>, IMoment, IGlobalGroup, IGlobalGroup, IMix>(
        domain, sdom_id,
        RAJA_LAMBDA (IMoment nm, IGlobalGroup g, IGlobalGroup gp, IMix mix){
        
          ILegendre n = moment_to_coeff(nm);
          IZone zone = mixed_to_zones(mix);
          IMaterial material = mixed_material(mix);
          double fraction = mixed_fraction(mix);

          phi_out(nm, gp, zone) +=
            sigs(n, g, gp, material) * phi(nm, g, zone) * fraction;

        });  
#endif
    END_FORALL // zonesets
  }

};

void Kernel::scattering(Grid_Data *domain) {
  callKernelWithPolicy(nesting_order, Kernel_Scattering(), *domain);
}



  
/**
 * Add an isotropic source, with flux of 1, to every zone with Region 1
 * (or material 0).
 *
 * Since it's isotropic, we're just adding this to nm=0.
 */

struct Kernel_Source {
  template<typename nest_type>
  RAJA_INLINE
  void operator()(nest_type, Grid_Data &domain) const {
    typedef DataPolicy<nest_type> POL;

    // Loop over zoneset subdomains
    FORALL_ZONESETS(seq_pol, domain, sdom_id, sdom)
      typename POL::View_Phi             phi_out       (domain, sdom_id, sdom.phi_out->ptr());
      typename POL::View_MixedToZones    mixed_to_zones(domain, sdom_id, (IZone*)&sdom.mixed_to_zones[0]);
      typename POL::View_MixedToMaterial mixed_material(domain, sdom_id, (IMaterial*)&sdom.mixed_material[0]);
      typename POL::View_MixedToFraction mixed_fraction(domain, sdom_id, &sdom.mixed_fraction[0]);

#ifdef KRIPKE_USE_FUNCTORS
      dForallN<SourcePolicy<nest_type>, IGlobalGroup, IMix>(
        domain, sdom_id,
        SourceFcn<typename POL::View_Phi,
                  typename POL::View_MixedToZones,
                  typename POL::View_MixedToMaterial,
                  typename POL::View_MixedToFraction>
                (phi_out, mixed_to_zones, mixed_material, mixed_fraction)
      );
      

#else
      dForallN<SourcePolicy<nest_type>, IGlobalGroup, IMix>(
        domain, sdom_id,
        RAJA_LAMBDA (IGlobalGroup g, IMix mix){
          IZone zone = mixed_to_zones(mix);
          IMaterial material = mixed_material(mix);
          double fraction = mixed_fraction(mix);

          if(*material == 0){
            phi_out(IMoment(0), g, zone) += 1.0 * fraction;
          }
      }); 
#endif
    END_FORALL
  }
};

void Kernel::source(Grid_Data *domain) {
  callKernelWithPolicy(nesting_order, Kernel_Source(), *domain);
}




struct Kernel_Sweep{

  template<typename nest_type>
  RAJA_INLINE
  void operator()(nest_type, Grid_Data &domain, int sdom_id) const {
    
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

#ifdef KRIPKE_USE_FUNCTORS
    RAJA::forallN<SweepPolicy<nest_type>, IDirection, IGroup, IZoneIdx>( 
      domain.indexRange<IDirection>(sdom_id),
      domain.indexRange<IGroup>(sdom_id),
      extent.indexset_sweep,
      SweepFcn<typename POL::View_Directions,
               typename POL::View_Psi,
               typename POL::View_SigT,
               typename POL::View_dx,
               typename POL::View_dy,
               typename POL::View_dz,
               typename POL::TLayout_Zone,
               typename POL::View_FaceI,
               typename POL::View_FaceJ,
               typename POL::View_FaceK,
               typename POL::View_IdxToI,
               typename POL::View_IdxToJ,
               typename POL::View_IdxToK>
               (direction, rhs, psi, sigt, dx, dy, dz, zone_layout, 
                face_lf, face_fr, face_bo, idx_to_i, idx_to_j, idx_to_k)
    );
#else

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
      }); 
#endif
  }
};


void Kernel::sweep(Grid_Data *domain, int sdom_id) {
  callKernelWithPolicy(nesting_order, Kernel_Sweep(), *domain, sdom_id);
}

/**
 *  Edit that sums up number of particles on mesh.
 */

struct Kernel_ParticleEdit {

  double &part;

  Kernel_ParticleEdit(double &t) : part(t) {}

  template<typename nest_type>
  RAJA_INLINE
  void operator()(nest_type, Grid_Data &domain) const {
    typedef DataPolicy<nest_type> POL;
 
    RAJA::ReduceSum<typename POL::reduce_policy, double> part_reduce(0.0);
       
    // Loop over zoneset subdomains
    FORALL_SUBDOMAINS(seq_pol, domain, sdom_id, sdom)
      typename POL::View_Psi         psi      (domain, sdom_id, sdom.psi->ptr());
      typename POL::View_Directions  direction(domain, sdom_id, sdom.directions);
      typename POL::View_Volume      volume   (domain, sdom_id, &sdom.volume[0]);

      
#ifdef KRIPKE_USE_FUNCTORS
      dForallN<ParticleEditPolicy<nest_type>, IDirection, IGroup, IZone>( 
        domain, sdom_id,
        ParticleEditFcn<decltype(part_reduce),
                  typename POL::View_Directions,
                  typename POL::View_Psi,
                  typename POL::View_Volume>
                (part_reduce, direction, psi, volume)
      );
#else
      dForallN<ParticleEditPolicy<nest_type>, IDirection, IGroup, IZone>( 
        domain, sdom_id,
        RAJA_LAMBDA (IDirection d, IGroup g, IZone z){
          part_reduce += direction(d).w * psi(d,g,z) * volume(z);              
        }
      ); 
#endif
    END_FORALL
    
    part = part_reduce;
    
    // reduce across MPI
#ifdef KRIPKE_USE_MPI
    double part_global;

    MPI_Reduce(&part, &part_global, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    part = part_global;
#endif
  }
};

double Kernel::particleEdit(Grid_Data *domain) {
  double total = 0.0;
  callKernelWithPolicy(nesting_order, Kernel_ParticleEdit(total), *domain);
  return total;
}
