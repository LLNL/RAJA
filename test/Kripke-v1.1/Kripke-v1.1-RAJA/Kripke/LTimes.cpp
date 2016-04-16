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
#include<Kripke/Kernel/DataPolicy.h>


// Uncomment to use Lambdas instead of Functors for each kernel
//#define KRIPKE_USE_FUNCTORS




#include<Kripke/Kernel/LTimesPolicy.h>

namespace {

#ifdef KRIPKE_USE_FUNCTORS
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
#endif


template<typename nest_type>
RAJA_INLINE
void kernel_LTimes(Grid_Data &domain) {

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

} // anon namespace

void Kernel::LTimes(Grid_Data *domain) {
  switch(nesting_order){
    case NEST_DGZ: kernel_LTimes<NEST_DGZ_T>(*domain); break;
    case NEST_DZG: kernel_LTimes<NEST_DZG_T>(*domain); break;
   /* case NEST_GDZ: kernel_LTimes<NEST_GDZ_T>(*domain); break;
    case NEST_GZD: kernel_LTimes<NEST_GZD_T>(*domain); break;
    case NEST_ZDG: kernel_LTimes<NEST_ZDG_T>(*domain); break;
    case NEST_ZGD: kernel_LTimes<NEST_ZGD_T>(*domain); break;
  */ }
}


