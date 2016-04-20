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

#include<Kripke/Kernel/Kernel_3d_ZGD.h>
#include<Kripke/Grid.h>
#include<Kripke/SubTVec.h>

Kernel_3d_ZGD::~Kernel_3d_ZGD() {}

Nesting_Order Kernel_3d_ZGD::nestingPsi(void) const {
  return NEST_ZGD;
}

Nesting_Order Kernel_3d_ZGD::nestingPhi(void) const {
  return NEST_ZGD;
}

Nesting_Order Kernel_3d_ZGD::nestingSigt(void) const {
  return NEST_DZG;
}

Nesting_Order Kernel_3d_ZGD::nestingEll(void) const {
  return NEST_ZGD;
}

Nesting_Order Kernel_3d_ZGD::nestingEllPlus(void) const {
  return NEST_ZDG;
}

Nesting_Order Kernel_3d_ZGD::nestingSigs(void) const {
  return NEST_ZGD;
}

void Kernel_3d_ZGD::LTimes(Grid_Data *grid_data) {
  // Outer parameters
  int num_moments = grid_data->total_num_moments;

  // Clear phi
  for(int ds = 0;ds < grid_data->num_zone_sets;++ ds){
    grid_data->phi[ds]->clear(0.0);
  }

  // Loop over Subdomains
  int num_subdomains = grid_data->subdomains.size();
  for (int sdom_id = 0; sdom_id < num_subdomains; ++ sdom_id){
    Subdomain &sdom = grid_data->subdomains[sdom_id];

    // Get dimensioning
    int num_groups = sdom.phi->groups;
    int num_zones = sdom.num_zones;
    int num_local_groups = sdom.num_groups;
    int group0 = sdom.group0;
    int num_local_directions = sdom.num_directions;
    int num_gnm = num_groups * num_moments;
    int num_locgd = num_local_groups * num_local_directions;

    // Get pointers
    double const * KRESTRICT ell = sdom.ell->ptr();
    double const * KRESTRICT psi = sdom.psi->ptr();
    double       * KRESTRICT phi = sdom.phi->ptr();

#ifdef KRIPKE_USE_OPENMP
#pragma omp parallel for
#endif
    for (int z = 0; z < num_zones; z++) {
      double const * KRESTRICT psi_z = psi + z*num_locgd;
      double       * KRESTRICT phi_z = phi + z*num_gnm;

      for (int g = 0; g < num_local_groups; ++g) {
        double const * KRESTRICT psi_z_g = psi_z + g*num_local_directions;
        double       * KRESTRICT phi_z_g = phi_z + (group0+g)*num_moments;

        for(int nm = 0;nm < num_moments;++nm){
          double const * KRESTRICT ell_nm = ell + nm*num_local_directions;

          double phi_z_g_nm = 0.0;
          for (int d = 0; d < num_local_directions; d++) {
            phi_z_g_nm += ell_nm[d] * psi_z_g[d];
          }
          phi_z_g[nm] += phi_z_g_nm;
        }
      }
    }
  }
}

void Kernel_3d_ZGD::LPlusTimes(Grid_Data *grid_data) {
  // Outer parameters
  int num_moments = grid_data->total_num_moments;

  // Loop over Subdomains
  int num_subdomains = grid_data->subdomains.size();
  for (int sdom_id = 0; sdom_id < num_subdomains; ++ sdom_id){
    Subdomain &sdom = grid_data->subdomains[sdom_id];

    // Get dimensioning
    int num_zones = sdom.num_zones;
    int num_groups = sdom.phi->groups;
    int num_local_groups = sdom.num_groups;
    int group0 = sdom.group0;
    int num_local_directions = sdom.num_directions;
    int num_gnm = num_moments*num_groups;
    int num_locgd = num_local_directions*num_local_groups;

    // Zero RHS
    sdom.rhs->clear(0.0);
    
    // Get pointers
    double const * KRESTRICT phi_out = sdom.phi_out->ptr();
    double const * KRESTRICT ell_plus = sdom.ell_plus->ptr();
    double       * KRESTRICT rhs = sdom.rhs->ptr();

#ifdef KRIPKE_USE_OPENMP
#pragma omp parallel for
#endif
    for(int z = 0;z < num_zones; ++ z){    
      double const * KRESTRICT phi_out_z = phi_out + z*num_gnm;
      double       * KRESTRICT rhs_z = rhs + z*num_locgd;
      
      for (int g = 0; g < num_local_groups; ++g) {
        double const * KRESTRICT phi_out_z_g = phi_out_z + (group0+g)*num_moments;
        double       * KRESTRICT rhs_z_g = rhs_z + g*num_local_directions;
      
        for (int d = 0; d < num_local_directions; d++) {
          double const * KRESTRICT ell_plus_d = ell_plus + d*num_moments;
                              
          double rhs_z_g_d = 0.0;
          for(int nm = 0;nm < num_moments; ++nm){
            rhs_z_g_d += ell_plus_d[nm] * phi_out_z_g[nm];
          }                    
          rhs_z_g[d] = rhs_z_g_d;
        }        
      }     
    }
  }
}


/**
  Compute scattering source term phi_out from flux moments in phi.
  phi_out(gp,z,nm) = sum_g { sigs(g, n, gp) * phi(g,z,nm) }
*/
void Kernel_3d_ZGD::scattering(Grid_Data *grid_data){
  // Loop over zoneset subdomains
  for(int zs = 0;zs < grid_data->num_zone_sets;++ zs){
    // get material mix information
    int sdom_id = grid_data->zs_to_sdomid[zs];
    Subdomain &sdom = grid_data->subdomains[sdom_id];
    int    const * KRESTRICT zones_to_mixed = &sdom.zones_to_mixed[0];
    int    const * KRESTRICT num_mixed = &sdom.num_mixed[0];
    int    const * KRESTRICT mixed_material = &sdom.mixed_material[0];
    double const * KRESTRICT mixed_fraction = &sdom.mixed_fraction[0];
    double const * KRESTRICT sigs = grid_data->sigs->ptr();

    int    const * KRESTRICT moment_to_coeff = &grid_data->moment_to_coeff[0];
    double const * KRESTRICT phi = grid_data->phi[zs]->ptr();
    double       * KRESTRICT phi_out = grid_data->phi_out[zs]->ptr();

    // Zero out source terms
    grid_data->phi_out[zs]->clear(0.0);

    // grab dimensions
    int num_zones = sdom.num_zones;
    int num_groups = grid_data->phi_out[zs]->groups;
    int num_moments = grid_data->total_num_moments;
    int num_coeff = grid_data->legendre_order+1;
    int num_nmg = num_moments*num_groups;

#ifdef KRIPKE_USE_OPENMP
#pragma omp parallel for
#endif
    for(int zone = 0;zone < num_zones;++ zone){
      int mix_start = zones_to_mixed[zone];
      int mix_stop = mix_start + num_mixed[zone];

      for(int mix = mix_start;mix < mix_stop;++ mix){
        int material = mixed_material[mix];
        double fraction = mixed_fraction[mix];
        
        double const * KRESTRICT sigs_mat = sigs + material*num_coeff*num_groups*num_groups;
        double const * KRESTRICT phi_z = phi + zone*num_nmg;
        double       * KRESTRICT phi_out_z = phi_out + zone*num_nmg;
        
        for(int g = 0;g < num_groups;++ g){      
          double const * KRESTRICT sigs_mat_g = sigs_mat + g*num_groups*num_coeff;
          double const * KRESTRICT phi_z_g = phi_z + g*num_moments;
                            
          for(int gp = 0;gp < num_groups;++ gp){
            double const * KRESTRICT sigs_mat_g_gp = sigs_mat_g + gp*num_coeff;
            double       * KRESTRICT phi_out_z_gp = phi_out_z + gp*num_moments;
          
            for(int nm = 0;nm < num_moments;++ nm){
              // map nm to n
              int n = moment_to_coeff[nm];
              
              phi_out_z_gp[nm] += sigs_mat_g_gp[n] * phi_z_g[nm] * fraction;
            }
          }        
        }
      }
    }
  }
}

/**
 * Add an isotropic source, with flux of 1, to every zone with Region 1
 * (or material 0).
 *
 * Since it's isotropic, we're just adding this to nm=0.
 */
void Kernel_3d_ZGD::source(Grid_Data *grid_data){
  // Loop over zoneset subdomains
  for(int zs = 0;zs < grid_data->num_zone_sets;++ zs){
  
    // get material mix information
    int sdom_id = grid_data->zs_to_sdomid[zs];
    Subdomain &sdom = grid_data->subdomains[sdom_id];
    int    const * KRESTRICT zones_to_mixed = &sdom.zones_to_mixed[0];
    int    const * KRESTRICT num_mixed = &sdom.num_mixed[0];
    int    const * KRESTRICT mixed_material = &sdom.mixed_material[0];
    double const * KRESTRICT mixed_fraction = &sdom.mixed_fraction[0];
    double       * KRESTRICT phi_out = grid_data->phi_out[zs]->ptr();

    // grab dimensions
    int num_zones = sdom.num_zones;
    int num_groups = grid_data->phi_out[zs]->groups;
    int num_moments = grid_data->total_num_moments;

#ifdef KRIPKE_USE_OPENMP
#pragma omp parallel for
#endif
    for(int zone = 0;zone < num_zones;++ zone){
      int mix_start = zones_to_mixed[zone];
      int mix_stop = mix_start + num_mixed[zone];

      for(int mix = mix_start;mix < mix_stop;++ mix){
        int material = mixed_material[mix];
        double fraction = mixed_fraction[mix];
        double * KRESTRICT phi_out_z = phi_out + zone*num_moments*num_groups;

        if(material == 0){        
          for(int g = 0;g < num_groups;++ g){
            phi_out_z[g*num_moments] += 1.0 * fraction;
          }
        }
      }
    }
  }
}

// Macros for offsets with fluxes on cell faces 
#define I_PLANE_INDEX(j, k) ((k)*(local_jmax) + (j))
#define J_PLANE_INDEX(i, k) ((k)*(local_imax) + (i))
#define K_PLANE_INDEX(i, j) ((j)*(local_imax) + (i))
#define Zonal_INDEX(i, j, k) ((i) + (local_imax)*(j) \
  + (local_imax)*(local_jmax)*(k))

void Kernel_3d_ZGD::sweep(Subdomain *sdom) {
  int num_directions = sdom->num_directions;
  int num_groups = sdom->num_groups;

  Directions *direction = sdom->directions;

  int local_imax = sdom->nzones[0];
  int local_jmax = sdom->nzones[1];

  double const * KRESTRICT dx = &sdom->deltas[0][0];
  double const * KRESTRICT dy = &sdom->deltas[1][0];
  double const * KRESTRICT dz = &sdom->deltas[2][0];
  
  double const * KRESTRICT sigt = sdom->sigt->ptr();
  double       * KRESTRICT psi  = sdom->psi->ptr();
  double const * KRESTRICT rhs  = sdom->rhs->ptr();

  double * KRESTRICT psi_lf = sdom->plane_data[0]->ptr();
  double * KRESTRICT psi_fr = sdom->plane_data[1]->ptr();
  double * KRESTRICT psi_bo = sdom->plane_data[2]->ptr();

  int num_gd = num_groups * num_directions;

  // All directions have same id,jd,kd, since these are all one Direction Set
  // So pull that information out now
  Grid_Sweep_Block const &extent = sdom->sweep_block;

  //  Perform transport sweep of the grid 1 cell at a time.
  for (int k = extent.start_k; k != extent.end_k; k += extent.inc_k) {
    double two_dz = 2.0 / dz[k + 1];
    for (int j = extent.start_j; j != extent.end_j; j += extent.inc_j) {
      double two_dy = 2.0 / dy[j + 1];
      for (int i = extent.start_i; i != extent.end_i; i += extent.inc_i) {
        double two_dx = 2.0 / dx[i + 1];

        int z = Zonal_INDEX(i, j, k);
        double const * KRESTRICT sigt_z = sigt + z*num_groups;
        double       * KRESTRICT psi_z  = psi  + z*num_gd;
        double const * KRESTRICT rhs_z  = rhs  + z*num_gd;

        double       * KRESTRICT psi_lf_z = psi_lf + I_PLANE_INDEX(j, k) * num_gd;
        double       * KRESTRICT psi_fr_z = psi_fr + J_PLANE_INDEX(i, k) * num_gd;
        double       * KRESTRICT psi_bo_z = psi_bo + K_PLANE_INDEX(i, j) * num_gd;
        
#ifdef KRIPKE_USE_OPENMP
#pragma omp parallel for
#endif
        for (int g = 0; g < num_groups; ++g) {
          double       * KRESTRICT psi_z_g = psi_z + g * num_directions;
          double const * KRESTRICT rhs_z_g = rhs_z + g * num_directions;

          double       * KRESTRICT psi_lf_z_g = psi_lf_z + g * num_directions;
          double       * KRESTRICT psi_fr_z_g = psi_fr_z + g * num_directions;
          double       * KRESTRICT psi_bo_z_g = psi_bo_z + g * num_directions;

          for (int d = 0; d < num_directions; ++d) {
            double xcos_dxi = direction[d].xcos * two_dx;
            double ycos_dyj = direction[d].ycos * two_dy;
            double zcos_dzk = direction[d].zcos * two_dz;

            // Calculate new zonal flux 
            double psi_z_g_d = (rhs_z_g[d]
                + psi_lf_z_g[d] * xcos_dxi
                + psi_fr_z_g[d] * ycos_dyj
                + psi_bo_z_g[d] * zcos_dzk)
                / (xcos_dxi + ycos_dyj + zcos_dzk + sigt_z[g]);

            psi_z_g[d] = psi_z_g_d;

            // Apply diamond-difference relationships 
            psi_lf_z_g[d] = 2.0 * psi_z_g_d - psi_lf_z_g[d];
            psi_fr_z_g[d] = 2.0 * psi_z_g_d - psi_fr_z_g[d];
            psi_bo_z_g[d] = 2.0 * psi_z_g_d - psi_bo_z_g[d];
          }
        }
      }
    }
  }
}

