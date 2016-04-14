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

#include <Kripke/Grid.h>
#include <Kripke/SubTVec.h>
#include <Kripke/Input_Variables.h>
#include <Kripke/Kernel/DataPolicy.h>

#include <cmath>
#include <sstream>


namespace {
  /**
    This function defined the material distribution in space.
    This defines Problem 3 from Kobayashi
    Where Region 1 is material 0, 2 is 1 and 3 is 2.
  */
  inline int queryMaterial(double x, double y, double z){
    // Problem is defined for one octant, with reflecting boundaries
    // We "unreflect" it here by taking abs values
    x = std::abs(x);
    y = std::abs(y);
    z = std::abs(z);

    // Central 20x20x20 box is Region 1
    if(x <= 10.0 && y <= 10.0 && z <= 10.0){
      return 0;
    }

    // Leg 1 of Region 2
    if(x <= 10.0 && y <= 60.0 && z <= 10.0){
      return 1;
    }

    // Leg 2 of Region 2
    if(x <= 40.0 && y >= 50.0 && y <= 60.0 && z <= 10.0){
      return 1;
    }

    // Leg 3 of Region 2
    if(x >= 30.0 && x <= 40.0 && y >= 50.0 && y <= 60.0 && z <= 40.0){
      return 1;
    }

    // Leg 4 of Region 2
    if(x >= 30.0 && x <= 40.0 && y >= 50.0 && z >= 30.0 && z <= 40.0){
      return 1;
    }

    // Rest is filled with region 3
    return 2;
  }
}



Subdomain::Subdomain() :
  idx_group_set(0),
  idx_dir_set(0),
  idx_zone_set(0),
  num_groups(0),
  num_directions(0),
  num_zones(0),
  group0(0),
  direction0(0),
  psi(NULL),
  rhs(NULL),
  sigt(NULL),
  directions(NULL),
  ell(NULL),
  ell_plus(NULL),
  phi(NULL),
  phi_out(NULL)
{
  for(int dim = 0;dim < 3;++ dim){
    plane_data[dim] = NULL;
    old_plane_data[dim] = NULL;
  }
}
Subdomain::~Subdomain(){
  delete psi;
  delete rhs;
  delete sigt;
  for(int dim = 0;dim < 3;++ dim){
    delete plane_data[dim];
    delete old_plane_data[dim];
  }
}


/**
  Setup subdomain and allocate data
*/
void Subdomain::setup(int sdom_id, Input_Variables *input_vars, int gs, int ds, int zs,
    std::vector<Directions> &direction_list, Kernel *kernel, Layout *layout)
{
  // set the set indices
  idx_group_set = gs;
  idx_dir_set = ds;
  idx_zone_set = zs;

  num_groups = input_vars->num_groups / input_vars->num_groupsets;
  group0 = gs * num_groups;

  num_directions = input_vars->num_directions / input_vars->num_dirsets;
  direction0 = ds * num_directions;
  directions = &direction_list[direction0];
  

  num_zones = 1;
  for(int dim = 0;dim < 3;++ dim){
    // Compute number of zones in this dimension
    nzones[dim] = layout->getNumZones(sdom_id, dim);
    num_zones *= nzones[dim];

    // Compute grid deltas in this dimension (including ghost zone deltas)
    std::pair<double, double> dim_extent = layout->getSpatialExtents(sdom_id, dim);
    zeros[dim] = dim_extent.first;
    double dx = (dim_extent.second-dim_extent.first)/(double)nzones[dim];
    deltas[dim].resize(nzones[dim]+2);
    for(int z = 0;z < nzones[dim]+2;++ z){
      deltas[dim][z] = dx;
    }
  }

  index_size[IMaterial::getName()] = 3;
  index_size[ILegendre::getName()] = input_vars->legendre_order+1;
  index_size[IMoment::getName()] = (input_vars->legendre_order+1)*(input_vars->legendre_order+1);
  index_size[IDirection::getName()] = num_directions;
  index_size[IGroup::getName()] = num_groups;
  index_size[IGlobalGroup::getName()] = input_vars->num_groups;
  index_size[IZone::getName()] = num_zones;
  index_size[IZoneIdx::getName()] = num_zones;
  index_size[IZoneI::getName()] = nzones[0];
  index_size[IZoneJ::getName()] = nzones[1];
  index_size[IZoneK::getName()] = nzones[2];

  // allocate storage for the sweep boundary data (upwind and downwind share same buffer)
  plane_data[0] = new SubTVec(kernel->nestingPsi(), num_groups, num_directions, nzones[1] * nzones[2]);
  plane_data[1] = new SubTVec(kernel->nestingPsi(), num_groups, num_directions, nzones[0] * nzones[2]);
  plane_data[2] = new SubTVec(kernel->nestingPsi(), num_groups, num_directions, nzones[0] * nzones[1]);

  // For block-jacobi parallel method
  old_plane_data[0] = new SubTVec(kernel->nestingPsi(), num_groups, num_directions, nzones[1] * nzones[2]);
  old_plane_data[1] = new SubTVec(kernel->nestingPsi(), num_groups, num_directions, nzones[0] * nzones[2]);
  old_plane_data[2] = new SubTVec(kernel->nestingPsi(), num_groups, num_directions, nzones[0] * nzones[1]);

  // allocate the storage for solution and source terms
  psi = new SubTVec(kernel->nestingPsi(), num_groups, num_directions, num_zones);
  psi->clear(0.0);
  rhs = new SubTVec(kernel->nestingPsi(), num_groups, num_directions, num_zones);
  sigt = new SubTVec(kernel->nestingSigt(), num_groups, 1, num_zones);
  sigt->clear(0.0);

  computeSweepIndexSet();

  // Setup neighbor data
  int dirs[3] = { directions[0].id, directions[0].jd, directions[0].kd};
  for(int dim = 0;dim < 3;++ dim){
    downwind[dim] = layout->getNeighbor(sdom_id, dim, dirs[dim]);
    upwind[dim] = layout->getNeighbor(sdom_id, dim, -1 * dirs[dim]);
  }

  // paint the mesh
  reg_volume[0] = 0.0;
  reg_volume[1] = 0.0;
  reg_volume[2] = 0.0;
  int num_subsamples = 4; // number of subsamples per spatial dimension
  double sample_vol_frac = 1.0 / (double)(num_subsamples*num_subsamples*num_subsamples);
  int zone_id = 0;
  double pz = zeros[2];

  for (int k = 0; k < nzones[2]; k++) {
    double sdz = deltas[2][k+1] / (double)(num_subsamples+1);
    double py = zeros[1];

    for (int j = 0; j != nzones[1]; j ++) {
      double sdy = deltas[1][j+1] / (double)(num_subsamples+1);
      double px = zeros[0];

      for (int i = 0; i != nzones[0]; i ++) {
        double sdx = deltas[0][i+1] / (double)(num_subsamples+1);

        double zone_volume = deltas[0][i+1] * deltas[1][j+1] * deltas[2][k+1];
        volume.push_back(zone_volume);

        // subsample probe the geometry to get our materials
        double frac[3] = {0.0, 0.0, 0.0}; // fraction of both materials
        double spz = pz + sdz;

        for(int sk = 0;sk < num_subsamples;++ sk){
          double spy = py + sdy;
          for(int sj = 0;sj < num_subsamples;++ sj){
            double spx = px + sdx;
            for(int si = 0;si < num_subsamples;++ si){

              int mat = queryMaterial(spx, spy, spz);
              frac[mat] += sample_vol_frac;

              spx += sdx;
            }
            spy += sdy;
          }
          spz += sdz;
        }

        // Add material to zone
        int nmixed = 0;
        for(int mat = 0;mat < 3;++ mat){          
          if(frac[mat] > 0.0){
            nmixed ++;
            if(nmixed == 1){
              zones_to_mixed.push_back(mixed_to_zones.size());
            }
            mixed_to_zones.push_back(zone_id);
            mixed_material.push_back(mat);
            mixed_fraction.push_back(frac[mat]);
            reg_volume[mat] += frac[mat] * zone_volume;
            
            // initialize background sigt
            for(int g = 0;g < num_groups;++ g){
              (*sigt)(g,0,zone_id) += frac[mat] * input_vars->sigt[mat];
            }
          }
        }
        num_mixed.push_back(nmixed);

        // increment zone
        px += deltas[0][i+1];
        zone_id ++;
      }
      py += deltas[1][j+1];
    }
    pz += deltas[2][k+1];
  }
  
  // store number of mixed elements
  index_size[IMix::getName()] = mixed_to_zones.size();
}

void Subdomain::setVars(SubTVec *ell_ptr, SubTVec *ell_plus_ptr,
    SubTVec *phi_ptr, SubTVec *phi_out_ptr){

  ell = ell_ptr;
  ell_plus = ell_plus_ptr;
  phi = phi_ptr;
  phi_out = phi_out_ptr;
}


/**
 * Randomizes data for a set.
 */
void Subdomain::randomizeData(void){
  psi->randomizeData();
  rhs->randomizeData();
  sigt->randomizeData();

  for(int d = 0;d < 3;++ d){
    for(int i = 0;i < deltas[d].size();++ i){
      deltas[d][i] = drand48();
    }
  }
}

/**
 * Copies two sets, allowing for different nestings.
 */
void Subdomain::copy(Subdomain const &b){
  psi->copy(*b.psi);
  rhs->copy(*b.rhs);
  sigt->copy(*b.sigt);

  for(int d = 0;d < 3;++ d){
    deltas[d] = b.deltas[d];
  }
}

/**
 * Compares two sets, allowing for different nestings.
 */
bool Subdomain::compare(Subdomain const &b, double tol, bool verbose){
  std::stringstream namess;
  namess << "gdset[gs=" << idx_group_set << ", ds=" << idx_dir_set << ", zs=" << idx_zone_set << "]";
  std::string name = namess.str();

  bool is_diff = false;
  is_diff |= psi->compare(name+".psi", *b.psi, tol, verbose);
  is_diff |= rhs->compare(name+".rhs", *b.rhs, tol, verbose);
  is_diff |= sigt->compare(name+".sigt", *b.sigt, tol, verbose);

  is_diff |= compareVector(name+".deltas[0]", deltas[0], b.deltas[0], tol, verbose);
  is_diff |= compareVector(name+".deltas[1]", deltas[1], b.deltas[1], tol, verbose);
  is_diff |= compareVector(name+".deltas[2]", deltas[2], b.deltas[2], tol, verbose);

  return is_diff;
}

/**
 * Compute sweep index sets.
 * Determines logical indices, and increments for i,j,k based on grid
 * information and quadrature set sweeping direction.
 */
void Subdomain::computeSweepIndexSet(void){
  if(directions[0].id > 0){
    sweep_block.start_i = 0;
    sweep_block.end_i = nzones[0];
    sweep_block.inc_i = 1;
  }
  else {
    sweep_block.start_i = nzones[0]-1;
    sweep_block.end_i = -1;
    sweep_block.inc_i = -1;
  }

  if(directions[0].jd > 0){
    sweep_block.start_j = 0;
    sweep_block.end_j = nzones[1];
    sweep_block.inc_j = 1;
  }
  else {
    sweep_block.start_j = nzones[1]-1;
    sweep_block.end_j = -1;
    sweep_block.inc_j = -1;
  }

  if(directions[0].kd > 0){
    sweep_block.start_k = 0;
    sweep_block.end_k = nzones[2];
    sweep_block.inc_k =  1;
  }
  else {
    sweep_block.start_k = nzones[2]-1;
    sweep_block.end_k = -1;
    sweep_block.inc_k = -1;
  }

  sweep_block.indexset_sweep = RAJA::IndexSet();
  int N = nzones[0];
  if (nzones[1] > N) N=nzones[1];
  if (nzones[2] > N) N=nzones[2];

  int i_inc = sweep_block.inc_i;
  int j_inc = sweep_block.inc_j;
  int k_inc = sweep_block.inc_k;
  int i_min, i_max, j_min, j_max, k_min, k_max;
  int counter = 0;
  int Nslices = 0;
  int offset[3*N+4];
  offset[0] = 0;

  if ( i_inc == 1){
    i_min = sweep_block.start_i;
    i_max = sweep_block.end_i-1;
  }
  else{
    i_min = sweep_block.end_i+1;
    i_max = sweep_block.start_i;
  }
  if ( j_inc == 1){
    j_min = sweep_block.start_j;
    j_max = sweep_block.end_j-1;
  }
  else{
    j_min = sweep_block.end_j+1;
    j_max = sweep_block.start_j;
  }
  if ( k_inc == 1){
    k_min = sweep_block.start_k;
    k_max = sweep_block.end_k-1;
  }
  else{
    k_min = sweep_block.end_k+1;
    k_max = sweep_block.start_k;
  }
  int ii_tmp = (1 - i_inc)/2*i_max;
  int jj_tmp = (1 - j_inc)/2*j_max;
  int kk_tmp = (1 - k_inc)/2*k_max;


  //sweep_block.ii_jj_kk_z_idx = new int[nzones[0]*nzones[1]*nzones[2]*4];
  sweep_block.idx_to_i = new int[nzones[0]*nzones[1]*nzones[2]];
  sweep_block.idx_to_j = new int[nzones[0]*nzones[1]*nzones[2]];
  sweep_block.idx_to_k = new int[nzones[0]*nzones[1]*nzones[2]];
  sweep_block.idx_to_z = new int[nzones[0]*nzones[1]*nzones[2]];
  
  if(false){ // try a dummy sweep pattern.. this will give you the WRONG answer.. just to understand memory performance
    int z = 0;
    int zstart = z;
    for(int k = 0;k < nzones[2];++k){          
      for(int j = 0;j < nzones[1];++j){        
        for(int i = 0;i < nzones[0];++i){ 
          sweep_block.idx_to_i[z] = i;
          sweep_block.idx_to_j[z] = j;
          sweep_block.idx_to_k[z] = k;
          sweep_block.idx_to_z[z] = z;
          ++ z;
        }        
      }
      
    }
    sweep_block.indexset_sweep.push_back(
            RAJA::RangeSegment(zstart, z)); 
  }
  else{
    for (int C = 0; C <=(3*N); ++C){   //for each C we can touch zone["i","j","k"]  as well as "d" and "group"    in parallel
     int FLAG=0;
     for (int i = 0; i <= C; ++i){
       for (int j = 0; j <= C; ++j){
          int k = C - i - j; // surface equation i+j+j=C
          //flip if needed

          int ii = ii_tmp + i*i_inc;
          int jj = jj_tmp + j*j_inc;
          int kk = kk_tmp + k*k_inc;

          if (ii <= i_max && jj <= j_max && kk <= k_max && ii >= i_min && jj >= j_min && kk >= k_min){

            sweep_block.idx_to_i[counter] = ii;
            sweep_block.idx_to_j[counter] = jj;
            sweep_block.idx_to_k[counter] = kk;
            sweep_block.idx_to_z[counter] = ii + nzones[0]*jj + nzones[0]*nzones[1]*kk;//  Zonal_INDEX(ii, jj, kk);
            counter++; //counts all elements
            FLAG++;   //counts elements per slice
         }
       }
     }
     if (FLAG){
        Nslices++;
        offset[Nslices] = offset[Nslices-1] + FLAG;

        // an index set which describes each hyperplane as a RangeSegment
        sweep_block.indexset_sweep.push_back(
            RAJA::RangeSegment(offset[Nslices-1], offset[Nslices]));
     }
   }
   } // hyper-plane generator
}

namespace {
  double FactFcn(int n)
  {
    double fact = 1.0;
    for(int i = n;i > 0 ;--i){
      fact *= (double)i;
    }
    return(fact);
  }

  inline double PnmFcn(int n, int m, double x)
  {
    /*-----------------------------------------------------------------
     * It is assumed that 0 <= m <= n and that abs(x) <= 1.0.
     * No error checking is done, however.
     *---------------------------------------------------------------*/
    double fact, pnn, pmm, pmmp1, somx2;

    int i, nn;

    if(std::abs(x) > 1.0){
      printf("Bad input to ardra_PnmFcn: abs(x) > 1.0, x = %e\n", x);
      MPI_Abort(MPI_COMM_WORLD, 1);
    }
    else if((x > 1.0) && (x <= 1.0)){
      x = 1.0;
    }
    else if((-1.0 <= x ) && (x < -1.0)){
      x = -1.0;
    }

    pmm=1.0;
    if(m > 0){
      somx2=sqrt((1.0-x)*(1.0+x));
      fact=1.0;
      for(i=1; i<=m; i++){
        pmm *= -fact*somx2;
        fact += 2.0;
      }
    }
    if(n == m){
      return(pmm);
    }
    else {
      pmmp1=x*(2*m+1)*pmm;
      if(n == (m+1)){
        return(pmmp1);
      }
      else {
        for(nn=m+2; nn<=n; nn++){
          pnn=(x*(2*nn-1)*pmmp1-(nn+m-1)*pmm)/(nn-m);
          pmm=pmmp1;
          pmmp1=pnn;
        }
        return(pnn);
      }
    }
  }

  inline double YnmFcn(int n, int m, double mu, double eta, double xi)
  {
    double fac1, fac2, anm, ynm, pnm, dm0, taum, tmp, phi, phi_tmp;
    double floor=1.e-20;
    int nn, mm;

    /* Calculate the correct phi for omega=(mu,eta,xi) */
    tmp = fabs(eta/(mu+floor));
    phi_tmp = atan(tmp);
    if( (mu>0) && (eta>0) ){
      phi = phi_tmp;
    }
    else if( (mu<0) && (eta>0) ){
      phi = M_PI - fabs(phi_tmp);
    }
    else if( (mu<0) && (eta<0) ){
      phi = M_PI + fabs(phi_tmp);
    }
    else {
      phi = 2.0*M_PI - fabs(phi_tmp);
    }

    /* Begin evaluation of Ynm(omega) */
    nn = n - std::abs(m);
    fac1 = (double) FactFcn(nn);
    nn = n + std::abs(m);
    fac2 = (double) FactFcn(nn);
    mm = std::abs(m);
    pnm = PnmFcn(n, mm, xi);
    tmp = ((double) m)*phi;
    if(m >= 0){
      taum = cos(tmp);
    }
    else {taum = sin(-tmp); }
    if(m == 0){
      dm0 = 1.0;
    }
    else {dm0 = 0.0; }
    tmp = ((2*n+1)*fac1)/(2.0*(1.0+dm0)*M_PI*fac2);
    anm = sqrt( tmp );
    ynm = anm*pnm*taum;
    return(ynm);
  }
}

/**
 * Compute L and L+
 * This assumes that the quadrature set is defined.
 */
void Subdomain::computeLLPlus(int legendre_order){
  double SQRT4PI = std::sqrt(4*M_PI);
  for(int n=0, nm=0; n < legendre_order+1; n++){
    for(int m=-n; m<=n; m++){
      for(int d=0; d<num_directions; d++){
        // Get quadrature point info
        double xcos = (directions[d].id)*(directions[d].xcos);
        double ycos = (directions[d].jd)*(directions[d].ycos);
        double zcos = (directions[d].kd)*(directions[d].zcos);
        double w =  directions[d].w;

        double ynm = YnmFcn(n, m, xcos, ycos, zcos);

        // Compute element of L and L+
        (*ell)(nm,d,0) = w*ynm/SQRT4PI;
        (*ell_plus)(nm,d,0) = ynm*SQRT4PI;
      }
      nm ++;
    }
  }
}

