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

#include<Kripke/Input_Variables.h>

#include<mpi.h>

/**
* Setup the default input choices
*/
Input_Variables::Input_Variables() : 
  nx(16), ny(16), nz(16),
  num_directions(96),
  num_groups(32),
  legendre_order(4),
  quad_num_polar(0),
  quad_num_azimuthal(0),
 
  nesting(NEST_DGZ),
 
  npx(1), npy(1), npz(1),
  num_dirsets(8),
  num_groupsets(2),
  layout_pattern(0),
  
  niter(10),
  parallel_method(PMETHOD_SWEEP),
  run_name("kripke")
{
  num_zonesets_dim[0] = 1; 
  num_zonesets_dim[1] = 1;
  num_zonesets_dim[2] = 1;

  sigt[0] = 0.1;  
  sigt[1] = 0.0001;
  sigt[2] = 0.1;
  
  sigs[0] = 0.05;  
  sigs[1] = 0.00005;
  sigs[2] = 0.05; 
}

/**
 *  Checks validity of inputs, returns 'true' on error.
 */
bool Input_Variables::checkValues(void) const{
  // make sure any output only goes to root
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  if(num_zonesets_dim[0] <= 0 || num_zonesets_dim[1] <= 0 || num_zonesets_dim[2] <= 0){
    if(!rank)
      printf("Number of zone-sets in each dim need to be greater than or equal to 1\n");
    return true;
  }
  
  if(layout_pattern < 0 || layout_pattern > 1){
    if(!rank)
      printf("Layout(%d) must be either 0 or 1\n", layout_pattern);
    return true;
  }
  
  if(nesting < 0){
    if(!rank)
      printf("Invalid nesting selected\n");
    return true;
  }
  
  if(num_groups < 1){
    if(!rank)
      printf("Number of groups (%d) needs to be at least 1\n", num_groups);
    return true;
  }
  
  if(num_groups % num_groupsets){
    if(!rank)
      printf("Number of groups (%d) must be evenly divided by number of groupsets (%d)\n",
        num_groups, num_groupsets);
    return true;
  }
  
  if(num_directions < 8){
    if(!rank)
      printf("Number of directions (%d) needs to be at least 8\n", num_directions);
    return true;
  }
  
  if(num_dirsets % 8 && num_dirsets < 8){
    if(!rank)
      printf("Number of direction sets (%d) must be a multiple of 8\n", num_dirsets);
    return true;
  }
  
  if(num_directions % num_dirsets){
    if(!rank)
      printf("Number of directions (%d) must be evenly divided by number of directionsets(%d)\n",
        num_directions, num_dirsets);
    return true;
  }
  
  if(legendre_order < 0){
    if(!rank)
      printf("Legendre scattering order (%d) must be >= 0\n", legendre_order);
    return true;
  }
  
  if(niter < 1){
    if(!rank)
      printf("You must run at least one iteration (%d)\n", niter);
    return true;
  }
  
  return false;
}
