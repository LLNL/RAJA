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

#include<Kripke/Layout.h>

#include<Kripke/Input_Variables.h>
#include<mpi.h>

namespace {
  /*
    The following 2 routines are used to map:
      1) mpi ranks to/from processors in x,y,z
      2) zoneset ids to/from zoneset in x,y,z
  */

  /**
    Helper routine to take an index, and return a 3-dimensional set of indices,
    given size of each index dimension.
  */
  inline void rankToIndices(int rank, int *indices, int const *sizes){
    indices[0] = rank / (sizes[1]*sizes[2]);
    rank = rank % (sizes[1]*sizes[2]);
    indices[1] = rank / sizes[2];
    indices[2] = rank % sizes[2];
  }

  /**
    Helper routine to take an index, and return a 3-dimensional set of indices,
    given size of each index dimension.
  */
  inline int indicesToRank(int const *indices, int const *sizes){
    int rank;

    rank =  indices[0]*(sizes[1]*sizes[2]);
    rank += indices[1]*sizes[2];
    rank += indices[2];

    return rank;
  }
}

Layout::Layout(Input_Variables *input_vars){
  num_group_sets = input_vars->num_groupsets;
  num_direction_sets = input_vars->num_dirsets;
  num_zone_sets = 1;
  for(int dim = 0;dim < 3;++ dim){
    num_zone_sets_dim[dim] = input_vars->num_zonesets_dim[dim];
    num_zone_sets *= input_vars->num_zonesets_dim[dim];
  }

  // grab total number of zones
  total_zones[0] = input_vars->nx;
  total_zones[1] = input_vars->ny;
  total_zones[2] = input_vars->nz;

  // Grab size of processor grid
  num_procs[0] = input_vars->npx;
  num_procs[1] = input_vars->npy;
  num_procs[2] = input_vars->npz;

  /* Set the requested processor grid size */
  int R = num_procs[0] * num_procs[1] * num_procs[2];

  /* Check requested size is the same as MPI_COMM_WORLD */
  int size;
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  if(R != size){
    int myid;
    MPI_Comm_rank(MPI_COMM_WORLD, &myid);
    if(myid == 0){
      printf("ERROR: Incorrect number of MPI tasks. Need %d MPI tasks.", R);
    }
    MPI_Abort(MPI_COMM_WORLD, 1);
  }

  /* Compute the local coordinates in the processor decomposition */
  int mpi_rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
  rankToIndices(mpi_rank, our_rank, num_procs);
}
Layout::~Layout(){

}

/**
  Computes the subdomain ID based on a given groupset, directionset, and zoneset.
*/
int Layout::setIdToSubdomainId(int gs, int ds, int zs) const{
  int indices[3] = {gs, ds, zs};
  int sizes[3] = {num_group_sets, num_direction_sets, num_zone_sets};

  return indicesToRank(indices, sizes);
}

/**
  Computes groupset, directionset, and zoneset from a subdomain ID.
*/
void Layout::subdomainIdToSetId(int sdom_id, int &gs, int &ds, int &zs) const {
  int indices[3];
  int sizes[3] = {num_group_sets, num_direction_sets, num_zone_sets};

  rankToIndices(sdom_id, indices, sizes);

  gs = indices[0];
  ds = indices[1];
  zs = indices[2];
}

/**
  Computes the zoneset id along a particular dimension.
*/
int Layout::subdomainIdToZoneSetDim(int sdom_id, int dim) const{
  // Compute zoneset
  int gs, ds, zs;
  subdomainIdToSetId(sdom_id, gs, ds, zs);

  // Compute zone set
  int zs_dim[3];
  rankToIndices(zs, zs_dim, num_zone_sets_dim);

  return zs_dim[dim];
}

/**
  Computes the number of zones in this subdomain, along specified dimension.
*/
int Layout::getNumZones(int sdom_id, int dim) const{

  // get the zoneset index along the specified dimension
  int zs_dim = subdomainIdToZoneSetDim(sdom_id, dim);

  int total_subdomains = num_procs[dim] * num_zone_sets_dim[dim];
  int global_subdomain  = num_zone_sets_dim[dim] * our_rank[dim] + zs_dim;

  // Compute subset of global zone indices
  int num_zones = total_zones[dim] / total_subdomains;
  int rem = total_zones[dim] % total_subdomains;
  if(rem != 0 && global_subdomain < rem){
    num_zones ++;
  }

  return num_zones;
}






BlockLayout::BlockLayout(Input_Variables *input_vars) :
  Layout(input_vars)
{

}
BlockLayout::~BlockLayout(){

}

Neighbor BlockLayout::getNeighbor(int our_sdom_id, int dim, int dir) const{
  Neighbor n;

  // get our processor indices, so we can find neighbors
  int proc[3] = {our_rank[0], our_rank[1], our_rank[2]};

  int gs, ds, zs;
  subdomainIdToSetId(our_sdom_id, gs, ds, zs);

  // Compute out spatial subdomain indices
  int zs_dim[3];
  for(int d = 0;d < 3;++ d){
    zs_dim[d] = subdomainIdToZoneSetDim(our_sdom_id, d);
  }

  // Offest along dir,dim to get neighboring indices
  zs_dim[dim] += dir;

  // Check if the neighbor is remote, and wrap zoneset indices
  if(zs_dim[dim] >= num_zone_sets_dim[dim]){
    zs_dim[dim] = 0;
    proc[dim] += dir;
  }
  else if(zs_dim[dim] < 0){
    zs_dim[dim] = num_zone_sets_dim[dim]-1;
    proc[dim] += dir;
  }

  // Compute the mpi rank of the neighbor
  if(proc[dim] < 0 || proc[dim] >= num_procs[dim]){
    // we hit a boundary condition
    n.mpi_rank = -1;
    n.subdomain_id = -1;
  }
  else{
    // There is a neighbor, so compute its rank
    n.mpi_rank = indicesToRank(proc, num_procs);

    // Compute neighboring subdomain id
    zs = indicesToRank(zs_dim, num_zone_sets_dim);
    n.subdomain_id = setIdToSubdomainId(gs, ds, zs);
  }

  return n;
}

/**
  Compute the spatial extents of a subdomain along a given dimension.
*/
std::pair<double, double> BlockLayout::getSpatialExtents(int sdom_id, int dim) const{

  // Start with global problem dimensions
  std::pair<double, double> ext_global(-60.0, 60.0);
  if(dim == 1){
    ext_global.first = -100.0;
    ext_global.second = 100.0;
  }

  // Subdivide by number of processors in specified dimension
  double dx = (ext_global.second - ext_global.first) / (double)num_procs[dim];
  std::pair<double, double> ext_proc(
    ext_global.first + dx*(double)our_rank[dim],
    ext_global.first + dx*(double)(our_rank[dim] + 1)
  );

  // get the zoneset index along the specified dimension
  int zs_dim = subdomainIdToZoneSetDim(sdom_id, dim);

  // Subdivide by number of subdomains in specified dimension
  double sdx = (ext_proc.second - ext_proc.first) / (double)num_zone_sets_dim[dim];
  std::pair<double, double> ext_sdom(
    ext_proc.first + sdx*(double)zs_dim,
    ext_proc.first + sdx*(double)(zs_dim + 1)
  );

  return ext_sdom;
}



ScatterLayout::ScatterLayout(Input_Variables *input_vars) :
  Layout(input_vars)
{

}
ScatterLayout::~ScatterLayout(){

}

Neighbor ScatterLayout::getNeighbor(int our_sdom_id, int dim, int dir) const{
  Neighbor n;

  // get our processor indices, so we can find neighbors
  int proc[3] = {our_rank[0], our_rank[1], our_rank[2]};

  int gs, ds, zs;
  subdomainIdToSetId(our_sdom_id, gs, ds, zs);

  // Compute our spatial subdomain indices
  int zs_dim[3];
  for(int d = 0;d < 3;++ d){
    zs_dim[d] = subdomainIdToZoneSetDim(our_sdom_id, d);
  }

  // Offest along dir,dim to get neighboring subdomain indices
  proc[dim] += dir;

  // Check if we wrapped mpi ranks, and should bump zoneset indices
  if(proc[dim] >= num_procs[dim]){
    proc[dim] = 0;
    zs_dim[dim] += dir;
  }
  else if(proc[dim] < 0){
    proc[dim] = num_procs[dim]-1;
    zs_dim[dim] += dir;
  }

  // Compute zone set indices, and detect boundary condition
  if(zs_dim[dim] < 0 || zs_dim[dim] >= num_zone_sets_dim[dim]){
    // we hit a boundary condition
    n.mpi_rank = -1;
    n.subdomain_id = -1;

  }
  else{
    // There is a neighbor, so compute its rank
    n.mpi_rank = indicesToRank(proc, num_procs);

    // Compute neighboring subdomain id
    zs = indicesToRank(zs_dim, num_zone_sets_dim);
    n.subdomain_id = setIdToSubdomainId(gs, ds, zs);
  }


  return n;
}

/**
  Compute the spatial extents of a subdomain along a given dimension.
*/
std::pair<double, double> ScatterLayout::getSpatialExtents(int sdom_id, int dim) const{

  // Start with global problem dimensions
  std::pair<double, double> ext_global(-60.0, 60.0);
  if(dim == 1){
    ext_global.first = -100.0;
    ext_global.second = 100.0;
  }

  // get the zoneset index along the specified dimension
  int zs_dim = subdomainIdToZoneSetDim(sdom_id, dim);

  // Subdivide by number of subdomains in specified dimension
  double sdx = (ext_global.second - ext_global.first) / (double)num_zone_sets_dim[dim];
  std::pair<double, double> ext_sdom(
    ext_global.first + sdx*(double)zs_dim,
    ext_global.first + sdx*(double)(zs_dim + 1)
  );

  // Subdivide by number of processors in specified dimension
  double dx = (ext_sdom.second - ext_sdom.first) / (double)num_procs[dim];
  std::pair<double, double> ext_proc(
    ext_sdom.first + dx*(double)our_rank[dim],
    ext_sdom.first + dx*(double)(our_rank[dim] + 1)
  );


  return ext_proc;
}


/**
  Factory to create Layout object based on user defined inputs
*/
Layout *createLayout(Input_Variables *input_vars){
  switch(input_vars->layout_pattern){
    case 0:
      return new BlockLayout(input_vars);
    case 1:
      return new ScatterLayout(input_vars);
  }
  printf("Unknown Layout patter\n");
  MPI_Abort(MPI_COMM_WORLD, 1);
  return NULL;
}
