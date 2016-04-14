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

#include <Kripke/Input_Variables.h>
#include <Kripke/Layout.h>
#include <Kripke/SubTVec.h>
#include <cmath>
#include <sstream>
#include <mpi.h>

#ifdef KRIPKE_USE_SILO
#include <sys/stat.h>
#include <silo.h>
#include <string.h>
#endif

/**
 * Grid_Data constructor
*/
Grid_Data::Grid_Data(Input_Variables *input_vars)
{
  // Create object to describe processor and subdomain layout in space
  // and their adjacencies
  Layout *layout = createLayout(input_vars);

  // create the kernel object based on nesting
  kernel = createKernel(input_vars->nesting, 3);

  // Create quadrature set (for all directions)
  int total_num_directions = input_vars->num_directions;
  InitDirections(this, input_vars);

  num_direction_sets = input_vars->num_dirsets;
  num_directions_per_set = total_num_directions/num_direction_sets;
  num_group_sets = input_vars->num_groupsets;
  num_groups_per_set = input_vars->num_groups/ num_group_sets;
  num_zone_sets = 1;
  for(int dim = 0;dim < 3;++ dim){
    num_zone_sets *= input_vars->num_zonesets_dim[dim];
  }

  legendre_order = input_vars->legendre_order;
  total_num_moments = (legendre_order+1)*(legendre_order+1);

  int num_subdomains = num_direction_sets*num_group_sets*num_zone_sets;

  Nesting_Order nest = input_vars->nesting;

  /* Set ncalls */
  niter = input_vars->niter;

  // setup mapping of moments to legendre coefficients
  moment_to_coeff.resize(total_num_moments);
  int nm = 0;
  for(int n = 0;n < legendre_order+1;++ n){
    for(int m = -n;m <= n; ++ m){
      moment_to_coeff[nm] = n;
      ++ nm;
    }
  }

  // setup cross-sections
  int total_num_groups = num_group_sets*num_groups_per_set;
  sigma_tot.resize(total_num_groups, 0.0);

  // Setup scattering transfer matrix for 3 materials  

  sigs = new SubTVec(kernel->nestingSigs(), total_num_groups*total_num_groups, legendre_order+1, 3);

  // Set to isotropic scattering given user inputs
  sigs->clear(0.0);
  for(int mat = 0;mat < 3;++ mat){
    for(int g = 0;g < total_num_groups;++ g){
      int idx_g_gp = g*total_num_groups + g;
      (*sigs)(idx_g_gp, 0, mat) = input_vars->sigs[mat];
    }
  }

  // just allocate pointer vectors, we will allocate them below
  ell.resize(num_direction_sets, NULL);
  ell_plus.resize(num_direction_sets, NULL);
  phi.resize(num_zone_sets, NULL);
  phi_out.resize(num_zone_sets, NULL);

  // Initialize Subdomains
  zs_to_sdomid.resize(num_zone_sets);
  subdomains.resize(num_subdomains);
  for(int gs = 0;gs < num_group_sets;++ gs){
    for(int ds = 0;ds < num_direction_sets;++ ds){
      for(int zs = 0;zs < num_zone_sets;++ zs){
        // Compupte subdomain id
        int sdom_id = layout->setIdToSubdomainId(gs, ds, zs);

        // Setup the subdomain
        Subdomain &sdom = subdomains[sdom_id];
        sdom.setup(sdom_id, input_vars, gs, ds, zs, directions, kernel, layout);

        // Create ell and ell_plus, if this is the first of this ds
        bool compute_ell = false;
        if(ell[ds] == NULL){
          ell[ds] = new SubTVec(kernel->nestingEll(), total_num_moments, sdom.num_directions, 1);
          ell_plus[ds] = new SubTVec(kernel->nestingEllPlus(), total_num_moments, sdom.num_directions, 1);

          compute_ell = true;
        }

        // Create phi and phi_out, if this is the first of this zs
        if(phi[zs] == NULL){
          phi[zs] = new SubTVec(nest, total_num_groups, total_num_moments, sdom.num_zones);
          phi_out[zs] = new SubTVec(nest, total_num_groups, total_num_moments, sdom.num_zones);
        }

        // setup zs to sdom mapping
        if(gs == 0 && ds == 0){
          zs_to_sdomid[zs] = sdom_id;
        }

        // Set the variables for this subdomain
        sdom.setVars(ell[ds], ell_plus[ds], phi[zs], phi_out[zs]);

        if(compute_ell){
          // Compute the L and L+ matrices
          sdom.computeLLPlus(legendre_order);
        }
      }
    }
  }
  delete layout;



  // Now compute number of elements allocated globally,
  // and get each materials volume
  long long vec_size[4] = {0,0,0,0};
  double vec_volume[3] = {0.0, 0.0, 0.0};
  for(int sdom_id = 0;sdom_id < subdomains.size();++sdom_id){
    Subdomain &sdom = subdomains[sdom_id];
    vec_size[0] += sdom.psi->elements;
    vec_size[1] += sdom.psi->elements;
  }
  for(int zs = 0;zs < num_zone_sets;++ zs){
    vec_size[2] += phi[zs]->elements;
    vec_size[3] += phi_out[zs]->elements;
    int sdom_id = zs_to_sdomid[zs];
    for(int mat = 0;mat < 3;++ mat){
      vec_volume[mat] += subdomains[sdom_id].reg_volume[mat];
    }
  }

  long long global_size[4];
  MPI_Reduce(vec_size, global_size, 4, MPI_LONG_LONG_INT, MPI_SUM, 0, MPI_COMM_WORLD);

  double global_volume[3];
  MPI_Reduce(vec_volume, global_volume, 3, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

  int mpi_rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
  if(mpi_rank == 0){
    printf("Unknown counts: psi=%ld, rhs=%ld, phi=%ld, phi_out=%ld\n",
      (long)global_size[0], (long)global_size[1], (long)global_size[2], (long)global_size[3]);
    printf("Region volumes: Reg1=%e, Reg2=%e, Reg3=%e\n",
        global_volume[0], global_volume[1], global_volume[2]);
  }
}

Grid_Data::~Grid_Data(){
  delete kernel;
  for(int zs = 0;zs < num_zone_sets;++ zs){
    delete phi[zs];
    delete phi_out[zs];
  }
  for(int ds = 0;ds < num_direction_sets;++ ds){
    delete ell[ds];
    delete ell_plus[ds];
  }
  delete sigs;
}

/**
 * Randomizes all variables and matrices for testing suite.
 */
void Grid_Data::randomizeData(void){
  for(int i = 0;i < sigma_tot.size();++i){
    sigma_tot[i] = drand48();
  }

  for(int i = 0;i < directions.size();++i){
    directions[i].xcos = drand48();
    directions[i].ycos = drand48();
    directions[i].zcos = drand48();
  }


  for(int s = 0;s < subdomains.size();++ s){
    subdomains[s].randomizeData();
  }

  for(int zs = 0;zs < num_zone_sets;++ zs){
    phi[zs]->randomizeData();
    phi_out[zs]->randomizeData();
  }

  for(int ds = 0;ds < num_direction_sets;++ ds){
    ell[ds]->randomizeData();
    ell_plus[ds]->randomizeData();
  }

  sigs->randomizeData();
}


/**
 * Returns the integral of psi.. to look at convergence
 */
double Grid_Data::particleEdit(void){
  // sum up particles for psi and rhs
  double part = 0.0;
  for(int sdom_id = 0;sdom_id < subdomains.size();++ sdom_id){
    Subdomain &sdom = subdomains[sdom_id];

    int num_zones = sdom.num_zones;
    int num_directions = sdom.num_directions;
    int num_groups= sdom.num_groups;
    Directions *dirs = sdom.directions;

    for(int z = 0;z < num_zones;++ z){
      double vol = sdom.volume[z];
      for(int d = 0;d < num_directions;++ d){
        double w = dirs[d].w;
        for(int g = 0;g < num_groups;++ g){
          part += w * (*sdom.psi)(g,d,z) * vol;
        }
      }
    }
  }

  // reduce
  double part_global;
  MPI_Reduce(&part, &part_global, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

  return part_global;
}


/**
 * Copies all variables and matrices for testing suite.
 * Correctly copies data from one nesting to another.
 */
void Grid_Data::copy(Grid_Data const &b){
  sigma_tot = b.sigma_tot;
  directions = b.directions;

  subdomains.resize(b.subdomains.size());
  for(int s = 0;s < subdomains.size();++ s){
    subdomains[s].copy(b.subdomains[s]);
  }

  for(int zs = 0;zs < num_zone_sets;++ zs){
    phi[zs]->copy(*b.phi[zs]);
    phi_out[zs]->copy(*b.phi_out[zs]);
  }

  for(int ds = 0;ds < ell.size();++ ds){
    ell[ds]->copy(*b.ell[ds]);
    ell_plus[ds]->copy(*b.ell_plus[ds]);
  }

  sigs->copy(*b.sigs);
}

/**
 * Compares all variables and matrices for testing suite.
 * Correctly compares data from one nesting to another.
 */
bool Grid_Data::compare(Grid_Data const &b, double tol, bool verbose){
  bool is_diff = false;

  for(int i = 0;i < directions.size();++i){
    std::stringstream dirname;
    dirname << "directions[" << i << "]";

    is_diff |= compareScalar(dirname.str()+".xcos",
        directions[i].xcos, b.directions[i].xcos, tol, verbose);

    is_diff |= compareScalar(dirname.str()+".ycos",
        directions[i].ycos, b.directions[i].ycos, tol, verbose);

    is_diff |= compareScalar(dirname.str()+".zcos",
        directions[i].zcos, b.directions[i].zcos, tol, verbose);
  }

  for(int s = 0;s < subdomains.size();++ s){
    is_diff |= subdomains[s].compare(
        b.subdomains[s], tol, verbose);

  }
  is_diff |= compareVector("sigma_tot", sigma_tot, b.sigma_tot, tol, verbose);

  for(int zs = 0;zs < num_zone_sets;++ zs){
    is_diff |= phi[zs]->compare("phi", *b.phi[zs], tol, verbose);
    is_diff |= phi_out[zs]->compare("phi_out", *b.phi_out[zs], tol, verbose);
  }

  for(int ds = 0;ds < ell.size();++ ds){
    is_diff |= ell[ds]->compare("ell", *b.ell[ds], tol, verbose);
    is_diff |= ell_plus[ds]->compare("ell_plus", *b.ell_plus[ds], tol, verbose);
  }

  is_diff |= sigs->compare("sigs", *b.sigs, tol, verbose);

  return is_diff;
}


#ifdef KRIPKE_USE_SILO

enum MultivarType {
  MULTI_MESH,
  MULTI_MAT,
  MULTI_VAR
};

namespace {
  /**
    Writes a multimesh or multivar to the root file.
  */

  void siloWriteMulti(DBfile *root, MultivarType mv_type,
    std::string const &fname_base, std::string const &var_name,
    std::vector<int> sdom_id_list, int var_type = 0)
  {
    int mpi_size;
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
    int num_sdom = sdom_id_list.size();

    // setup names and types
    std::vector<int> var_types(mpi_size*num_sdom, var_type);
    std::vector<char *> var_names(mpi_size*num_sdom);
    int var_idx = 0;
    for(int rank = 0;rank < mpi_size;++ rank){
      for(int idx = 0;idx < num_sdom;++ idx){
        int sdom_id = sdom_id_list[idx];
        std::stringstream name;
        name << fname_base << "/rank_" << rank << ".silo:/sdom" << sdom_id << "/" << var_name;
        var_names[var_idx] = strdup(name.str().c_str());
        var_idx ++;
      }
    }

    if(mv_type == MULTI_MESH){
      DBPutMultimesh(root, var_name.c_str(), mpi_size*num_sdom,
          &var_names[0], &var_types[0], NULL);
    }
    else if(mv_type == MULTI_MAT){
      DBPutMultimat(root, var_name.c_str(), mpi_size*num_sdom,
          &var_names[0],  NULL);
    }
    else{
      DBPutMultivar(root, var_name.c_str(), mpi_size*num_sdom,
          &var_names[0],  &var_types[0] , NULL);
    }

    // cleanup
    for(int i = 0;i < mpi_size*num_sdom; ++i){
      free(var_names[i]);
    }
  }

  void siloWriteRectMesh(DBfile *silo_file,
    std::string const &mesh_name,
    int const *nzones,
    double const *zeros,
    double const *deltas_x,
    double const *deltas_y,
    double const *deltas_z)
  {
    static char const *coordnames[3] = {"X", "Y", "Z"};
    double const *deltas[3] = {deltas_x, deltas_y, deltas_z};
    double *coords[3];
    for(int dim = 0;dim < 3;++ dim){
      coords[dim] = new double[nzones[dim]];
      coords[dim][0] = zeros[dim];
      for(int z = 0;z < nzones[dim];++ z){
        coords[dim][1+z] = coords[dim][z] + deltas[dim][z];
      }
    }
    int nnodes[3] = {
      nzones[0]+1,
      nzones[1]+1,
      nzones[2]+1
    };

    DBPutQuadmesh(silo_file, mesh_name.c_str(), const_cast<char**>(coordnames), coords, nnodes, 3, DB_DOUBLE,
        DB_COLLINEAR, NULL);

    // cleanup
    delete[] coords[0];
    delete[] coords[1];
    delete[] coords[2];
  }


} //namespace


void Grid_Data::writeSilo(std::string const &fname_base){

  // Recompute Phi... so we can write out phi0
  kernel->LTimes(this);

  int mpi_rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);

  if(mpi_rank == 0){
    // Create a root file
    std::string fname_root = fname_base + ".silo";
    DBfile *root = DBCreate(fname_root.c_str(),
        DB_CLOBBER, DB_LOCAL, NULL, DB_HDF5);

    // Write out multimesh and multivars
    siloWriteMulti(root, MULTI_MESH, fname_base, "mesh", zs_to_sdomid, DB_QUAD_RECT);
    siloWriteMulti(root, MULTI_MAT, fname_base, "material", zs_to_sdomid);
    siloWriteMulti(root, MULTI_VAR, fname_base, "phi0", zs_to_sdomid, DB_QUADVAR);

    // Close root file
    DBClose(root);

    // Create a subdirectory to hold processor info
    mkdir(fname_base.c_str(), 0750);
  }

  // Sync up, so everyone sees the subdirectory
  MPI_Barrier(MPI_COMM_WORLD);

  // Create our processor file
  std::stringstream ss_proc;
  ss_proc << fname_base << "/rank_" << mpi_rank << ".silo";
  DBfile *proc = DBCreate(ss_proc.str().c_str(),
      DB_CLOBBER, DB_LOCAL, NULL, DB_HDF5);

  // Write out data for each subdomain
  int num_zone_sets = zs_to_sdomid.size();
  for(int sdom_idx = 0;sdom_idx < num_zone_sets;++ sdom_idx){
    int sdom_id = zs_to_sdomid[sdom_idx];
    Subdomain &sdom = subdomains[sdom_id];

    // Create a directory for the subdomain
    std::stringstream dirname;
    dirname << "/sdom" << sdom_id;
    DBMkDir(proc, dirname.str().c_str());

    // Set working directory
    DBSetDir(proc, dirname.str().c_str());

    // Write the mesh
    siloWriteRectMesh(proc, "mesh", sdom.nzones, sdom.zeros,
      &sdom.deltas[0][1], &sdom.deltas[1][1], &sdom.deltas[2][1]);


    // Write the material
    {
      int num_zones = sdom.num_zones;
      int num_mixed = sdom.mixed_material.size();
      int matnos[3] = {1, 2, 3};
      std::vector<int> matlist(num_zones, 0);
      std::vector<int> mix_next(num_mixed, 0);
      std::vector<int> mix_mat(num_mixed, 0);

      // setup matlist and mix_next arrays
      int last_z = -1;
      for(int m = 0;m < num_mixed;++ m){
        mix_mat[m] = sdom.mixed_material[m] + 1;
        int z = sdom.mixed_to_zones[m];
        if(matlist[z] == 0){
            matlist[z] = -(1+m);
        }
        // if we are still on the same zone, make sure the last mix points
        // here
        if(z == last_z){
          mix_next[m-1] = m+1;
        }
        last_z = z;
      }

      DBPutMaterial(proc, "material", "mesh", 3, matnos,
          &matlist[0], sdom.nzones, 3,
          &mix_next[0], &mix_mat[0], &sdom.mixed_to_zones[0], &sdom.mixed_fraction[0], num_mixed,
          DB_DOUBLE, NULL);
    }

    // Write phi0
    {

      int num_zones = sdom.num_zones;
      std::vector<double> phi0(num_zones);

      // extract phi0 from phi for the 0th group
      for(int z = 0;z < num_zones;++ z){
        phi0[z] = (*sdom.phi)(0,0,z);
      }

      DBPutQuadvar1(proc, "phi0", "mesh", &phi0[0],
          sdom.nzones, 3, NULL, 0, DB_DOUBLE, DB_ZONECENT, NULL);
    }
  }

  // Close processor file
  DBClose(proc);
}
#endif


