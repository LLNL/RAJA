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

#include <Kripke.h>
#include <Kripke/Subdomain.h>
#include <Kripke/SubTVec.h>
#include <Kripke/ParallelComm.h>
#include <Kripke/Grid.h>
#include <vector>
#include <stdio.h>


/**
  Run solver iterations.
*/
int SweepSolver (Grid_Data *grid_data, bool block_jacobi)
{
  Kernel *kernel = grid_data->kernel;

  int mpi_rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);

  BLOCK_TIMER(grid_data->timing, Solve);


  // Loop over iterations
  double part_last = 0.0;
  for(int iter = 0;iter < grid_data->niter;++ iter){

    /*
     * Compute the RHS:  rhs = LPlus*S*L*psi + Q
     */

    // Discrete to Moments transformation (phi = L*psi)
    {
      BLOCK_TIMER(grid_data->timing, LTimes);
      kernel->LTimes(grid_data);
    }

    // Compute Scattering Source Term (psi_out = S*phi)
    {
      BLOCK_TIMER(grid_data->timing, Scattering);
      kernel->scattering(grid_data);
    }

    // Compute External Source Term (psi_out = psi_out + Q)
    {
      BLOCK_TIMER(grid_data->timing, Source);
      kernel->source(grid_data);
    }

    // Moments to Discrete transformation (rhs = LPlus*psi_out)
    {
      BLOCK_TIMER(grid_data->timing, LPlusTimes);
      kernel->LPlusTimes(grid_data);
    }

    /*
     * Sweep (psi = Hinv*rhs)
     */
    {
      BLOCK_TIMER(grid_data->timing, Sweep);

      if(true){
        // Create a list of all groups
        std::vector<int> sdom_list(grid_data->subdomains.size());
        for(int i = 0;i < grid_data->subdomains.size();++ i){
          sdom_list[i] = i;
        }

        // Sweep everything
        SweepSubdomains(sdom_list, grid_data, block_jacobi);
      }
      // This is the ARDRA version, doing each groupset sweep independently
      else{
        for(int group_set = 0;group_set < grid_data->num_group_sets;++ group_set){
          std::vector<int> sdom_list;
          // Add all subdomains for this groupset
          for(int s = 0;s < grid_data->subdomains.size();++ s){
            if(grid_data->subdomains[s].idx_group_set == group_set){
              sdom_list.push_back(s);
            }
          }

          // Sweep the groupset
          SweepSubdomains(sdom_list, grid_data, block_jacobi);
        }
      }
    }

    {
      //BLOCK_TIMER(grid_data->timing, ParticleEdit);
      //double part = grid_data->particleEdit();
      double part = -1.0;
      if(mpi_rank==0){
        printf("iter %d: particle count=%e, change=%e\n", iter, part, (part-part_last)/part);
      }
      part_last = part;
    }
  }
  return(0);
}



/**
  Perform full parallel sweep algorithm on subset of subdomains.
*/
void SweepSubdomains (std::vector<int> subdomain_list, Grid_Data *grid_data, bool block_jacobi)
{
  // Create a new sweep communicator object
  ParallelComm *comm = NULL;
  if(block_jacobi){
    comm = new BlockJacobiComm(grid_data);
  }
  else {
    comm = new SweepComm(grid_data);
  }

  // Add all subdomains in our list
  for(int i = 0;i < subdomain_list.size();++ i){
    int sdom_id = subdomain_list[i];
    comm->addSubdomain(sdom_id, grid_data->subdomains[sdom_id]);
  }

  /* Loop until we have finished all of our work */
  while(comm->workRemaining()){

    // Get a list of subdomains that have met dependencies
    std::vector<int> sdom_ready = comm->readySubdomains();
    int backlog = sdom_ready.size();

    // Run top of list
    if(backlog > 0){
      int sdom_id = sdom_ready[0];
      Subdomain &sdom = grid_data->subdomains[sdom_id];
      // Clear boundary conditions
      for(int dim = 0;dim < 3;++ dim){
        if(sdom.upwind[dim].subdomain_id == -1){
          sdom.plane_data[dim]->clear(0.0);
        }
      }
      {
        BLOCK_TIMER(grid_data->timing, Sweep_Kernel);
        // Perform subdomain sweep
        grid_data->kernel->sweep(grid_data, sdom_id);
      }

      // Mark as complete (and do any communication)
      comm->markComplete(sdom_id);
    }
  }

  delete comm;
}


