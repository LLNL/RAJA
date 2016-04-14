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

#include <Kripke/ParallelComm.h>
#include <Kripke/SubTVec.h>
#include <Kripke/Grid.h>

#include <fcntl.h>
#include <unistd.h>
#include <mpi.h>
#include <vector>
#include <stdio.h>


BlockJacobiComm::BlockJacobiComm(Grid_Data *data) : ParallelComm(data), posted_sends(false)
{

}

BlockJacobiComm::~BlockJacobiComm(){
}

/**
  Adds a subdomain to the work queue.
  Determines if upwind dependencies require communication, and posts appropirate Irecv's.
*/
void BlockJacobiComm::addSubdomain(int sdom_id, Subdomain &sdom){
  // Copy old flux data to send buffers
  for(int dim = 0;dim < 3;++ dim){
    int nelem = sdom.plane_data[dim]->elements;
    double const * KRESTRICT src = sdom.plane_data[dim]->ptr();
    double * KRESTRICT dst = sdom.old_plane_data[dim]->ptr();
    for(int i = 0;i < nelem;++ i){
      dst[i] = src[i];
    }
  }

  // post recieves
  postRecvs(sdom_id, sdom);

}

// Checks if there are any outstanding subdomains to complete
// false indicates all work is done, and all sends have completed
bool BlockJacobiComm::workRemaining(void){
  if(!posted_sends){
    // post sends for all queued subdomains
    for(int i = 0;i < queue_subdomains.size();++ i){
      Subdomain *sdom = queue_subdomains[i];

      // Send new downwind info for sweep
      double *buf[3] = {
        sdom->old_plane_data[0]->ptr(),
        sdom->old_plane_data[1]->ptr(),
        sdom->old_plane_data[2]->ptr()
      };

      postSends(sdom, buf);
    }
    posted_sends = true;
  }
  // Since we communicate fluxes before local sweeps, when we are
  // out of work, there is no further synchronization
  if(ParallelComm::workRemaining()){
    return true;
  }
  waitAllSends();

  return false;
}

/**
  Checks for incomming messages, and returns a list of ready subdomain id's
*/
std::vector<int> BlockJacobiComm::readySubdomains(void){
  testRecieves();

  // return list of any ready subdomains
  return getReadyList();
}



void BlockJacobiComm::markComplete(int sdom_id){
  // remove subdomain from work queue
  dequeueSubdomain(sdom_id);
}


