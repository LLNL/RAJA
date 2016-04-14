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


SweepComm::SweepComm(Grid_Data *data) : ParallelComm(data)
{

}

SweepComm::~SweepComm(){
}

/**
  Adds a subdomain to the work queue.
  Determines if upwind dependencies require communication, and posts appropirate Irecv's.
*/
void SweepComm::addSubdomain(int sdom_id, Subdomain &sdom){
  // Post recieves for upwind dependencies, and add to the queue
  postRecvs(sdom_id, sdom);
}


// Checks if there are any outstanding subdomains to complete
// false indicates all work is done, and all sends have completed
bool SweepComm::workRemaining(void){
  // If there are outstanding subdomains to process, return true
  if(ParallelComm::workRemaining()){
    return true;
  }

  // No more work, so make sure all of our sends have completed
  // before we continue
  waitAllSends();

  return false;
}


/**
  Checks for incomming messages, and returns a list of ready subdomain id's
*/
std::vector<int> SweepComm::readySubdomains(void){
  // check for incomming messages
  testRecieves();

  // build up a list of ready subdomains
  return getReadyList();
}


void SweepComm::markComplete(int sdom_id){
  // Get subdomain pointer and remove from work queue
  Subdomain *sdom = dequeueSubdomain(sdom_id);

  // Send new downwind info for sweep
  double *buf[3] = {
    sdom->plane_data[0]->ptr(),
    sdom->plane_data[1]->ptr(),
    sdom->plane_data[2]->ptr()
  };
  postSends(sdom, buf);
}

