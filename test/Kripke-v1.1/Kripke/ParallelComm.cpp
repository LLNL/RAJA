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

#include <Kripke/Grid.h>
#include <Kripke/Subdomain.h>
#include <Kripke/SubTVec.h>

ParallelComm::ParallelComm(Grid_Data *grid_data_ptr) :
  grid_data(grid_data_ptr)
{

}

ParallelComm::~ParallelComm(){

}

int ParallelComm::computeTag(int mpi_rank, int sdom_id){
  int mpi_size;
  MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);

  int tag = mpi_rank + mpi_size*sdom_id;

  return tag;
}

void ParallelComm::computeRankSdom(int tag, int &mpi_rank, int &sdom_id){
  int mpi_size;
  MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);

  mpi_rank = tag % mpi_size;
  sdom_id = tag / mpi_size;
}

/**
  Finds subdomain in the queue by its subdomain id.
*/
int ParallelComm::findSubdomain(int sdom_id){

  // find subdomain in queue
  int index;
  for(index = 0;index < queue_sdom_ids.size();++ index){
    if(queue_sdom_ids[index] == sdom_id){
      break;
    }
  }
  if(index == queue_sdom_ids.size()){
    printf("Cannot find subdomain id %d in work queue\n", sdom_id);
    MPI_Abort(MPI_COMM_WORLD, 1);
  }

  return index;
}


Subdomain *ParallelComm::dequeueSubdomain(int sdom_id){
  int index = findSubdomain(sdom_id);

  // Get subdomain pointer before removing it from queue
  Subdomain *sdom = queue_subdomains[index];

  // remove subdomain from queue
  queue_sdom_ids.erase(queue_sdom_ids.begin()+index);
  queue_subdomains.erase(queue_subdomains.begin()+index);
  queue_depends.erase(queue_depends.begin()+index);

  return sdom;
}

/**
  Adds a subdomain to the work queue.
  Determines if upwind dependencies require communication, and posts appropirate Irecv's.
  All recieves use the plane_data[] arrays as recieve buffers.
*/
void ParallelComm::postRecvs(int sdom_id, Subdomain &sdom){
  int mpi_rank, mpi_size;
  MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);

  // go thru each dimensions upwind neighbors, and add the dependencies
  int num_depends = 0;
  for(int dim = 0;dim < 3;++ dim){
    // If it's a boundary condition, skip it
    if(sdom.upwind[dim].mpi_rank < 0){
      continue;
    }

    // If it's an on-rank communication (from another subdomain)
    if(sdom.upwind[dim].mpi_rank == mpi_rank){
      // skip it, but track the dependency
      num_depends ++;
      continue;
    }

    // Add request to pending list
    recv_requests.push_back(MPI_Request());
    recv_subdomains.push_back(sdom_id);

    // compute the tag id of THIS subdomain (tags are always based on destination)
    int tag = computeTag(sdom.upwind[dim].mpi_rank, sdom.upwind[dim].subdomain_id);

    // Post the recieve
    MPI_Irecv(sdom.plane_data[dim]->ptr(), sdom.plane_data[dim]->elements, MPI_DOUBLE, sdom.upwind[dim].mpi_rank,
      tag, MPI_COMM_WORLD, &recv_requests[recv_requests.size()-1]);

    // increment number of dependencies
    num_depends ++;
  }

  // add subdomain to queue
  queue_sdom_ids.push_back(sdom_id);
  queue_subdomains.push_back(&sdom);
  queue_depends.push_back(num_depends);
}

void ParallelComm::postSends(Subdomain *sdom, double *src_buffers[3]){
  // post sends for downwind dependencies
  int mpi_rank, mpi_size;
  MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
  for(int dim = 0;dim < 3;++ dim){
    // If it's a boundary condition, skip it
    if(sdom->downwind[dim].mpi_rank < 0){
      continue;
    }

    // If it's an on-rank communication (to another subdomain)
    if(sdom->downwind[dim].mpi_rank == mpi_rank){
      // find the local subdomain in the queue, and decrement the counter
      for(int i = 0;i < queue_sdom_ids.size();++ i){
        if(queue_sdom_ids[i] == sdom->downwind[dim].subdomain_id){
          queue_depends[i] --;
          break;
        }
      }

      // copy the boundary condition data into the downwinds plane data
      Subdomain &sdom_downwind = grid_data->subdomains[sdom->downwind[dim].subdomain_id];
      sdom_downwind.plane_data[dim]->copy(*sdom->plane_data[dim]);
      int num_elem = sdom_downwind.plane_data[dim]->elements;
      //double const * KRESTRICT src_ptr = sdom->plane_data[dim]->ptr();
      double * KRESTRICT dst_ptr = sdom_downwind.plane_data[dim]->ptr();
      for(int i = 0;i < num_elem;++ i){
        dst_ptr[i] = src_buffers[dim][i];
      }
      continue;
    }

    // At this point, we know that we have to send an MPI message
    // Add request to send queue
    send_requests.push_back(MPI_Request());

    // compute the tag id of TARGET subdomain (tags are always based on destination)
    int tag = computeTag(mpi_rank, sdom->downwind[dim].subdomain_id);

    // Post the send
    MPI_Isend(src_buffers[dim], sdom->plane_data[dim]->elements, MPI_DOUBLE, sdom->downwind[dim].mpi_rank,
      tag, MPI_COMM_WORLD, &send_requests[send_requests.size()-1]);
  }
}


// Checks if there are any outstanding subdomains to complete
bool ParallelComm::workRemaining(void){
  return (recv_requests.size() > 0 || queue_subdomains.size() > 0);
}


// Blocks until all sends have completed, and flushes the send queues
void ParallelComm::waitAllSends(void){
  // Wait for all remaining sends to complete, then return false
  int num_sends = send_requests.size();
  if(num_sends > 0){
    std::vector<MPI_Status> status(num_sends);
    MPI_Waitall(num_sends, &send_requests[0], &status[0]);
    send_requests.clear();
  }
}

/**
  Checks for incomming messages, and does relevant bookkeeping.
*/
void ParallelComm::testRecieves(void){

  // Check for any recv requests that have completed
  int num_requests = recv_requests.size();
  bool done = false;
  while(!done && num_requests > 0){
    // Create array of status variables
    std::vector<MPI_Status> recv_status(num_requests);

    // Ask if either one or none of the recvs have completed?
    int index; // this will be the index of request that completed
    int complete_flag; // this is set to TRUE if somthing completed
    MPI_Testany(num_requests, &recv_requests[0], &index, &complete_flag, &recv_status[0]);

    if(complete_flag != 0){

      // get subdomain that this completed for
      int sdom_id = recv_subdomains[index];

      // remove the request from the list
      recv_requests.erase(recv_requests.begin()+index);
      recv_subdomains.erase(recv_subdomains.begin()+index);
      num_requests --;

      // decrement the dependency count for that subdomain
      for(int i = 0;i < queue_sdom_ids.size();++ i){
        if(queue_sdom_ids[i] == sdom_id){
          queue_depends[i] --;
          break;
        }
      }
    }
    else{
      done = true;
    }
  }
}


std::vector<int> ParallelComm::getReadyList(void){
  // build up a list of ready subdomains
  std::vector<int> ready;
  for(int i = 0;i < queue_depends.size();++ i){
    if(queue_depends[i] == 0){
      ready.push_back(queue_sdom_ids[i]);
    }
  }
  return ready;
}
