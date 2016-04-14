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

#ifndef KRIPKE_COMM_H__
#define KRIPKE_COMM_H__

#include<vector>
#include<mpi.h>

struct Grid_Data;
struct Subdomain;

class ParallelComm {
  public:
    explicit ParallelComm(Grid_Data *grid_data_ptr);
    virtual ~ParallelComm();

    // Adds a subdomain to the work queue
    virtual void addSubdomain(int sdom_id, Subdomain &sdom) = 0;

    // Checks if there are any outstanding subdomains to complete
    // false indicates all work is done, and all sends have completed
    virtual bool workRemaining(void);

    // Returns a vector of ready subdomains, and clears them from the ready queue
    virtual std::vector<int> readySubdomains(void) = 0;

    // Marks subdomains as complete, and performs downwind communication
    virtual void markComplete(int sdom_id) = 0;

  protected:
    static int computeTag(int mpi_rank, int sdom_id);
    static void computeRankSdom(int tag, int &mpi_rank, int &sdom_id);

    int findSubdomain(int sdom_id);
    Subdomain *dequeueSubdomain(int sdom_id);
    void postRecvs(int sdom_id, Subdomain &sdom);
    void postSends(Subdomain *sdom, double *buffers[3]);
    void testRecieves(void);
    void waitAllSends(void);
    std::vector<int> getReadyList(void);


    Grid_Data *grid_data;

    // These vectors contian the recieve requests
    std::vector<MPI_Request> recv_requests;
    std::vector<int> recv_subdomains;

    // These vectors have the subdomains, and the remaining dependencies
    std::vector<int> queue_sdom_ids;
    std::vector<Subdomain *> queue_subdomains;
    std::vector<int> queue_depends;

    // These vectors have the remaining send requests that are incomplete
    std::vector<MPI_Request> send_requests;
};


class SweepComm : public ParallelComm {
  public:
    explicit SweepComm(Grid_Data *data);
    virtual ~SweepComm();

    virtual void addSubdomain(int sdom_id, Subdomain &sdom);
    virtual bool workRemaining(void);
    virtual std::vector<int> readySubdomains(void);
    virtual void markComplete(int sdom_id);
};


class BlockJacobiComm : public ParallelComm {
  public:
    explicit BlockJacobiComm(Grid_Data *data);
    virtual ~BlockJacobiComm();

    void addSubdomain(int sdom_id, Subdomain &sdom);
    bool workRemaining(void);
    std::vector<int> readySubdomains(void);
    void markComplete(int sdom_id);

  private:
    bool posted_sends;
};



#endif
