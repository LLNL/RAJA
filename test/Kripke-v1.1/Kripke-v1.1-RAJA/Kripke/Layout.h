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

#ifndef KRIPKE_LAYOUT_H__
#define KRIPKE_LAYOUT_H__

#include<algorithm>

// foreward decl
struct Input_Variables;

/**
  Describes a neighboring Subdomain using both mpi-rank and subdomin id
*/
struct Neighbor{
  int mpi_rank;     // Neighbors MPI rank, or -1 for boundary condition
  int subdomain_id; // Subdomain ID of neighbor
};



/**
   Describes relationships between MPI-ranks and subdomains.
   This is an interface, allowing different layout schemes to be implemented as derived types.
 */
class Layout {
  public:
    explicit Layout(Input_Variables *input_vars);
    virtual ~Layout();

    virtual int setIdToSubdomainId(int gs, int ds, int zs) const;
    virtual int subdomainIdToZoneSetDim(int sdom_id, int dim) const;
    virtual void subdomainIdToSetId(int sdom_id, int &gs, int &ds, int &zs) const;
    virtual Neighbor getNeighbor(int our_sdom_id, int dim, int dir) const = 0;
    virtual std::pair<double, double> getSpatialExtents(int sdom_id, int dim) const = 0;
    virtual int getNumZones(int sdom_id, int dim) const;

  protected:
    int num_group_sets;      // Number of group sets
    int num_direction_sets;  // Number of direction sets
    int num_zone_sets;       // Number of zone sets
    int num_zone_sets_dim[3];// Number of zone sets in each dimension

    int total_zones[3];      // Total number of zones in each dimension

    int num_procs[3];        // Number of MPI ranks in each dimensions
    int our_rank[3];         // Our mpi indices in xyz
};

class BlockLayout : public Layout {
  public:
    explicit BlockLayout(Input_Variables *input_vars);
    virtual ~BlockLayout();

    virtual Neighbor getNeighbor(int our_sdom_id, int dim, int dir) const;
    virtual std::pair<double, double> getSpatialExtents(int sdom_id, int dim) const;
};

class ScatterLayout : public Layout {
  public:
    explicit ScatterLayout(Input_Variables *input_vars);
    virtual ~ScatterLayout();

    virtual Neighbor getNeighbor(int our_sdom_id, int dim, int dir) const;
    virtual std::pair<double, double> getSpatialExtents(int sdom_id, int dim) const;
};


// Factory to create layout object
Layout *createLayout(Input_Variables *input_vars);

#endif
