/// \file
/// Functions to maintain link cell structures for fast pair finding.

#ifndef __LINK_CELLS_H_
#define __LINK_CELLS_H_

#include "mytype.h"

/// The maximum number of atoms that can be stored in a link cell.
#define MAXATOMS 64 

struct DomainSt;
struct AtomsSt;

/// Link cell data.  For convenience, we keep a copy of the localMin and
/// localMax coordinates that are also found in the DomainsSt.
typedef struct LinkCellSt
{
   int gridSize[3];     //!< number of boxes in each dimension on processor
   int nLocalBoxes;     //!< total number of local boxes on processor
   int nHaloBoxes;      //!< total number of remote halo/ghost boxes on processor
   int nTotalBoxes;     //!< total number of boxes on processor
                        //!< nLocalBoxes + nHaloBoxes
   real3 localMin;      //!< minimum local bounds on processor
   real3 localMax;      //!< maximum local bounds on processor
   real3 boxSize;       //!< size of box in each dimension
   real3 invBoxSize;    //!< inverse size of box in each dimension

   int* nAtoms;         //!< total number of atoms in each box
   int** nbrBoxes;      //!< neighbor boxes for each box
} LinkCell;

LinkCell* initLinkCells(const struct DomainSt* domain, real_t cutoff);
void destroyLinkCells(LinkCell** boxes);

int getNeighborBoxes(LinkCell* boxes, int iBox, int* nbrBoxes);
void putAtomInBox(LinkCell* boxes, struct AtomsSt* atoms,
                  const int gid, const int iType,
                  const real_t x,  const real_t y,  const real_t z,
                  const real_t px, const real_t py, const real_t pz);
int getBoxFromTuple(LinkCell* boxes, int x, int y, int z);

void moveAtom(LinkCell* boxes, struct AtomsSt* atoms, int iId, int iBox, int jBox);

/// Update link cell data structures when the atoms have moved.
void updateLinkCells(LinkCell* boxes, struct AtomsSt* atoms);

int maxOccupancy(LinkCell* boxes);


#endif
