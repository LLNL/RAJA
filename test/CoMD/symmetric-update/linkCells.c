/// \file
/// Functions to maintain link cell structures for fast pair finding.
///
/// In CoMD 1.1, atoms are stored in link cells.  Link cells are widely
/// used in classical MD to avoid an O(N^2) search for atoms that
/// interact.  Link cells are formed by subdividing the local spatial
/// domain with a Cartesian grid where the grid spacing in each
/// direction is at least as big as he potential's cutoff distance.
/// Because atoms don't interact beyond the potential cutoff, for an
/// atom iAtom in any given link cell, we can be certain that all atoms
/// that interact with iAtom are contained in the same link cell, or one
/// of the 26 neighboring link cells.
/// 
/// CoMD chooses the link cell size (boxSize) on each axis to be the
/// shortest possible distance, longer than cutoff, such that the local
/// domain size divided by boxSize is an integer.  I.e., the link cells
/// are commensurate with with the local domain size.  While this does
/// not result in the smallest possible link cells, it does allow us to
/// keep a strict separation between the link cells that are entirely
/// inside the local domain and those that represent halo regions.
///
/// The number of local link cells in each direction is stored in
/// gridSize.  Local link cells have 3D grid coordinates (ix, iy, iz)
/// where ix, iy, and iz can range from 0 to gridSize[iAxis]-1,
/// whiere iAxis is 0 for x, 1 for y and 2 for the z direction.  The
/// number of local link cells is thus nLocalBoxes =
/// gridSize[0]*gridSize[1]*gridSize[2].
///
/// The local link cells are surrounded by one complete shell of halo
/// link cells.  The halo cells provide temporary storage for halo or
/// "ghost" atoms that belong to other tasks, but whose coordinates are
/// needed locally to complete the force calculation.  Halo link cells
/// have at least one coordinate with a value of either -1 or
/// gridSize[iAxis].
///
/// Because CoMD stores data in ordinary 1D C arrays, a mapping is
/// needed from the 3D grid coords to a 1D array index.  For the local
/// cells we use the conventional mapping ix + iy*nx + iz*nx*ny.  This
/// keeps all of the local cells in a contiguous region of memory
/// starting from the beginning of any relevant array and makes it easy
/// to iterate the local cells in a single loop.  Halo cells are mapped
/// differently.  After the local cells, the two planes of link cells
/// that are face neighbors with local cells across the -x or +x axis
/// are next.  These are followed by face neighbors across the -y and +y
/// axis (including cells that are y-face neighbors with an x-plane of
/// halo cells), followed by all remaining cells in the -z and +z planes
/// of halo cells.  The total number of link cells (on each rank) is
/// nTotalBoxes.
///
/// Data storage arrays that are used in association with link cells
/// should be allocated to store nTotalBoxes*MAXATOMS items.  Data for
/// the first atom in linkCell iBox is stored at index iBox*MAXATOMS.
/// Data for subsequent atoms in the same link cell are stored
/// sequentially, and the number of atoms in link cell iBox is
/// nAtoms[iBox].
///
/// \see getBoxFromTuple is the 3D->1D mapping for link cell indices.
/// \see getTuple is the 1D->3D mapping
///
/// \param [in] cutoff The cutoff distance of the potential.

#include "linkCells.h"

#include <stdio.h>
#include <string.h>
#include <assert.h>
#include <math.h>

#include "parallel.h"
#include "memUtils.h"
#include "decomposition.h"
#include "performanceTimers.h"
#include "CoMDTypes.h"

#define   MIN(A,B) ((A) < (B) ? (A) : (B))
#define   MAX(A,B) ((A) > (B) ? (A) : (B))

static void copyAtom(LinkCell* boxes, Atoms* atoms, int iAtom, int iBox, int jAtom, int jBox);
static int getBoxFromCoord(LinkCell* boxes, real_t rr[3]);
static void emptyHaloCells(LinkCell* boxes);
static void getTuple(LinkCell* boxes, int iBox, int* ixp, int* iyp, int* izp);

extern "C" int boxSort(const void *v1, const void *v2)
{
   return *(static_cast<const int *>(v2)) - *(static_cast<const int *>(v1)) ;
}

LinkCell* initLinkCells(const Domain* domain, real_t cutoff)
{
   assert(domain);
   LinkCell* ll = (LinkCell*)comdMalloc(sizeof(LinkCell));

   for (int i = 0; i < 3; i++)
   {
      ll->localMin[i] = domain->localMin[i];
      ll->localMax[i] = domain->localMax[i];
      ll->gridSize[i] = domain->localExtent[i] / cutoff; // local number of boxes
      ll->boxSize[i] = domain->localExtent[i] / ((real_t) ll->gridSize[i]);
      ll->invBoxSize[i] = 1.0/ll->boxSize[i];
   }

   ll->nLocalBoxes = ll->gridSize[0] * ll->gridSize[1] * ll->gridSize[2];
   
   ll->nHaloBoxes = 2 * ((ll->gridSize[0] + 2) *
                         (ll->gridSize[1] + ll->gridSize[2] + 2) +
                         (ll->gridSize[1] * ll->gridSize[2]));

   ll->nTotalBoxes = ll->nLocalBoxes + ll->nHaloBoxes;
   
   ll->nAtoms = (int *)comdMalloc(ll->nTotalBoxes*sizeof(int));
   for (int iBox=0; iBox<ll->nTotalBoxes; ++iBox)
      ll->nAtoms[iBox] = 0;

   assert ( (ll->gridSize[0] >= 2) && (ll->gridSize[1] >= 2) && (ll->gridSize[2] >= 2) );

   // Added creating neighbors once
   ll->nbrBoxes = (int **)comdMalloc(ll->nTotalBoxes*sizeof(int*));
   for (int iBox=0; iBox<ll->nTotalBoxes; ++iBox)
   {
      ll->nbrBoxes[iBox] = (int *)comdMalloc(27*sizeof(int));
   }

   for (int iBox=0; iBox<ll->nLocalBoxes; ++iBox)
   {
      int nNbrBoxes = getNeighborBoxes(ll, iBox, ll->nbrBoxes[iBox]);
      qsort(ll->nbrBoxes[iBox], 27, sizeof(int), boxSort) ;
   }

   return ll;
}

void destroyLinkCells(LinkCell** boxes)
{
   if (! boxes) return;
   if (! *boxes) return;

   comdFree((*boxes)->nAtoms);
   comdFree(*boxes);
   *boxes = NULL;

   return;
}

/// \details
/// Populates the nbrBoxes array with the 27 boxes that are adjacent to
/// iBox.  The count is 27 instead of 26 because iBox is included in the
/// list (as neighbor 13).  Caller is responsible to alloc and free
/// nbrBoxes.
/// \return The number of nbr boxes (always 27 in this implementation).
int getNeighborBoxes(LinkCell* boxes, int iBox, int* nbrBoxes)
{
   int ix, iy, iz;
   getTuple(boxes, iBox, &ix, &iy, &iz);
   
   int count = 0;
   for (int i=ix-1; i<=ix+1; i++)
      for (int j=iy-1; j<=iy+1; j++)
         for (int k=iz-1; k<=iz+1; k++)
            nbrBoxes[count++] = getBoxFromTuple(boxes,i,j,k);
   
   return count;
}

/// \details
/// Finds the appropriate link cell for an atom based on the spatial
/// coordinates and stores data in that link cell.
/// \param [in] gid   The global of the atom.
/// \param [in] iType The species index of the atom.
/// \param [in] x     The x-coordinate of the atom.
/// \param [in] y     The y-coordinate of the atom.
/// \param [in] z     The z-coordinate of the atom.
/// \param [in] px    The x-component of the atom's momentum.
/// \param [in] py    The y-component of the atom's momentum.
/// \param [in] pz    The z-component of the atom's momentum.
void putAtomInBox(LinkCell* boxes, Atoms* atoms,
                  const int gid, const int iType,
                  const real_t x,  const real_t y,  const real_t z,
                  const real_t px, const real_t py, const real_t pz)
{
   real_t xyz[3] = {x,y,z};
   
   // Find correct box.
   int iBox = getBoxFromCoord(boxes, xyz);
   int iOff = iBox*MAXATOMS;
   iOff += boxes->nAtoms[iBox];
   
   // assign values to array elements
   if (iBox < boxes->nLocalBoxes)
      atoms->nLocal++;
   boxes->nAtoms[iBox]++;
   atoms->gid[iOff] = gid;
   atoms->iSpecies[iOff] = iType;
   
   atoms->r[iOff][0] = x;
   atoms->r[iOff][1] = y;
   atoms->r[iOff][2] = z;
   
   atoms->p[iOff][0] = px;
   atoms->p[iOff][1] = py;
   atoms->p[iOff][2] = pz;
}

/// Calculates the link cell index from the grid coords.  The valid
/// coordinate range in direction ii is [-1, gridSize[ii]].  Any
/// coordinate that involves a -1 or gridSize[ii] is a halo link cell.
/// Because of the order in which the local and halo link cells are
/// stored the indices of the halo cells are special cases.
/// \see initLinkCells for an explanation of storage order.
int getBoxFromTuple(LinkCell* boxes, int ix, int iy, int iz)
{
   int iBox = 0;
   const int* gridSize = boxes->gridSize; // alias
   
   // Halo in Z+
   if (iz == gridSize[2])
   {
      iBox = boxes->nLocalBoxes + 2*gridSize[2]*gridSize[1] + 2*gridSize[2]*(gridSize[0]+2) +
         (gridSize[0]+2)*(gridSize[1]+2) + (gridSize[0]+2)*(iy+1) + (ix+1);
   }
   // Halo in Z-
   else if (iz == -1)
   {
      iBox = boxes->nLocalBoxes + 2*gridSize[2]*gridSize[1] + 2*gridSize[2]*(gridSize[0]+2) +
         (gridSize[0]+2)*(iy+1) + (ix+1);
   }
   // Halo in Y+
   else if (iy == gridSize[1])
   {
      iBox = boxes->nLocalBoxes + 2*gridSize[2]*gridSize[1] + gridSize[2]*(gridSize[0]+2) +
         (gridSize[0]+2)*iz + (ix+1);
   }
   // Halo in Y-
   else if (iy == -1)
   {
      iBox = boxes->nLocalBoxes + 2*gridSize[2]*gridSize[1] + iz*(gridSize[0]+2) + (ix+1);
   }
   // Halo in X+
   else if (ix == gridSize[0])
   {
      iBox = boxes->nLocalBoxes + gridSize[1]*gridSize[2] + iz*gridSize[1] + iy;
   }
   // Halo in X-
   else if (ix == -1)
   {
      iBox = boxes->nLocalBoxes + iz*gridSize[1] + iy;
   }
   // local link celll.
   else
   {
      iBox = ix + gridSize[0]*iy + gridSize[0]*gridSize[1]*iz;
   }
   assert(iBox >= 0);
   assert(iBox < boxes->nTotalBoxes);

   return iBox;
}

/// Move an atom from one link cell to another.
/// \param iId [in]  The index with box iBox of the atom to be moved.
/// \param iBox [in] The index of the link cell the particle is moving from.
/// \param jBox [in] The index of the link cell the particle is moving to.
void moveAtom(LinkCell* boxes, Atoms* atoms, int iId, int iBox, int jBox)
{
   int nj = boxes->nAtoms[jBox];
   copyAtom(boxes, atoms, iId, iBox, nj, jBox);
   boxes->nAtoms[jBox]++;

   assert(boxes->nAtoms[jBox] < MAXATOMS);

   boxes->nAtoms[iBox]--;
   int ni = boxes->nAtoms[iBox];
   if (ni) copyAtom(boxes, atoms, ni, iBox, iId, iBox);

   if (jBox > boxes->nLocalBoxes)
      --atoms->nLocal;
   
   return;
}

/// \details
/// This is the first step in returning data structures to a consistent
/// state after the atoms move each time step.  First we discard all
/// atoms in the halo link cells.  These are all atoms that are
/// currently stored on other ranks and so any information we have about
/// them is stale.  Next, we move any atoms that have crossed link cell
/// boundaries into their new link cells.  It is likely that some atoms
/// will be moved into halo link cells.  Since we have deleted halo
/// atoms from other tasks, it is clear that any atoms that are in halo
/// cells at the end of this routine have just transitioned from local
/// to halo atoms.  Such atom must be sent to other tasks by a halo
/// exchange to avoid being lost.
/// \see redistributeAtoms
void updateLinkCells(LinkCell* boxes, Atoms* atoms)
{
   emptyHaloCells(boxes);
   
   for (int iBox=0; iBox<boxes->nLocalBoxes; ++iBox)
   {
      int iOff = iBox*MAXATOMS;
      int ii=0;
      while (ii < boxes->nAtoms[iBox])
      {
         int jBox = getBoxFromCoord(boxes, atoms->r[iOff+ii]);
         if (jBox != iBox)
            moveAtom(boxes, atoms, ii, iBox, jBox);
         else
            ++ii;
      }
   }
}

/// \return The largest number of atoms in any link cell.
int maxOccupancy(LinkCell* boxes)
{
   int localMax = 0;
   for (int ii=0; ii<boxes->nLocalBoxes; ++ii)
      localMax = MAX(localMax, boxes->nAtoms[ii]);

   int globalMax;

   startTimer(commReduceTimer);
   maxIntParallel(&localMax, &globalMax, 1);
   stopTimer(commReduceTimer);

   return globalMax;
}

/// Copy atom iAtom in link cell iBox to atom jAtom in link cell jBox.
/// Any data at jAtom, jBox is overwritten.  This routine can be used to
/// re-order atoms within a link cell.
void copyAtom(LinkCell* boxes, Atoms* atoms, int iAtom, int iBox, int jAtom, int jBox)
{
   const int iOff = MAXATOMS*iBox+iAtom;
   const int jOff = MAXATOMS*jBox+jAtom;
   atoms->gid[jOff] = atoms->gid[iOff];
   atoms->iSpecies[jOff] = atoms->iSpecies[iOff];
   memcpy(atoms->r[jOff], atoms->r[iOff], sizeof(real3));
   memcpy(atoms->p[jOff], atoms->p[iOff], sizeof(real3));
   memcpy(atoms->f[jOff], atoms->f[iOff], sizeof(real3));
   memcpy(atoms->U+jOff,  atoms->U+iOff,  sizeof(real_t));
}

/// Get the index of the link cell that contains the specified
/// coordinate.  This can be either a halo or a local link cell.
///
/// Because the rank ownership of an atom is strictly determined by the
/// atom's position, we need to take care that all ranks will agree which
/// rank owns an atom.  The conditionals at the end of this function are
/// special care to ensure that all ranks make compatible link cell
/// assignments for atoms that are near a link cell boundaries.  If no
/// ranks claim an atom in a local cell it will be lost.  If multiple
/// ranks claim an atom it will be duplicated.
int getBoxFromCoord(LinkCell* boxes, real_t rr[3])
{
   const real_t* localMin = boxes->localMin; // alias
   const real_t* localMax = boxes->localMax; // alias
   const int*    gridSize = boxes->gridSize; // alias
   int ix = (int)(floor((rr[0] - localMin[0])*boxes->invBoxSize[0]));
   int iy = (int)(floor((rr[1] - localMin[1])*boxes->invBoxSize[1]));
   int iz = (int)(floor((rr[2] - localMin[2])*boxes->invBoxSize[2]));


   // For each axis, if we are inside the local domain, make sure we get
   // a local link cell.  Otherwise, make sure we get a halo link cell.
   if (rr[0] < localMax[0]) 
   {
      if (ix == gridSize[0]) ix = gridSize[0] - 1;
   }
   else
      ix = gridSize[0]; // assign to halo cell
   if (rr[1] < localMax[1])
   {
      if (iy == gridSize[1]) iy = gridSize[1] - 1;
   }
   else
      iy = gridSize[1];
   if (rr[2] < localMax[2])
   {
      if (iz == gridSize[2]) iz = gridSize[2] - 1;
   }
   else
      iz = gridSize[2];
   
   return getBoxFromTuple(boxes, ix, iy, iz);
}

/// Set the number of atoms to zero in all halo link cells.
void emptyHaloCells(LinkCell* boxes)
{
   for (int ii=boxes->nLocalBoxes; ii<boxes->nTotalBoxes; ++ii)
      boxes->nAtoms[ii] = 0;
}

/// Get the grid coordinates of the link cell with index iBox.  Local
/// cells are easy as they use a standard 1D->3D mapping.  Halo cell are
/// special cases.
/// \see initLinkCells for information on link cell order.
/// \param [in]  iBox Index to link cell for which tuple is needed.
/// \param [out] ixp  x grid coord of link cell.
/// \param [out] iyp  y grid coord of link cell.
/// \param [out] izp  z grid coord of link cell.
void getTuple(LinkCell* boxes, int iBox, int* ixp, int* iyp, int* izp)
{
   int ix, iy, iz;
   const int* gridSize = boxes->gridSize; // alias
   
   // If a local box
   if( iBox < boxes->nLocalBoxes)
   {
      ix = iBox % gridSize[0];
      iBox /= gridSize[0];
      iy = iBox % gridSize[1];
      iz = iBox / gridSize[1];
   }
   // It's a halo box
   else 
   {
      int ink;
      ink = iBox - boxes->nLocalBoxes;
      if (ink < 2*gridSize[1]*gridSize[2])
      {
         if (ink < gridSize[1]*gridSize[2]) 
         {
            ix = 0;
         }
         else 
         {
            ink -= gridSize[1]*gridSize[2];
            ix = gridSize[0] + 1;
         }
         iy = 1 + ink % gridSize[1];
         iz = 1 + ink / gridSize[1];
      }
      else if (ink < (2 * gridSize[2] * (gridSize[1] + gridSize[0] + 2))) 
      {
         ink -= 2 * gridSize[2] * gridSize[1];
         if (ink < ((gridSize[0] + 2) *gridSize[2])) 
         {
            iy = 0;
         }
         else 
         {
            ink -= (gridSize[0] + 2) * gridSize[2];
            iy = gridSize[1] + 1;
         }
         ix = ink % (gridSize[0] + 2);
         iz = 1 + ink / (gridSize[0] + 2);
      }
      else 
      {
         ink -= 2 * gridSize[2] * (gridSize[1] + gridSize[0] + 2);
         if (ink < ((gridSize[0] + 2) * (gridSize[1] + 2))) 
         {
            iz = 0;
         }
         else 
         {
            ink -= (gridSize[0] + 2) * (gridSize[1] + 2);
            iz = gridSize[2] + 1;
         }
         ix = ink % (gridSize[0] + 2);
         iy = ink / (gridSize[0] + 2);
      }
      
      // Calculated as off by 1
      ix--;
      iy--;
      iz--;
   }
   
   *ixp = ix;
   *iyp = iy;
   *izp = iz;
}

static void getlimit(int span, int parts, int index, int *begin, int *end)
{
    int chunk = span / parts ;
    int rem   = span % parts ;
    *begin = index*chunk + ((index < rem) ? index : rem) ;
    *end   = (index+1)*chunk + (((index+1) < rem) ? (index+1) : rem) ;
}

static void WavefrontCorner(int nx, int ny, int nz, int xs, int ys, int zs,
               LinkCell *boxes, int *outlist, int stride, int pad)
{
   ++nx ;
   int max = nx+ny+nz - 2 ;
   int index = 0 ;

   for (int plane = 0; plane<max ; ++plane) {
      for (int z = 0; z<plane; ++z) {
         for (int y = 0; y < (plane - z); ++y) {
            int x = plane - y - z ;
            if (x<(nx-pad) && y<(ny-pad) && z<(nz-pad)) {
               outlist[index*stride] = getBoxFromTuple(boxes, xs+x-1,  ys+y, zs+z) ;
               ++index ;
            }
         }
      }
   }

   if (pad != 0) {
      for (int plane = 0; plane<max ; ++plane) {
         for (int z = 0; z<plane; ++z) {
            for (int y = 0; y < (plane - z); ++y) {
               int x = plane - y - z ;
               if ( (x>=(nx-pad) || y>=(ny-pad) || z>=(nz-pad)) &&
                   x<nx && y<ny && z<nz) {
                  outlist[index*stride] = getBoxFromTuple(boxes, xs+x-1,  ys+y, zs+z) ;
                  ++index ;
               }
            }
         }
      }
   }
}

static void WavefrontEdge(int nx, int ny, int nz, int xs, int ys, int zs,
                          LinkCell *boxes, int *outlist, int stride, int pad)
{
   ++ny ;
   int max = ny+nz - 1 ;
   int index = 0 ;

   for (int plane = 0; plane<max ; ++plane) {
      for (int z = 0; z<plane; ++z) {
         int y = plane - z ;
         for (int x=0; x<nx; ++x) {
            if (x<(nx-pad) && y<(ny-pad) && z<(nz-pad)) {
               outlist[index*stride] = getBoxFromTuple(boxes, xs+x,  ys+y-1, zs+z) ;
               ++index ;
            }
         }
      }
   }

   if (pad != 0) {
      for (int plane = 0; plane<max ; ++plane) {
         for (int z = 0; z<plane; ++z) {
            int y = plane - z ;
            for (int x = 0; x < nx; ++x) {
               if ( (x>=(nx-pad) || y>=(ny-pad) || z>=(nz-pad)) &&
                   x<nx && y<ny && z<nz) {
                  outlist[index*stride] = getBoxFromTuple(boxes, xs+x,  ys+y-1, zs+z) ;
                  ++index ;
               }
            }
         }
      }
   }
}

static void WavefrontPlane(int nx, int ny, int nz, int xs, int ys, int zs,
                           LinkCell *boxes, int *outlist, int stride, int pad)
{
   int index = 0 ;

   for (int z = 0; z<nz ; ++z) {
      for (int y = 0; y<ny; ++y) {
         for (int x=0; x<nx; ++x) {
            if (x<(nx-pad) && y<(ny-pad) && z<(nz-pad)) {
               outlist[index*stride] = getBoxFromTuple(boxes, xs+x,  ys+y, zs+z) ;
               ++index ;
            }
         }
      }
   }

   if (pad != 0) {
      for (int z = 0; z<nz ; ++z) {
         for (int y = 0; z<ny; ++y) {
            for (int x = 0; x < nx; ++x) {
               if ( (x>=(nx-pad) || y>=(ny-pad) || z>=(nz-pad)) &&
                   x<nx && y<ny && z<nz) {
                  outlist[index*stride] = getBoxFromTuple(boxes, xs+x,  ys+y, zs+z) ;
                  ++index ;
               }
            }
         }
      }
   }
}

RAJA::IndexSet *BuildWavefront(LinkCell *boxes,
                               RAJA::IndexSet *master,
                               int nx, int ny, int nz,
                               int numThreads, int pad)
{
   RAJA::IndexSet *slave ;
   int px, py, pz ;
   int maxX = 0, maxY = 0, maxZ = 0 ;

   switch (numThreads) {

      case 1:
         // return clone of master ;
         px = 1 ; py = 1 ; pz = 1 ;
         break ;

      case 2:
         px = 1 ; py = 1 ; pz = 2 ;
         break ;

      case 4:
         px = 1 ; py = 2 ; pz = 2 ;
         break ;

      case 8:
         px = 1 ; py = 2 ; pz = 4 ;
         break ;

      case 16:
         px = 2 ; py = 2 ; pz = 4 ;
         break ;
   }

   for (int z=0; z<pz; ++z) {
      for (int y=0; y<py; ++y) {
         for (int x=0; x<px; ++x) {
            int dx, dy, dz ;
            int xs, xe ;
            int ys, ye ;
            int zs, ze ;
            getlimit(nx, px, x, &xs, &xe) ;
            getlimit(ny, py, y, &ys, &ye) ;
            getlimit(nz, pz, z, &zs, &ze) ;
            dx = xe-xs ;
            dy = ye-ys ;
            dz = ze-zs ;
            if (dx > maxX) maxX = dx ;
            if (dy > maxY) maxY = dy ;
            if (dz > maxZ) maxZ = dz ;
         }
      }
   }

   int *segList = (int *) comdMalloc(numThreads*maxX*maxY*maxZ*sizeof(int)) ;
   for (int i=0 ; i<maxX*maxY*maxZ; ++i) {
      segList[i] = -1 ; /* initialize to empty segments */
   }

   int tid = 0 ;
   for (int z=0; z<pz; ++z) {
      for (int y=0; y<py; ++y) {
         for (int x=0; x<px; ++x) {
            int xs, xe ;
            int ys, ye ;
            int zs, ze ;
            getlimit(nx, px, x, &xs, &xe) ;
            getlimit(ny, py, y, &ys, &ye) ;
            getlimit(nz, pz, z, &zs, &ze) ;

            /* assume x is most dominant (stride-1) and z is least dominant */
            if (px != 1 && py != 1 && pz != 1) {
               WavefrontCorner(xe-xs, ye-ys, ze-zs,
                               xs, ys, zs,
                               boxes, &segList[tid], numThreads, pad) ;
            }
            else if (py != 1 && pz != 1) {
               WavefrontEdge(xe-xs, ye-ys, ze-zs,
                             xs, ys, zs,
                             boxes, &segList[tid], numThreads, pad) ;
            }
            else {
               WavefrontPlane(xe-xs, ye-ys, ze-zs,
                              xs, ys, zs,
                              boxes, &segList[tid], numThreads, pad) ;
            }

            ++tid ;
         }
      }
   }

   slave = master->createView(segList, numThreads*maxX*maxY*maxZ) ;
   comdFree(segList) ;

   return slave ;
}

