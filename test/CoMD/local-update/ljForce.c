/// \file
/// Computes forces for the 12-6 Lennard Jones (LJ) potential.
///
/// The Lennard-Jones model is not a good representation for the
/// bonding in copper, its use has been limited to constant volume
/// simulations where the embedding energy contribution to the cohesive
/// energy is not included in the two-body potential
///
/// The parameters here are taken from Wolf and Phillpot and fit to the
/// room temperature lattice constant and the bulk melt temperature
/// Ref: D. Wolf and S.Yip eds. Materials Interfaces (Chapman & Hall
///      1992) Page 230.
///
/// Notes on LJ:
///
/// http://en.wikipedia.org/wiki/Lennard_Jones_potential
///
/// The total inter-atomic potential energy in the LJ model is:
///
/// \f[
///   E_{tot} = \sum_{ij} U_{LJ}(r_{ij})
/// \f]
/// \f[
///   U_{LJ}(r_{ij}) = 4 \epsilon
///           \left\{ \left(\frac{\sigma}{r_{ij}}\right)^{12}
///           - \left(\frac{\sigma}{r_{ij}}\right)^6 \right\}
/// \f]
///
/// where \f$\epsilon\f$ and \f$\sigma\f$ are the material parameters in the potential.
///    - \f$\epsilon\f$ = well depth
///    - \f$\sigma\f$   = hard sphere diameter
///
///  To limit the interation range, the LJ potential is typically
///  truncated to zero at some cutoff distance. A common choice for the
///  cutoff distance is 2.5 * \f$\sigma\f$.
///  This implementation can optionally shift the potential slightly
///  upward so the value of the potential is zero at the cuotff
///  distance.  This shift has no effect on the particle dynamics.
///
///
/// The force on atom i is given by
///
/// \f[
///   F_i = -\nabla_i \sum_{jk} U_{LJ}(r_{jk})
/// \f]
///
/// where the subsrcipt i on the gradient operator indicates that the
/// derivatives are taken with respect to the coordinates of atom i.
/// Liberal use of the chain rule leads to the expression
///
/// \f{eqnarray*}{
///   F_i &=& - \sum_j U'_{LJ}(r_{ij})\hat{r}_{ij}\\
///       &=& \sum_j 24 \frac{\epsilon}{r_{ij}} \left\{ 2 \left(\frac{\sigma}{r_{ij}}\right)^{12}
///               - \left(\frac{\sigma}{r_{ij}}\right)^6 \right\} \hat{r}_{ij}
/// \f}
///
/// where \f$\hat{r}_{ij}\f$ is a unit vector in the direction from atom
/// i to atom j.
/// 
///

#include "ljForce.h"

#include <stdlib.h>
#include <assert.h>
#include <string.h>
#include <omp.h>

#include "constants.h"
#include "mytype.h"
#include "parallel.h"
#include "linkCells.h"
#include "memUtils.h"
#include "CoMDTypes.h"

#define POT_SHIFT 1.0

/// Derived struct for a Lennard Jones potential.
/// Polymorphic with BasePotential.
/// \see BasePotential
typedef struct LjPotentialSt
{
   real_t cutoff;          //!< potential cutoff distance in Angstroms
   real_t mass;            //!< mass of atoms in intenal units
   real_t lat;             //!< lattice spacing (angs) of unit cell
   char latticeType[8];    //!< lattice type, e.g. FCC, BCC, etc.
   char  name[3];	   //!< element name
   int	 atomicNo;	   //!< atomic number  
   int  (*force)(SimFlat* s); //!< function pointer to force routine
   void (*print)(FILE* file, BasePotential* pot);
   void (*destroy)(BasePotential** pot); //!< destruction of the potential
   real_t sigma;
   real_t epsilon;
} LjPotential;

static int ljForce(SimFlat* s);
static void ljPrint(FILE* file, BasePotential* pot);

void ljDestroy(BasePotential** inppot)
{
   if ( ! inppot ) return;
   LjPotential* pot = (LjPotential*)(*inppot);
   if ( ! pot ) return;
   comdFree(pot);
   *inppot = NULL;

   return;
}

/// Initialize an Lennard Jones potential for Copper.
BasePotential* initLjPot(void)
{
   LjPotential *pot = (LjPotential*)comdMalloc(sizeof(LjPotential));
   pot->force = ljForce;
   pot->print = ljPrint;
   pot->destroy = ljDestroy;
   pot->sigma = 2.315;	                  // Angstrom
   pot->epsilon = 0.167;                  // eV
   pot->mass = 63.55 * amuToInternalMass; // Atomic Mass Units (amu)

   pot->lat = 3.615;                      // Equilibrium lattice const in Angs
   strcpy(pot->latticeType, "FCC");       // lattice type, i.e. FCC, BCC, etc.
   pot->cutoff = 2.5*pot->sigma;          // Potential cutoff in Angs

   strcpy(pot->name, "Cu");
   pot->atomicNo = 29;

   return (BasePotential*) pot;
}

void ljPrint(FILE* file, BasePotential* pot)
{
   LjPotential* ljPot = (LjPotential*) pot;
   fprintf(file, "  Potential type   : Lennard-Jones\n");
   fprintf(file, "  Species name     : %s\n", ljPot->name);
   fprintf(file, "  Atomic number    : %d\n", ljPot->atomicNo);
   fprintf(file, "  Mass             : " FMT1 " amu\n", ljPot->mass / amuToInternalMass); // print in amu
   fprintf(file, "  Lattice Type     : %s\n", ljPot->latticeType);
   fprintf(file, "  Lattice spacing  : " FMT1 " Angstroms\n", ljPot->lat);
   fprintf(file, "  Cutoff           : " FMT1 " Angstroms\n", ljPot->cutoff);
   fprintf(file, "  Epsilon          : " FMT1 " eV\n", ljPot->epsilon);
   fprintf(file, "  Sigma            : " FMT1 " Angstroms\n", ljPot->sigma);
}

int ljForce(SimFlat* s)
{
// static int pcount = 0 ;
// ReduceSum<RAJA::omp_reduce, int>  pairsChecked(0) ;
   LjPotential* pot = (LjPotential *) s->pot;
   real_t sigma = pot->sigma;
   real_t epsilon = pot->epsilon;
   real_t rCut = pot->cutoff;
   real_t rCut2 = rCut*rCut;

   // zero forces and energy
   RAJA::ReduceSum<RAJA::omp_reduce, real_t> ePot(0.0);
   s->ePotential = 0.0;
   
   RAJA::forall<atomWork>(*s->isTotal, [=] (int ii) {
      zeroReal3(s->atoms->f[ii]);
      s->atoms->U[ii] = 0.;
   } ) ;
   
   real_t s6 = sigma*sigma*sigma*sigma*sigma*sigma;

   real_t rCut6 = s6 / (rCut2*rCut2*rCut2);
   real_t eShift = POT_SHIFT * rCut6 * (rCut6 - 1.0);

   {
   // loop over local boxes
   // lock-free schedule and programming model are embedded in the IndexSets
   RAJA::forall_segments<task_graph_policy>(*s->isLocal, 
   [&] (RAJA::IndexSet *iBox) {
      // real_t ePotLocal = 0.0;
//    int localPairs = 0 ;

      RAJA::IndexSet *iBoxNeighbors =
           static_cast<RAJA::IndexSet *>(iBox->getSegment(0)->getPrivate()) ;

      RAJA::forall<linkCellWork>(*iBoxNeighbors, [&] (int jOff) {
         RAJA::forall<linkCellWork>(*iBox, [&] (int iOff) {
            real3 dr;
            real_t r2 = 0.0;
            real3_ptr r =  s->atoms->r ;
            for (int m=0; m<3; m++)
            {
#if 0
               dr[m] = s->atoms->r[iOff][m]-s->atoms->r[jOff][m];
#else
               dr[m] = r[iOff][m] - r[jOff][m];
#endif
               r2 += dr[m]*dr[m];
            }
            if ( r2 <= rCut2 && r2 > 0.0)
            {

               // Important note:
               // from this point on r actually refers to 1.0/r
               real_ptr U = s->atoms->U ;
               real3_ptr f = s->atoms->f ;
               r2 = 1.0/r2;
               real_t r6 = s6 * (r2*r2*r2);
               real_t eLocal = r6 * (r6 - 1.0) - eShift;
#if 0
               s->atoms->U[iOff] += 0.5*eLocal;
#else
               U[iOff] += 0.5*eLocal;
#endif
               // ePotLocal += 0.5*eLocal;
               ePot += 0.5*eLocal;

               // different formulation to avoid sqrt computation
               real_t fr = - 4.0*epsilon*r6*r2*(12.0*r6 - 6.0);
               for (int m=0; m<3; m++)
               {
#if 0
                  s->atoms->f[iOff][m] -= dr[m]*fr;
#else
                  f[iOff][m] -= dr[m]*fr;
#endif
               }
            }
//          ++localPairs ;
//          if (++pcount <1000) {
//             printf("%d %d\n", iOff, jOff) ;
//          }
         } ) ; // loop over atoms in iBox
      } ) ; // loop over atoms in IBoxNeighbors
//    pairsChecked += localPairs ;
      // ePot += ePotLocal;
   } ) ; // loop over local boxes in system
   }

   s->ePotential = ePot*4.0*epsilon;

// printf("pairs = %d\n", pairsChecked) ;

   return 0;
}
