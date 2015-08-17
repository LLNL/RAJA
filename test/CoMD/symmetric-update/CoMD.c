/// \file
/// Main program
///
/// \mainpage CoMD: A Classical Molecular Dynamics Mini-app
///
/// CoMD is a reference implementation of typical classical molecular
/// dynamics algorithms and workloads.  It is created and maintained by
/// The Exascale Co-Design Center for Materials in Extreme Environments
/// (ExMatEx).  http://codesign.lanl.gov/projects/exmatex.  The
/// code is intended to serve as a vehicle for co-design by allowing
/// others to extend and/or reimplement it as needed to test performance of 
/// new architectures, programming models, etc.
///
/// The current version of CoMD is available from:
/// http://exmatex.github.io/CoMD
///
/// To contact the developers of CoMD send email to: exmatex-comd@llnl.gov.
///
/// Table of Contents
/// =================
///
/// Click on the links below to browse the CoMD documentation.
///
/// \subpage pg_openmp_specifics
///
/// \subpage pg_md_basics
///
/// \subpage pg_building_comd
///
/// \subpage pg_running_comd
///
/// \subpage pg_measuring_performance
///
/// \subpage pg_problem_selection_and_scaling
///
/// \subpage pg_verifying_correctness
///
/// \subpage pg_comd_architecture
///
/// \subpage pg_optimization_targets
///
/// \subpage pg_whats_new

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>
#include <unistd.h>
#include <assert.h>

#include "RAJA/RAJA.hxx"
#include "CoMDTypes.h"
#include "decomposition.h"
#include "linkCells.h"
#include "eam.h"
#include "ljForce.h"
#include "initAtoms.h"
#include "memUtils.h"
#include "yamlOutput.h"
#include "parallel.h"
#include "performanceTimers.h"
#include "mycommand.h"
#include "timestep.h"
#include "constants.h"

#define REDIRECT_OUTPUT 0
#define   MIN(A,B) ((A) < (B) ? (A) : (B))

static SimFlat* initSimulation(Command cmd);
static void destroySimulation(SimFlat** ps);

static void initSubsystems(void);
static void finalizeSubsystems(void);

static BasePotential* initPotential(
   int doeam, const char* potDir, const char* potName, const char* potType);
static SpeciesData* initSpecies(BasePotential* pot);
static Validate* initValidate(SimFlat* s);
static void validateResult(const Validate* val, SimFlat *sim);

static void sumAtoms(SimFlat* s);
static void printThings(SimFlat* s, int iStep, double elapsedTime);
static void printSimulationDataYaml(FILE* file, SimFlat* s);
static void sanityChecks(Command cmd, double cutoff, double latticeConst, char latticeType[8]);


int main(int argc, char** argv)
{
   // Prolog
   initParallel(&argc, &argv);
   profileStart(totalTimer);
   initSubsystems();
   timestampBarrier("Starting Initialization\n");

   yamlAppInfo(yamlFile);
   yamlAppInfo(screenOut);

   Command cmd = parseCommandLine(argc, argv);
   printCmdYaml(yamlFile, &cmd);
   printCmdYaml(screenOut, &cmd);

   SimFlat* sim = initSimulation(cmd);
   printSimulationDataYaml(yamlFile, sim);
   printSimulationDataYaml(screenOut, sim);

   Validate* validate = initValidate(sim); // atom counts, energy
   timestampBarrier("Initialization Finished\n");

   timestampBarrier("Starting simulation\n");

   // This is the CoMD main loop
   const int nSteps = sim->nSteps;
   const int printRate = sim->printRate;
   int iStep = 0;
   profileStart(loopTimer);
   for (; iStep<nSteps;)
   {
      startTimer(commReduceTimer);
      sumAtoms(sim);
      stopTimer(commReduceTimer);

      printThings(sim, iStep, getElapsedTime(timestepTimer));

      startTimer(timestepTimer);
      timestep(sim, printRate, sim->dt);
      stopTimer(timestepTimer);

      iStep += printRate;
   }
   profileStop(loopTimer);

   sumAtoms(sim);
   printThings(sim, iStep, getElapsedTime(timestepTimer));
   timestampBarrier("Ending simulation\n");

   // Epilog
   validateResult(validate, sim);
   profileStop(totalTimer);

   printPerformanceResults(sim->atoms->nGlobal, sim->printRate);
   printPerformanceResultsYaml(yamlFile);

   destroySimulation(&sim);
   comdFree(validate);
   finalizeSubsystems();

   timestampBarrier("CoMD Ending\n");
   destroyParallel();

   return 0;
}

/// Initialized the main CoMD data stucture, SimFlat, based on command
/// line input from the user.  Also performs certain sanity checks on
/// the input to screen out certain non-sensical inputs.
///
/// Simple data members such as the time step dt are initialized
/// directly, substructures such as the potential, the link cells, the
/// atoms, etc., are initialized by calling additional initialization
/// functions (initPotential(), initLinkCells(), initAtoms(), etc.).
/// Initialization order is set by the natural dependencies of the
/// substructure such as the atoms need the link cells so the link cells
/// must be initialized before the atoms.
SimFlat* initSimulation(Command cmd)
{
   SimFlat* sim = (SimFlat *) comdMalloc(sizeof(SimFlat));
   sim->nSteps = cmd.nSteps;
   sim->printRate = cmd.printRate;
   sim->dt = cmd.dt;
   sim->domain = NULL;
   sim->boxes = NULL;
   sim->atoms = NULL;
   sim->ePotential = 0.0;
   sim->eKinetic = 0.0;
   sim->atomExchange = NULL;

   sim->pot = initPotential(cmd.doeam, cmd.potDir, cmd.potName, cmd.potType);
   real_t latticeConstant = cmd.lat;
   if (cmd.lat < 0.0)
      latticeConstant = sim->pot->lat;

   // ensure input parameters make sense.
   sanityChecks(cmd, sim->pot->cutoff, latticeConstant, sim->pot->latticeType);

   sim->species = initSpecies(sim->pot);

   real3 globalExtent;
   globalExtent[0] = cmd.nx * latticeConstant;
   globalExtent[1] = cmd.ny * latticeConstant;
   globalExtent[2] = cmd.nz * latticeConstant;

   sim->domain = initDecomposition(
      cmd.xproc, cmd.yproc, cmd.zproc, globalExtent);

   sim->boxes = initLinkCells(sim->domain, sim->pot->cutoff);
   sim->atoms = initAtoms(sim->boxes);

   // create lattice with desired temperature and displacement.
   createFccLattice(cmd.nx, cmd.ny, cmd.nz, latticeConstant, sim);

   /* Create Total IndexSets */
   RAJA::RangeSegment *rangeArray = new RAJA::RangeSegment[sim->boxes->nTotalBoxes] ;

   sim->isTotal = new RAJA::IndexSet() ;
   for (int i=0; i<sim->boxes->nTotalBoxes; ++i) {
      rangeArray[i].setBegin(i*MAXATOMS) ;
      rangeArray[i].setEnd(i*MAXATOMS + sim->boxes->nAtoms[i]) ;
      sim->isTotal->push_back( rangeArray[i] ) ;
   }

   /* Create Neighbor IndexSet Views */
   for (int i=0; i<sim->boxes->nLocalBoxes; ++i) {
      RAJA::IndexSet *neighbors ;

      if (cmd.doeam) {
         int tmpBox[27] ;
         int tmpCount = 0 ;

         for (int j=0; j<27; ++j) {
            if (sim->boxes->nbrBoxes[i][j] >= i) {
               tmpBox[tmpCount++] = sim->boxes->nbrBoxes[i][j] ;
            }
         }
         neighbors =
            sim->isTotal->createView(tmpBox, tmpCount) ;
      }
      else {
         neighbors =
            sim->isTotal->createView(sim->boxes->nbrBoxes[i], 27) ;
      }

      sim->isTotal->getSegment(i)->setPrivate(
         reinterpret_cast<void *>(neighbors)
      ) ;
   }

   /* Create Wavefront traversal */
   sim->isWavefront = BuildWavefront(sim->boxes, sim->isTotal,
                                     sim->boxes->gridSize[0],
                                     sim->boxes->gridSize[1],
                                     sim->boxes->gridSize[2],
                                     omp_get_max_threads()) ;

   /* Create Local IndexSet View */
   if (0 /* cmd.doeam || (sim->boxes->gridSize[0]%4 + sim->boxes->gridSize[1]%4 + sim->boxes->gridSize[2]%4 != 0) */) {
      /* Create "Simple" Local IndexSet View */
      sim->isLocal = sim->isTotal->createView(0, sim->boxes->nLocalBoxes);
   }
   else {
      sim->isLocal = sim->isWavefront ;
   }

   setTemperature(sim, cmd.temperature);
   randomDisplacements(sim, cmd.initialDelta);

   sim->atomExchange = initAtomHaloExchange(sim->domain, sim->boxes);

   // Forces must be computed before we call the time stepper.
   startTimer(redistributeTimer);
   redistributeAtoms(sim);
   stopTimer(redistributeTimer);

   startTimer(computeForceTimer);
   computeForce(sim);
   stopTimer(computeForceTimer);

   kineticEnergy(sim);


   return sim;
}

/// frees all data associated with *ps and frees *ps
void destroySimulation(SimFlat** ps)
{
   if ( ! ps ) return;

   SimFlat* s = *ps;
   if ( ! s ) return;

   BasePotential* pot = s->pot;
   if ( pot) pot->destroy(&pot);
   destroyLinkCells(&(s->boxes));
   destroyAtoms(s->atoms);
   destroyHaloExchange(&(s->atomExchange));
   comdFree(s->species);
   comdFree(s->domain);
   comdFree(s);
   *ps = NULL;

   return;
}

void initSubsystems(void)
{
#if REDIRECT_OUTPUT
   freopen("testOut.txt","w",screenOut);
#endif

   yamlBegin();
}

void finalizeSubsystems(void)
{
#if REDIRECT_OUTPUT
   fclose(screenOut);
#endif
   yamlEnd();
}

/// decide whether to get LJ or EAM potentials
BasePotential* initPotential(
   int doeam, const char* potDir, const char* potName, const char* potType)
{
   BasePotential* pot = NULL;

   if (doeam) 
      pot = initEamPot(potDir, potName, potType);
   else 
      pot = initLjPot();
   assert(pot);
   return pot;
}

SpeciesData* initSpecies(BasePotential* pot)
{
   SpeciesData* species = (SpeciesData*)comdMalloc(sizeof(SpeciesData));

   strcpy(species->name, pot->name);
   species->atomicNo = pot->atomicNo;
   species->mass = pot->mass;

   return species;
}

Validate* initValidate(SimFlat* sim)
{
   sumAtoms(sim);
   Validate* val = (Validate*)comdMalloc(sizeof(Validate));
   val->eTot0 = (sim->ePotential + sim->eKinetic) / sim->atoms->nGlobal;
   val->nAtoms0 = sim->atoms->nGlobal;

   if (printRank())
   {
      fprintf(screenOut, "\n");
      printSeparator(screenOut);
      fprintf(screenOut, "Initial energy : %14.12f, atom count : %d \n", 
            val->eTot0, val->nAtoms0);
      fprintf(screenOut, "\n");
   }
   return val;
}

void validateResult(const Validate* val, SimFlat* sim)
{
   if (printRank())
   {
      real_t eFinal = (sim->ePotential + sim->eKinetic) / sim->atoms->nGlobal;

      int nAtomsDelta = (sim->atoms->nGlobal - val->nAtoms0);

      fprintf(screenOut, "\n");
      fprintf(screenOut, "\n");
      fprintf(screenOut, "Simulation Validation:\n");

      fprintf(screenOut, "  Initial energy  : %14.12f\n", val->eTot0);
      fprintf(screenOut, "  Final energy    : %14.12f\n", eFinal);
      fprintf(screenOut, "  eFinal/eInitial : %f\n", eFinal/val->eTot0);
      if ( nAtomsDelta == 0)
      {
         fprintf(screenOut, "  Final atom count : %d, no atoms lost\n",
               sim->atoms->nGlobal);
      }
      else
      {
         fprintf(screenOut, "#############################\n");
         fprintf(screenOut, "# WARNING: %6d atoms lost #\n", nAtomsDelta);
         fprintf(screenOut, "#############################\n");
      }
   }
}

void sumAtoms(SimFlat* s)
{
   // sum atoms across all processers
   s->atoms->nLocal = 0;
   for (int i = 0; i < s->boxes->nLocalBoxes; i++)
   {
      s->atoms->nLocal += s->boxes->nAtoms[i];
   }

   startTimer(commReduceTimer);
   addIntParallel(&s->atoms->nLocal, &s->atoms->nGlobal, 1);
   stopTimer(commReduceTimer);
}

/// Prints current time, energy, performance etc to monitor the state of
/// the running simulation.  Performance per atom is scaled by the
/// number of local atoms per process this should give consistent timing
/// assuming reasonable load balance
void printThings(SimFlat* s, int iStep, double elapsedTime)
{
   // keep track previous value of iStep so we can calculate number of steps.
   static int iStepPrev = -1; 
   static int firstCall = 1;

   int nEval = iStep - iStepPrev; // gives nEval = 1 for zeroth step.
   iStepPrev = iStep;
   
   if (! printRank() )
      return;

   if (firstCall)
   {
      firstCall = 0;
      fprintf(screenOut, 
       "#                                                                                         Performance\n" 
       "#  Loop   Time(fs)       Total Energy   Potential Energy     Kinetic Energy  Temperature   (us/atom)     # Atoms\n");
      fflush(screenOut);
   }

   real_t time = iStep*s->dt;
   real_t eTotal = (s->ePotential+s->eKinetic) / s->atoms->nGlobal;
   real_t eK = s->eKinetic / s->atoms->nGlobal;
   real_t eU = s->ePotential / s->atoms->nGlobal;
   real_t Temp = (s->eKinetic / s->atoms->nGlobal) / (kB_eV * 1.5);

   double timePerAtom = 1.0e6*elapsedTime/(double)(nEval*s->atoms->nLocal);

   fprintf(screenOut, " %6d %10.2f %18.12f %18.12f %18.12f %12.4f %10.4f %12d\n",
           iStep, time, eTotal, eU, eK, Temp, timePerAtom, s->atoms->nGlobal);
}

/// Print information about the simulation in a format that is (mostly)
/// YAML compliant.
void printSimulationDataYaml(FILE* file, SimFlat* s)
{
   // All ranks get maxOccupancy
   int maxOcc = maxOccupancy(s->boxes);

   // Only rank 0 prints
   if (! printRank())
      return;
   
   fprintf(file,"Simulation data: \n");
   fprintf(file,"  Total atoms        : %d\n", 
           s->atoms->nGlobal);
   fprintf(file,"  Min global bounds  : [ %14.10f, %14.10f, %14.10f ]\n",
           s->domain->globalMin[0], s->domain->globalMin[1], s->domain->globalMin[2]);
   fprintf(file,"  Max global bounds  : [ %14.10f, %14.10f, %14.10f ]\n",
           s->domain->globalMax[0], s->domain->globalMax[1], s->domain->globalMax[2]);
   printSeparator(file);
   fprintf(file,"Decomposition data: \n");
   fprintf(file,"  Processors         : %6d,%6d,%6d\n", 
           s->domain->procGrid[0], s->domain->procGrid[1], s->domain->procGrid[2]);
   fprintf(file,"  Local boxes        : %6d,%6d,%6d = %8d\n", 
           s->boxes->gridSize[0], s->boxes->gridSize[1], s->boxes->gridSize[2], 
           s->boxes->gridSize[0]*s->boxes->gridSize[1]*s->boxes->gridSize[2]);
   fprintf(file,"  Box size           : [ %14.10f, %14.10f, %14.10f ]\n", 
           s->boxes->boxSize[0], s->boxes->boxSize[1], s->boxes->boxSize[2]);
   fprintf(file,"  Box factor         : [ %14.10f, %14.10f, %14.10f ] \n", 
           s->boxes->boxSize[0]/s->pot->cutoff,
           s->boxes->boxSize[1]/s->pot->cutoff,
           s->boxes->boxSize[2]/s->pot->cutoff);
   fprintf(file, "  Max Link Cell Occupancy: %d of %d\n",
           maxOcc, MAXATOMS);
   printSeparator(file);
   fprintf(file,"Potential data: \n");
   s->pot->print(file, s->pot);
   
   fflush(file);      
}

/// Check that the user input meets certain criteria.
void sanityChecks(Command cmd, double cutoff, double latticeConst, char latticeType[8])
{
   int failCode = 0;

   // Check that domain grid matches number of ranks. (fail code 1)
   int nProcs = cmd.xproc * cmd.yproc * cmd.zproc;
   if (nProcs != getNRanks())
   {
      failCode |= 1;
      if (printRank() )
         fprintf(screenOut,
                 "\nNumber of MPI ranks must match xproc * yproc * zproc\n");
   }

   // Check whether simuation is too small (fail code 2)
   double minx = 2*cutoff*cmd.xproc;
   double miny = 2*cutoff*cmd.yproc;
   double minz = 2*cutoff*cmd.zproc;
   double sizex = cmd.nx*latticeConst;
   double sizey = cmd.ny*latticeConst;
   double sizez = cmd.nz*latticeConst;

   if ( sizex < minx || sizey < miny || sizez < minz)
   {
      failCode |= 2;
      if (printRank())
         fprintf(screenOut,"\nSimulation too small.\n"
                 "  Increase the number of unit cells to make the simulation\n"
                 "  at least (%3.2f, %3.2f. %3.2f) Ansgstroms in size\n",
                 minx, miny, minz);
   }

   // Check for supported lattice structure (fail code 4)
   if (strcasecmp(latticeType, "FCC") != 0)
   {
      failCode |= 4;
      if ( printRank() )
         fprintf(screenOut,
                 "\nOnly FCC Lattice type supported, not %s. Fatal Error.\n",
                 latticeType);
   }
   int checkCode = failCode;
   bcastParallel(&checkCode, sizeof(int), 0);
   // This assertion can only fail if different tasks failed different
   // sanity checks.  That should not be possible.
   assert(checkCode == failCode);
      
   if (failCode != 0)
      exit(failCode);
}

// --------------------------------------------------------------


/// \page pg_building_comd Building CoMD
///
/// CoMD is written with portability in mind and should compile using
/// practically any compiler that implements the C99 standard.  You will
/// need to create a Makefile by copying the sample provided with the
/// distribution (Makefile.vanilla).
/// 
///     $ cp Makefile.vanilla Makefile
///
/// and use the make command to build the code
/// 
///    $ make
///
/// The sample Makefile will compile the code on many platforms.  See
/// comments in Makefile.vanilla for information about specifying the
/// name of the C compiler, and/or additional compiler switches that
/// might be necessary for your platform.
/// 
/// The main options available in the Makefile are toggling single/double 
/// precision and enabling/disabling MPI. In the event MPI is not
/// available, setting the DO_MPI flag to OFF will create a purely
/// serial build (you will likely also need to change the setting of
/// CC).
/// 
/// The makefile should handle all the dependency checking needed, via
/// makedepend.
/// 
/// 'make clean' removes the object and dependency files. 
/// 
/// 'make distclean' additionally removes the executable file and the
/// documentation files.
/// 
/// Other build options
/// -------------------
///
/// Various other options are made available by \#define arguments within 
/// some of the source files. 
///
/// #REDIRECT_OUTPUT in CoMD.c
///
/// Setting this to 1 will redirect all screen output to a file,
/// currently set to 'testOut.txt'.
///
/// #POT_SHIFT in ljForce.c
///
/// This is set to 1.0 by default, and shifts the values of the cohesive
/// energy given by the Lennard-Jones potential so it is zero at the
/// cutoff radius.  This setting improves energy conservation
/// step-to-step as it reduces the noise generated by atoms crossing the
/// cutoff threshold. However, it does not affect the long-term energy
/// conservation of the code.
///
/// #MAXATOMS in linkCells.h
/// 
/// The default value is 64, which allows ample padding of the linkCell
/// structure to allow for density fluctuations. Reducing it may improve
/// the efficiency of the code via improved thread utilization and
/// reduced memory footprint.

// --------------------------------------------------------------


// --------------------------------------------------------------


/// \page pg_measuring_performance Measuring Performance
///
/// CoMD implements a simple and extensible system of internal timers to
/// measure the performance profile of the code.  As explained in
/// performanceTimers.c, it is easy to create additional timers and
/// associate them with code regions of specific interest.  In addition,
/// the getTime() and getTick() functions can be easily reimplemented to
/// take advantage of platform specific timing resources.
///
/// A timing report is printed at the end of each simulation. 
///
/// ~~~~
/// Timings for Rank 0
///         Timer        # Calls    Avg/Call (s)   Total (s)    % Loop
/// ___________________________________________________________________
/// total                      1      50.6701       50.6701      100.04
/// loop                       1      50.6505       50.6505      100.00
/// timestep                   1      50.6505       50.6505      100.00
///   position             10000       0.0000        0.0441        0.09
///   velocity             20000       0.0000        0.0388        0.08
///   redistribute         10001       0.0003        3.4842        6.88
///     atomHalo           10001       0.0002        2.4577        4.85
///   force                10001       0.0047       47.0856       92.96
///     eamHalo            10001       0.0001        1.0592        2.09
/// commHalo               60006       0.0000        1.7550        3.46
/// commReduce                12       0.0000        0.0003        0.00
/// 
/// Timing Statistics Across 8 Ranks:
///         Timer        Rank: Min(s)       Rank: Max(s)      Avg(s)    Stdev(s)
/// _____________________________________________________________________________
/// total                3:   50.6697       0:   50.6701     50.6699      0.0001
/// loop                 0:   50.6505       4:   50.6505     50.6505      0.0000
/// timestep             0:   50.6505       4:   50.6505     50.6505      0.0000
///   position           2:    0.0437       0:    0.0441      0.0439      0.0001
///   velocity           2:    0.0380       4:    0.0392      0.0385      0.0004
///   redistribute       0:    3.4842       1:    3.7085      3.6015      0.0622
///     atomHalo         0:    2.4577       7:    2.6441      2.5780      0.0549
///   force              1:   46.8624       0:   47.0856     46.9689      0.0619
///     eamHalo          3:    0.2269       6:    1.2936      1.0951      0.3344
/// commHalo             3:    1.0803       6:    2.1856      1.9363      0.3462
/// commReduce           6:    0.0002       2:    0.0003      0.0003      0.0000
/// 
/// ---------------------------------------------------
///  Average atom update rate:   9.39 us/atom/task
/// ---------------------------------------------------
///
/// ~~~~
/// This report consists of two blocks.  The upper block lists the absolute
/// wall clock time spent in each timer on rank 0 of the job.  The lower
/// block reports minimum, maximum, average, and standard deviation of
/// times across all tasks.
/// The ranks where the minimum and maximum values occured are also reported
/// to aid in identifying hotspots or load imbalances.
///
/// The last line of the report gives the atom update rate in
/// microseconds/atom/task.  Since this quantity is normalized by both
/// the number of atoms and the number of tasks it provides a simple
/// figure of merit to compare performance between runs with different
/// numbers of atoms and different numbers of tasks.  Any increase in
/// this number relative to a large number of atoms on a single task
/// represents a loss of parallel efficiency.
/// 
/// Choosing the problem size correctly has important implications for the 
/// reported performance. Small problem sizes may run entirely in the cache 
/// of some architectures, leading to very good performance results. 
/// For general characterization of performance, it is probably best to 
/// choose problem sizes which force the code to access main memory, even
/// though there may be strong scaling scenarios where the code is indeed 
/// running mainly in cache.
///
/// *** Architecture/Configuration for above timing numbers:
/// SGI XE1300 cluster with dual-socket Intel quad-core Nehalem processors. 
/// Each node has 2 Quad-Core Xeon X5550 processors runnning at 2.66 GHz
/// with 3 GB of memory per core.

// --------------------------------------------------------------


/// \page pg_problem_selection_and_scaling Problem Selection and Scaling
///
/// CoMD is a reference molecular dynamics simulation code as used in
/// materials science.
///
/// Problem Specification  {#sec_problem_spec}
/// ======================
///
/// The reference problem is solid Copper starting from a face-centered
/// cubic (FCC) lattice.  The initial thermodynamic conditions
/// (Temperature and Volume (via the lattice spacing, lat))can be specified
/// from the command line input. The default is 600 K and standard
/// volume (lat = 3.615 Angstroms).  
/// Different temperatures (e.g. T =3000K) and volumes can be
/// specified to melt the system and enhance the interchange of atoms
/// between domains.
///
/// The dynamics is micro-canonical (NVE = constant Number of atoms,
/// constant total system Volume, and constant total system Energy). As
/// a result, the temperature is not fixed. Rather, the temperature will
/// adjust from the initial temperature (as specified on the command line)
/// to a final temperature as the total system kinetic energy comes into
/// equilibrium with the total system potential energy.
///
/// The total size of the problem (number of atoms) is specified by the
/// number (nx, ny, nz) of FCC unit cells in the x, y, z directions: nAtoms
/// = 4 * nx * ny * nz. The default size is nx = ny = nz = 20 or 32,000 atoms.
///
/// The simulation models bulk copper by replicating itself in every
/// direction using periodic boundary conditions.
///
/// Two interatomic force models are available: the Lennard-Jones (LJ)
/// two-body potential (ljForce.c) and the many-body Embedded-Atom Model (EAM)
/// potential (eam.c). The LJ potential is included for comparison and
/// is a valid approximation for constant volume and uniform
/// density. The EAM potential is a more accurate model of cohesion in
/// simple metals like Copper and includes the energetics necessary to
/// model non-uniform density and free surfaces.
///
/// Scaling Studies in CoMD  {#sec_scaling_studies}
/// =======================
///
/// CoMD implements a simple geometric domain decomposition to divide
/// the total problem space into domains, which are owned by MPI
/// ranks. Each domain is a single-program multiple data (SPMD)
/// partition of the larger problem.
///
/// Caution: When doing scaling studies, it is important to distinguish
/// between the problem setup phase and the problem execution phase. Both
/// are important to the workflow of doing molecular dynamics, but it
/// is the execution phase we want to quantify in the scaling studies
/// described below, for that dominates the execution time for long runs
/// (millions of time steps). The problem setup can be an appreciable fraction
/// of the execution time for short runs (the default is 100 time steps)
/// and erroneous conclusions drawn.
///
/// This code is configured with timers. The times are reported per particle
/// and the timers for the force calculation, timestep, etc start after the
/// initialization phase is done.
///
/// Weak Scaling  {#ssec_weak_scaling}
/// -----------
///
/// A weak scaling test fixes the amount of work per processor and
/// compares the execution time over number of processors. Weak scaling
/// keeps the ratio of inter-processor communication (surface) to
/// intra-processor work (volume) fixed. The amount of inter-processor
/// work scales with the number of processors in the domain and O(1000)
/// atoms per domain are needed for reasonable performance.
///
/// Examples,
///
/// - Increase in processor count by 8: <br>
///    (xproc=yproc=zproc=2, nx=ny=nz=20) -> (xproc=yproc=zproc=4, nx=ny=nz=40)
///
/// - Increase in processor count by 2: <br>
///    (xproc=yproc=zproc=2, nx=ny=nz=20) -> (xproc=yproc=2, zproc=4, nx=ny=20, nz=40)
///
/// In general, it is wise to keep the ratio of processor count to
/// system size in each direction fixed (i.e. cubic domains): xproc_0 / nx_0 = xproc_1 /
/// nx_1, since this minimizes surface area to volume. 
/// Feel free to experiment, you might learn something about
/// algorithms to optimize communication relative to work.
///
/// Strong Scaling {#ssec_strong_scaling}
/// ---------------
///
/// A strong scaling test fixes the total problem size and compares the
/// execution time for different numbers of processors. Strong scaling
/// increases the ratio of inter-processor communication (surface) to
/// intra-processor work (volume).
///
/// Examples,
///
/// - Increase in processor count by 8: <br>
///    (xproc=yproc=zproc=2, nx=ny=nz=20) -> (xproc=yproc=zproc=4, nx=ny=nz=20)
///
/// - Increase in processor count by 2: <br>
///    (xproc=yproc=zproc=2, nx=ny=nz=20) -> (xproc=yproc=2, zproc=4, nx=ny=nz=20)
///
/// The domain decomposition requires O(1000) atoms per domain and
/// begins to scale poorly for small numbers of atoms per domain. 
/// Again, feel free to experiment, you might learn something here as
/// well.  For example, when molecular dynamics codes were written for
/// vector supercomputers, large lists of force pairs were created for
/// the vector processor. These force lists provide a natural force
/// decomposition for early parallel computers (Fast Parallel Algorithms
/// for Short-Range Molecular Dynamics, S. J. Plimpton, J Comp Phys,
/// 117, 1-19 (1995).) Using replicated data, force decomposition can
/// scale to fewer than one atom per processor and is a natural
/// mechanism to exploit intra-processor parallelism.
///
/// For further details see for example:
/// https://support.scinet.utoronto.ca/wiki/index.php/Introduction_To_Performance


// --------------------------------------------------------------


/// \page pg_verifying_correctness Verifying Correctness
///
/// Verifying the correctness of an MD simulation is challenging.
/// Because MD is Lyapunov unstable, any small errors, even harmless
/// round-off errors, will lead to a long-term divergence in the atom
/// trajectories.  Hence, comparing atom positions at the end of a run
/// is not always a useful verification technique.  (Such divergences
/// are not a problem for science applications of MD since they do not
/// alter the statistical physics.)  Small, single-particle errors can
/// also be difficult to detect in system-wide quantities such as the
/// kinetic or potential energy that are averaged over a large number of
/// particles.
///
/// In spite of these challenges, there are several methods which are
/// likely to catch significant errors.
///
/// Cohesive Energy {#sec_ver_cohesive_energy}
/// ===============
///
/// With a perfect lattice as the initial structure (this is the
/// default), the potential energy per atom is the cohesive energy.
/// This value should be computed correctly to many decimal places.  Any
/// variation beyond the last 1 or 2 decimal places is cause for
/// investigation.  The correct values for the cohesive energy are
///
/// | Potential      | Cohesive Energy |
/// | :------------- | :-------------- |
/// | Lennard-Jones  | -1.243619295058 |
/// | EAM (Adams)    | -3.538079224691 |
/// | EAM (Mishin)   | -3.539999969176 |
///
/// The \link sec_command_line_options command
/// line options \endlink documentation explains the switches used to
/// select the potential used in the simulation.
///
/// Note that the cohesive energy calculation is not sensitive to errors
/// in forces.  It is also performed on a highly symmetric structure so
/// there are many errors this will not catch.  Still, it is a good
/// first check.
///
/// Energy Conservation {#sec_ver_energy_conservation}
/// ===================
///
/// A correctly implemented force kernel, with an appropriate time step
/// (the default value of 1 fs is conservative for temperatures under
/// 10,000K) will conserve total energy over long times to 5 or more
/// digits.  Any long term systematic drift in the total energy is a
/// cause for concern.
///
/// To facilitate checking energy conservation CoMD prints the final and
/// initial values of the total energy.  When comparing these values, pay
/// careful attention to these details:
///
/// - It is common to observe an initial transient change in the total
///   energy.  Differences in the total energy of 2-3% can be expected in
///   the first 10-100 time steps.
/// - The best way to check energy conservation is to run at least
///   several thousand steps and look at the slope of the total energy
///   ignoring at least the first one or two thousand steps.  More steps
///   are even better.
/// - Set the temperature to at least several hundred K.  This ensures
///   that atoms will sample a large range of configurations and expose
///   possible errors.
/// - Fluctuations in the energy can make it difficult to tell if
///   conservation is observed.  Increasing the number of atoms will reduce
///   the fluctuations.
/// 
///
/// Particle Conservation {#sec_ver_particle_conservation}
/// =====================
///
/// The simulation should always end with the same number of particles
/// it started with.  Any change is a bug.  CoMD checks the initial and
/// final number of particles and prints a warning at the end of the
/// simulation if they are not equal.
///
/// Reproducibility {#sec_ver_reproducibility}
/// ===============
///
/// The same simulation run repeatedly on the same hardware should
/// produce the same result.  Because parallel computing can add
/// elements of non-determinism we do not expect perfect long term
/// reproducibility, however over a few hundred to a few thousand time
/// steps the energies should not exhibit run-to-run differences outside
/// the last 1 or 2 decimal places.  Larger differences are a sign of
/// trouble and should be investigated.  This kind of test is
/// practically the only way to detect race conditions in shared memory
/// parallelism.
///
/// Portability {#sec_ver_portability}
/// ===========
///
/// In our experience, simulations that start from the same initial
/// condition tend to produce very similar trajectories over short terms
/// (100 to 1000 time step), even on different hardware platforms.
/// Short term differences beyond the last 1 or 2 decimal places should
/// likely be investigated.
///
/// General Principles {#sec_ver_general}
/// =======================
///
/// - Simulations run at 0K are too trivial for verification, set
///   the initial temperature to at least several hundred K.
/// - Longer runs are better to check conservation.  Compare
///   energies after initial transients are damped out.
/// - Larger runs are better to check conservation.  Fluctuations in the
///   energy are averaged out.
/// - Short term (order 100 time steps) discrepancies from run-to-run
///   or platform-to platform beyond the last one or two decimal places
///   are reason for concern.  Differences in 4th or 5th decimal place
///   is almost certainly a bug.
/// - Contact the CoMD developers (exmatex-comd@llnl.gov) if you have
///   questions about validation.
///

// --------------------------------------------------------------


/// \page pg_comd_architecture CoMD Architecture
///
/// Program Flow {#sec_program_flow}
/// ============
///
/// We have attempted to make the program flow in CoMD 1.1 as simple and
/// transparent as possible.  The main program consists of three blocks:
/// prolog, main loop, and epilog.
///
/// Prolog {#ssec_flow_prolog}
/// -------
///
/// The job of the prolog is to initialize the simulation and prepare
/// for the main loop.  Notable tasks in the prolog include calling
/// - initParallel() to start MPI
/// - parseCommandLine() to read the command line options
/// - initSimulation() to initialize the main data structure, SimFlatSt.
///   This includes tasks such as
///   - initEamPot() to read tabular data for the potential function
///   - initDecomposition() to set up the domain decomposition
///   - createFccLattice() to generate an initial structure for the atoms
/// - initValidate() to store initial data for a simple validation check
///
/// In CoMD 1.1 all atomic structures are internally generated so
/// there is no need to read large files with atom coordinate data.
///
/// Main Loop {#ssec_flow_main_loop}
/// ---------
///
/// The main loop calls
/// - timestep(), the integrator to update particle positions,
/// - printThings() to periodically prints simulation information
///
/// The timestep() function is the heart of the code as it choreographs
/// updating the particle positions, along with computing forces
/// (computeForce()) and communicating atoms between ranks
/// (redistributeAtoms()).
///
/// Epilog {#ssec_flow_epilog}
/// -------
///
/// The epilog code handles end of run bookkeeping such as
/// - validateResult() to check validation
/// - printPerformanceResults() to print a performance summary 
/// - destroySimulation() to free memory
///
/// Key Data Structures {#sec_key_data_structures}
/// ==================
///
/// Practically all data in CoMD belongs to the SimFlatSt structure.
/// This includes:
/// - BasePotentialSt A polymorphic structure for the potential model
/// - HaloExchangeSt A polymorphic strcuture for communication halo data
/// - DomainSt The parallel domain decomposition
/// - LinkCellSt The link cells
/// - AtomsSt The atom coordinates and velocities
/// - SpeciesDataSt Properties of the atomic species being simulated.
///
/// Consult the individual pages for each of these structures to learn
/// more.  The descriptions in haloExchange.c and initLinkCells() are
/// especially useful to understand how the atoms are commuicated
/// between tasks and stored in link cells for fast pair finding.

// --------------------------------------------------------------


/// \page pg_optimization_targets Optimization Targets
///
/// Computation {#sec_computation}
/// ============
///
/// The computational effort of classical MD is usually highly focused
/// in the force kernel.  The two force kernels supplied by CoMD are
/// eamForce() and ljForce().  Both kernels are fundamentally loops over
/// pairs of atoms with significant opportunity to exploit high levels
/// of concurrency.  One potential challenge when reordering or
/// parallelizing the pair loop structure is preventing race conditions
/// that result if two concurrent pair evaluations try to simultaneously
/// increment the forces and energies on the same atom.
///
/// The supplied EAM kernel uses interpolation from tabular data to
/// evaluate functions.  Hence the interpolate() function is another
/// potential optimization target.  Note that the two potential files
/// distributed with CoMD have very different sizes.  The Adams
/// potential (Cu_u6.eam) has 500 points per function in the table while
/// the Mishin potential (Cu01.eam.alloy) has 10,000 points per
/// function.  This difference could potentially impact important
/// details such as cache miss rates.
///
/// Communication {#sec_communication}
/// =============
///  
/// As the number of atoms per MPI rank decreases, the communication
/// routines will start to require a significant fraction of the
/// run time.  The main communication routine in CoMD is haloExchange().
/// The halo exchange is simple nearest neighbor, point-to-point
/// communication so it should scale well to practically any number of
/// nodes.
///
/// The halo exchange in CoMD 1.1 is a very simple 6-direction
/// structured halo exchange (see haloExchange.c).  Other exchange
/// algorithms can be implemented without much difficulty.
///
/// The halo exchange function is called in two very different contexts.
/// The main usage is to exchange halo particle information (see
/// initAtomHaloExchange()).  This process is coordinated by the
/// redistributeAtoms() function.
///
/// In addition to the atom exchange, when using the EAM potential, a
/// halo exchange is performed in the force routine (see
/// initForceHaloExchange()).


// --------------------------------------------------------------


/// \page pg_whats_new New Features and Changes in CoMD 1.1
///
/// The main goals of the 1.1 release were to add support for MPI and to
/// improve the structure and clarity of the code.  Achieving these
/// goals required considerable changes compared to the 1.0 release.
/// However, the core structure of the most computationally intensive
/// kernels (the force routines) is mostly unchanged.  We believe that
/// lessons learned from optimizing 1.0 force kernels to specific
/// hardware or programming models can be quickly transferred to kernels
/// in the 1.1 release.
///
/// Significant changes in CoMD 1.1 include:
///
/// - MPI support.  Both MPI and single node serial executables can be
///   built from the same source files.
///
/// - Improved modularity and code clarity.  Major data structures are
///   now organized with their own structs and initialization routines.
///
/// - The build system has been simplified to use only standard
///   Makefiles instead of CMake.
///
/// - The halo exchange operation needed to communicate remote particle
///   data between MPI ranks also creates "image" particles in the
///   serial build.
///
/// - Unified force kernels for both serial and MPI builds
///
///   - The addition of remote/image atoms allows periodic boundary
///     conditions to be handled outside the force loop.
///
///   - An additional communication/data copy step to handle electron
///     density on remote/image atoms has been added to the EAM force
///     loop.
///
/// - The coordinate system has been simplified to a single global
///   coordinate system for all particles.
///
/// - Evaluation of energies and forces using a Chebyshev polynomial
///   fits has been removed.  Polynomial approximation of energies and
///   forces will return in a future CoMD version.
///
/// - Atomic structures are now generated internally, eliminating the
///   requirement to read, write, and distribute large atom
///   configuration files.  Arbitrarily large initial structures can
///   be generated with specified initial temperature and random
///   displacements from lattice positions.  Code to read/write atomic
///   positions has been removed.
///
/// - EAM potentials are now read from standard funcfl and setfl format
///   files.  Voter style files are no longer supported.
///
/// - Collection of performance metrics is significantly improved.
///   Developers can easily add new timers to regions of interest.  The
///   system is also designed to allow easy integration with platform
///   specific API's to high resolution timers, cycle counters,
///   hardware counters, etc.
///
///
/// - Hooks to in-situ analysis and visualization have been removed.
///   In-situ analysis capabilities will return in a future CoMD release.
///
/// Please contact the CoMD developers (exematex-comd@llnl.gov) if
/// any of the deleted features negative impacts your work.  We
/// may be able to help produce a custom version that includes the code
/// you need.


// --------------------------------------------------------------


/// \page pg_md_basics MD Basics
///
/// The molecular dynamics (MD) computer simulation method is a well
/// established and important tool for the study of the dynamical
/// properties of liquids, solids, and other systems of interest in
/// Materials Science and Engineering, Chemistry and Biology. A material
/// is represented in terms of atoms and molecules. The method of MD
/// simulation involves the evaluation of the force acting on each atom
/// due to all other atoms in the system and the numerical integration
/// of the Newtonian equations of motion. Though MD was initially
/// developed to compute the equilibrium thermodynamic behavior of
/// materials (equation of state), most recent applications have used MD
/// to study non-equilibrium processes.
///
/// Wikipeda offers a basic introduction to molecular dynamics with
/// many references:
///
/// http://en.wikipedia.org/wiki/Molecular_dynamics
///
/// For a thorough treatment of MD methods, see:
/// - "Computer simulation of liquids" by M.P. Allen and D.J. Tildesley
///    (Oxford, 1989)
///    ISBN-10: 0198556454 | ISBN-13: 978-0198556459.
///
/// For an understanding of MD simulations and application to statistical mechanics:
/// - "Understanding Molecular Simulation, Second Edition: From Algorithms
///    to Applications," by D. Frenkel and B. Smit (Academic Press, 2001)
///    ISBN-10: 0122673514 | ISBN-13: 978-0122673511
/// - "Statistical and Thermal Physics: With Computer Applications," by
///    H. Gould and J. Tobochnik (Princeton, 2010)
///    ISBN-10: 0691137447 | ISBN-13: 978-0691137445
///
/// CoMD implements both the Lennard-Jones Potential (ljForce.c) and the
/// Embedded Atom Method Potential (eam.c).
///
