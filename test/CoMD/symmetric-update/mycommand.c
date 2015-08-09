/// \file
/// Handle command line arguments.

#include "mycommand.h"

#include <string.h>
#include <stdlib.h>

#include "cmdLineParser.h"
#include "parallel.h"
#include "mytype.h"

/// \page pg_running_comd Running CoMD
///
/// \section sec_command_line_options Command Line Options
///
/// CoMD accepts a number of command line options to set the parameters
/// of the simulation.  Every option has both a long form and a short
/// form.  The long and short form of the arguments are entirely
/// interchangeable and may be mixed. All the arguments are independent
/// with the exception of the \--potDir, \--potName, and \--potType,
/// (short forms -d, -n, and -t) arguments which are only relevant when
/// used in conjunction with \--doeam, (-e).
///
/// Supported options are:
///
/// | Long  Form    | Short Form  | Default Value | Description
/// | :------------ | :---------: | :-----------: | :----------
/// | \--help       | -h          | N/A           | print this message
/// | \--potDir     | -d          | pots          | potential directory
/// | \--potName    | -p          | Cu_u6.eam     | potential name
/// | \--potType    | -t          | funcfl        | potential type (funcfl or setfl)
/// | \--doeam      | -e          | N/A           | compute eam potentials (default is LJ)
/// | \--nx         | -x          | 20            | number of unit cells in x
/// | \--ny         | -y          | 20            | number of unit cells in y
/// | \--nz         | -z          | 20            | number of unit cells in z
/// | \--xproc      | -i          | 1             | number of ranks in x direction
/// | \--yproc      | -j          | 1             | number of ranks in y direction
/// | \--zproc      | -k          | 1             | number of ranks in z direction
/// | \--nSteps     | -N          | 100           | total number of time steps
/// | \--printRate  | -n          | 10            | number of steps between output
/// | \--dt         | -D          | 1             | time step (in fs)
/// | \--lat        | -l          | -1            | lattice parameter (Angstroms)
/// | \--temp       | -T          | 600           | initial temperature (K)
/// | \--delta      | -r          | 0             | initial delta (Angstroms)
///
/// Notes: 
/// 
/// The negative value for the lattice parameter (such as the default
/// value, -1) is interpreted as a flag to indicate that the lattice
/// parameter should be set from the potential. All supplied potentials
/// are for copper and have a lattice constant of 3.615
/// Angstroms. Setting the lattice parameter to any positive value will
/// override the values provided in the potential files.
///
/// The default potential name for the funcfl potential type is
/// Cu_u6.eam (Adams potential).  For the setfl type the default
/// potential name is Cu01.eam.alloy (Mishin potential).  Although these
/// will yield similar dynamics, the table have a very different number
/// of entries (500 vs. 10,000 points, respectively) This may give very
/// different performance, depending on the hardware.
///
/// The default temperature is 600K.  However, when using a perfect
/// lattice the system will rapidly cool to 300K due to equipartition of
/// energy.
///
/// 
/// \subsection ssec_example_command_lines Examples
///
/// All of the examples below assume:
/// - The current working directory contains a copy of the pots dir (or
///   a link to it).
/// - The CoMD bin directory is located in ../bin
///
/// Running in the examples directory will satisfy these requirements.
///
/// ------------------------------
///
/// The canonical base simulation, is 
///
///     $ mpirun -np 1 ../bin/CoMD-mpi 
/// 
/// Or, if the code was built without MPI:
///
///     $ ../bin/CoMD-serial
///
/// ------------------------------
///
/// \subsubsection cmd_examples_potential Changing Potentials
///
/// To run with the default (Adams) EAM potential, specify -e:
///
///     $ ../bin/CoMD-mpi -e 
///
/// ------------------------------
///
/// To run using the Mishin EAM potential contained in the setfl file
/// Cu01.eam.alloy. This potential uses much larger tables (10,000
/// entries vs. 500 for the Adams potential).
///
///     $ ../bin/CoMD-mpi -e -t setfl 
///
/// ------------------------------
///
/// Selecting the name of a setfl file without setting the appropriate
/// potential type
///
///     $ ../bin/CoMD-mpi -e -p Cu01.eam.alloy 
/// 
/// will result in an error message:
///
/// Only FCC Lattice type supported, not . Fatal Error.
/// 
/// Instead use:
///
///     $ ../bin/CoMD-mpi -e -t setfl -p Cu01.eam.alloy 
///
/// ------------------------------
///
/// \subsubsection cmd_example_struct Initial Structure Modifications
///
/// To change the lattice constant and run with an expanded or
/// compressed lattice:
///
///     $ ../bin/CoMD-mpi -l 3.5 
///
/// This can be useful to test that the potential is being correctly
/// evaluated as a function of interatomic spacing (the cold
/// curve). However, due to the high degree of symmetry of a perfect
/// lattice, this type of test is unlikely to detect errors in the force
/// computation.
///
/// ------------------------------
///
/// Initialize with zero temperature (zero instantaneous particle
/// velocity) but with a random displacements of the atoms (in this
/// case the maximum displacement is 0.1 Angstrom along each axis).  
/// 
///      $ ../bin/CoMD-mpi --delta 0.1 -T 0
///
/// Typical values of delta are in the range of 0.1 to 0.5 Angstroms.
/// Larger values of delta correspond to higher initial potential energy
/// which in turn produce higer temperatures as the structure
/// equilibrates.
///
/// ------------------------------
///
///
/// \subsubsection cmd_examples_scaling Scaling Examples
///
/// Simple shell scripts that demonstrate weak and strong scaling
/// studies are provided in the examples directory.
///
/// ------------------------------
///
/// Run the default global simulation size (32,000 atoms) distributed
/// over 8 cubic subdomains, an example of strong scaling.  If the
/// number of processors does not equal (i*j*k) the run will abort.
/// Notice that spaces are optional between short form options and their
/// arguments.
/// 
///     $ mpirun -np 8 ../bin/CoMD-mpi -i2 -j2 -k2
/// 
/// ------------------------------
///
/// Run a weak scaling example: the simulation is doubled in each
/// dimension from the default 20 x 20 x 20 and the number of subdomains
/// in each direction is also doubled.
///
///     $ mpirun -np 8 ../bin/CoMD-mpi -i2 -j2 -k2 -x 40 -y 40 -z 40
///
/// ------------------------------
///
/// The same weak scaling run, but for 10,000 timesteps, with output
/// only every 100 steps.
///
///     $ mpirun -np 8 ../bin/CoMD-mpi -i2 -j2 -k2 -x 40 -y 40 -z 40 -N 10000 -n 100
///

/// \details Initialize a Command structure with default values, then
/// parse any command line arguments that were supplied to overwrite
/// defaults.
///
/// \param [in] argc the number of command line arguments
/// \param [in] argv the command line arguments array
Command parseCommandLine(int argc, char** argv)
{
   Command cmd;

   memset(cmd.potDir, 0, 1024);
   memset(cmd.potName, 0, 1024);
   memset(cmd.potType, 0, 1024);
   strcpy(cmd.potDir,  "pots");
   strcpy(cmd.potName, "\0"); // default depends on potType
   strcpy(cmd.potType, "funcfl");
   cmd.doeam = 0;
   cmd.nx = 20;
   cmd.ny = 20;
   cmd.nz = 20;
   cmd.xproc = 1;
   cmd.yproc = 1;
   cmd.zproc = 1;
   cmd.nSteps = 100;
   cmd.printRate = 10;
   cmd.dt = 1.0;
   cmd.lat = -1.0;
   cmd.temperature = 600.0;
   cmd.initialDelta = 0.0;

   int help=0;
   // add arguments for processing.  Please update the html documentation too!
   addArg("help",       'h', 0, 'i',  &(help),             0,             "print this message");
   addArg("potDir",     'd', 1, 's',  cmd.potDir,    sizeof(cmd.potDir),  "potential directory");
   addArg("potName",    'p', 1, 's',  cmd.potName,   sizeof(cmd.potName), "potential name");
   addArg("potType",    't', 1, 's',  cmd.potType,   sizeof(cmd.potType), "potential type (funcfl or setfl)");
   addArg("doeam",      'e', 0, 'i',  &(cmd.doeam),        0,             "compute eam potentials");
   addArg("nx",         'x', 1, 'i',  &(cmd.nx),           0,             "number of unit cells in x");
   addArg("ny",         'y', 1, 'i',  &(cmd.ny),           0,             "number of unit cells in y");
   addArg("nz",         'z', 1, 'i',  &(cmd.nz),           0,             "number of unit cells in z");
   addArg("xproc",      'i', 1, 'i',  &(cmd.xproc),        0,             "processors in x direction");
   addArg("yproc",      'j', 1, 'i',  &(cmd.yproc),        0,             "processors in y direction");
   addArg("zproc",      'k', 1, 'i',  &(cmd.zproc),        0,             "processors in z direction");
   addArg("nSteps",     'N', 1, 'i',  &(cmd.nSteps),       0,             "number of time steps");
   addArg("printRate",  'n', 1, 'i',  &(cmd.printRate),    0,             "number of steps between output");
   addArg("dt",         'D', 1, 'd',  &(cmd.dt),           0,             "time step (in fs)");
   addArg("lat",        'l', 1, 'd',  &(cmd.lat),          0,             "lattice parameter (Angstroms)");
   addArg("temp",       'T', 1, 'd',  &(cmd.temperature),  0,             "initial temperature (K)");
   addArg("delta",      'r', 1, 'd',  &(cmd.initialDelta), 0,             "initial delta (Angstroms)");

   processArgs(argc,argv);

   // If user didn't set potName, set type dependent default.
   if (strlen(cmd.potName) == 0) 
   {
      if (strcmp(cmd.potType, "setfl" ) == 0)
         strcpy(cmd.potName, "Cu01.eam.alloy");
      if (strcmp(cmd.potType, "funcfl") == 0)
         strcpy(cmd.potName, "Cu_u6.eam");
   }
      
   if (help)
   {
      printArgs();
      freeArgs();
      exit(2);
   }
   freeArgs();

   return cmd;
}

void printCmdYaml(FILE* file, Command* cmd)
{
   if (! printRank())
      return;
   fprintf(file,
           "Command Line Parameters:\n"
           "  doeam: %d\n"
           "  potDir: %s\n"
           "  potName: %s\n"
           "  potType: %s\n"
           "  nx: %d\n"
           "  ny: %d\n"
           "  nz: %d\n"
           "  xproc: %d\n"
           "  yproc: %d\n"
           "  zproc: %d\n"
           "  Lattice constant: %g Angstroms\n"
           "  nSteps: %d\n"
           "  printRate: %d\n"
           "  Time step: %g fs\n"
           "  Initial Temperature: %g K\n"
           "  Initial Delta: %g Angstroms\n"
           "\n",
           cmd->doeam,
           cmd->potDir,
           cmd->potName,
           cmd->potType,
           cmd->nx, cmd->ny, cmd->nz,
           cmd->xproc, cmd->yproc, cmd->zproc,
           cmd->lat,
           cmd->nSteps,
           cmd->printRate,
           cmd->dt,
           cmd->temperature,
           cmd->initialDelta
   );
   fflush(file);
}

