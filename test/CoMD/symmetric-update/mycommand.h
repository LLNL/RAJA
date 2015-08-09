/// \file
/// Handle command line arguments.

#ifndef MYCOMMAND_H
#define MYCOMMAND_H

#include <stdio.h>

/// A structure to hold the value of every run-time parameter that can
/// be read from the command line.
typedef struct CommandSt
{
   char potDir[1024];  //!< the directory where EAM potentials reside
   char potName[1024]; //!< the name of the potential
   char potType[1024]; //!< the type of the potential (funcfl or setfl)
   int doeam;          //!< a flag to determine whether we're running EAM potentials
   int nx;             //!< number of unit cells in x
   int ny;             //!< number of unit cells in y
   int nz;             //!< number of unit cells in z
   int xproc;          //!< number of processors in x direction
   int yproc;          //!< number of processors in y direction
   int zproc;          //!< number of processors in z direction
   int nSteps;         //!< number of time steps to run
   int printRate;      //!< number of steps between output
   double dt;          //!< time step (in femtoseconds)
   double lat;         //!< lattice constant (in Angstroms)
   double temperature; //!< simulation initial temperature (in Kelvin)
   double initialDelta; //!< magnitude of initial displacement from lattice (in Angstroms)
} Command;

/// Process command line arguments into an easy to handle structure.
Command parseCommandLine(int argc, char** argv);

/// Print run parameters in yaml format on the supplied output stream.
void printCmdYaml(FILE* file, Command* cmd);

#endif
