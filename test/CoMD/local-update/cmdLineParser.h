/// \file 
/// A parser for command line arguments.
///
/// \author Sriram Swaminarayan
/// \date July 24, 2007

#ifndef CMDLINEPARSER_H_
#define CMDLINEPARSER_H_

/// Specifies a command line argument that should be accepted by the program.
/// \param [in]  longOption  The long name of option i.e., --optionname
/// \param [in]  shortOption The short name of option i.e., -o
/// \param [in]  has_arg  Whether this option has an argument i.e., -o value.
///                       If has_arg is 0, then dataPtr must be an integer
///                       pointer.
/// \param [in]  type  The type of the argument. Valid values are:
///                    -  i   integer
///                    -  f   float
///                    -  d   double
///                    -  s   string
///                    -  c   character
///
/// \param [in]  dataPtr  A pointer to where the value will be stored.
/// \param [in]  dataSize The length of dataPtr, only useful for character
///                       strings.
/// \param [in]  help     A short help string, preferably a single line or
///                       less.
int addArg(const char *longOption, const char shortOption,
           int has_arg, const char type, void *dataPtr, int dataSize,
           const char *help);

/// Call this to process your arguments.
void processArgs(int argc, char **argv);

/// Prints the arguments to the stdout stream.
void printArgs(void);

void freeArgs(void);

#endif
