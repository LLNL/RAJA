/// \file
/// Write simulation information in YAML format.
///
/// Information regarding platform, run parameters, performance, etc.,
/// are written to a file whose name is generated from the CoMDVariant
/// and the time of the run.  This provides a simple mechanism to track
/// and compare performance etc.
///
/// There are much more sophisticated libraries and routines available
/// to handle YAML, but this simple implemenation handles everything we
/// really need.

#include "yamlOutput.h"

#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <omp.h>

#include "CoMD_info.h"
#include "mytype.h"
#include "parallel.h"

FILE* yamlFile = NULL;

static const char* CoMDVersion = "1.1";
static const char* CoMDVariant = CoMD_VARIANT;

static void getTimeString(char* timestring)
{
   time_t rawtime;
   struct tm* timeinfo;
   time(&rawtime);
   timeinfo = localtime(&rawtime);

   sprintf(timestring,
           "%4d-%02i-%02d, %02d:%02d:%02d",
           timeinfo->tm_year+1900,
           timeinfo->tm_mon+1,
           timeinfo->tm_mday,
           timeinfo->tm_hour,
           timeinfo->tm_min,
           timeinfo->tm_sec);
}

void yamlBegin(void)
{
   if (! printRank())
      return;

   char filename[64];
   time_t rawtime;
   time (&rawtime);
   struct tm* ptm = localtime(&rawtime);
   char sdate[25];
  //use tm_mon+1 because tm_mon is 0 .. 11 instead of 1 .. 12
   sprintf (sdate,"%04d:%02d:%02d-%02d:%02d:%02d",
            ptm->tm_year + 1900, ptm->tm_mon+1,
            ptm->tm_mday, ptm->tm_hour, ptm->tm_min,ptm->tm_sec);
   sprintf(filename, "%s.%s.yaml", CoMDVariant, sdate);
   yamlFile = fopen(filename, "w");
}

void yamlAppInfo(FILE* file)
{

   int numThreads = omp_get_max_threads();

   if (! printRank())
      return;
   printSeparator(file);
   fprintf(file,"Mini-Application Name    : %s\n", CoMDVariant);
   fprintf(file,"Mini-Application Version : %s\n", CoMDVersion);
   fprintf(file,"Platform:\n");
   fprintf(file,"  hostname: %s\n",         CoMD_HOSTNAME);
   fprintf(file,"  kernel name: %s\n",      CoMD_KERNEL_NAME);
   fprintf(file,"  kernel release: %s\n",   CoMD_KERNEL_RELEASE);
   fprintf(file,"  processor: %s\n",        CoMD_PROCESSOR);
   fprintf(file,"Build:\n");
   fprintf(file,"  CC: %s\n",               CoMD_COMPILER);
   fprintf(file,"  compiler version: %s\n", CoMD_COMPILER_VERSION);
   fprintf(file,"  CFLAGS: %s\n",           CoMD_CFLAGS);
   fprintf(file,"  LDFLAGS: %s\n",          CoMD_LDFLAGS);
   fprintf(file,"  using MPI: %s\n",        builtWithMpi() ? "true":"false");
   fprintf(file,"  Threading: OpenMP (%d threads) \n", numThreads);
   fprintf(file,"  Double Precision: %s\n", (sizeof(real_t)==sizeof(double)?"true":"false"));
   char timestring[32];
   getTimeString(timestring);
   fprintf(file,"Run Date/Time: %s\n", timestring);
   fprintf(file, "\n");
   fflush(file);
}

void yamlEnd(void)
{
   if (! printRank())
      return;

   fclose(yamlFile);
}

void printSeparator(FILE* file)
{
      //fprintf(file,"=========================================================================\n");
      fprintf(file,"\n");
}

