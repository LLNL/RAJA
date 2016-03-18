
===================================
RAJA source code organization
===================================

The RAJA root directory has several subdirectories with contents as follows:

  * **cmake:** modules used for the CMake-based build system.
  * **docs:** user documentation.
  * **host-configs:** configuration files to initialize the CMake cache
    for machines and compilers most commonly used in the Livermore Computing 
    Center at Lawrence Livermore National Laboratory.
  * **include:** header files defining the RAJA API.
  * **scripts:** helper scripts, currently only contains a python script
    that simplifies the library configuration process.
  * **src:** source files that are compiled.
  * **test:** simple test codes and proxy-app examples that illustrate RAJA 
    usage.

For most users, the most important directories are 'include' and 'test'.
RAJA is largely a header file library; the 'include/RAJA' directory
contains most of what you need to understand to start using RAJA in an
application. We discuss the contents of the 'test' directory in a later section.

The 'include/RAJA' directory contains the header files defining interfaces for
key RAJA classes, such as IndexSet and Segment types, and methods that are
largely independent of different RAJA programming model backends. The main 
RAJA header is 'RAJA.hxx', which is all that needs to be included in an
application that uses RAJA. That is, the line ::

  #include "RAJA/RAJA.hxx"

in your application code will include all other RAJA header files that are 
needed for a particular configuration and build of the RAJA code.

RAJA traversals, reductions, and execution policies for individual programming
model backends are defined in header files that live in subdirectories of
the 'include/RAJA' directory. Currently, these directories are:

  * exec-cilk
  * exec-cuda
  * exec-openmp
  * exec-sequential
  * exec-simd

The directory names are descriptive of their contents.
