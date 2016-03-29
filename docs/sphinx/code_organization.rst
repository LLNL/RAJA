.. ##
.. ## Copyright (c) 2016, Lawrence Livermore National Security, LLC.
.. ##
.. ## Produced at the Lawrence Livermore National Laboratory.
.. ##
.. ## All rights reserved.
.. ##
.. ## For release details and restrictions, please see raja/README-license.txt
.. ##


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

The main RAJA header 'RAJA.hxx' is all that needs to be 
included in an application that uses RAJA. That is, the line ::

  #include "RAJA/RAJA.hxx"

in your application code will include all other RAJA header files that are 
needed for a particular configuration and build of the RAJA code.

Header files that define general RAJA capabilities that don't depend on the 
underlying programming model choice live in the 'include/RAJA' directory.
The directory contains header files for key RAJA classes, such as IndexSet and 
Segment types. Files that define generic RAJA traversals, which work for any 
IndexSet, Segment or execution policy type, also reside there; e.g., 
'forall_generic.hxx'.

RAJA traversals, reductions, and execution policies for individual programming
models are defined in header files that live in subdirectories of
the 'include/RAJA' directory. Currently, these directories are as follows
(their names are descriptive of the programming model supported by their
contents):

  * exec-cilk
  * exec-cuda
  * exec-openmp
  * exec-sequential
  * exec-simd

Each of these directories contains files that defines traversal and
reduction execution policies, *forall* traversal template specializations 
based on the execution policies, and *reduction operation* template 
specializations based on the reduction policies. For example, the OpenMP 
directory has the header files:

  * raja_openmp.hxx
  * forall_openmp.hxx
  * forallN_openmp.hxx
  * reduce_openmp.hxx 

Note that SIMD execution shares reduction types with sequential execution, 
so the 'exec-simd' directory does not contain a reduction header file. 

