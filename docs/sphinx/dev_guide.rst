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
Developer Guide
===================================

.. warning:: This section will contain useful information for core developers
             and others who want to contribute (e.g., the RAJA code strucure,
             where to put new things, adding unit tests, etc. The text below is 
             from the original sphinx docs. I kept it so we can use bits that 
             we want to keep.

This guide is intended for people who want to work on RAJA and contribute
to its development. It also contains some useful information for users
who may need to look through the source code.

Code Structure
---------------

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

Unit Testing
------------

The directory 'raja/test/unit-tests' has two subdirectories, each of which
contains files that run various traversal and reduction operations for RAJA
IndexSets and Segments. All RAJA "forall" template and execution policy
options that are available for a given compiler are included. Running these
tests is a good sanity check that the code is built correctly and works. The
two subdirectories are:

  * **CPUtests.** It contains codes to run traversl and reduction tests for
    sequential, OpenMP, and CilkPlus (if available) execution policies.

  * **GPUtests.** It contains codes to run traversl and reduction tests for
    GPU CUDA execution policies. Note that these tests use Unified Memory
    to simplify host-device memory transfers.

**NOTE:** RAJA must be built with CUDA enabled to generate GPU variants.
When running CUDA variants of the tests and examples, we advise you to set the
environment variable CUDA_VISIBLE_DEVICES to zero before running.

For example, for C-shell users ::

   $ setenv CUDA_VISIBLE_DEVICES 0

We are using CUDA Unified Memory and we find that this environment setting
greatly improves performance.

Performance Testing
--------------------
