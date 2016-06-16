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
Configuring and Building RAJA 
===================================

This section describes RAJA configuration options and how to build RAJA. 

It is important to note that RAJA requires compiler support for lambda 
functions and a few other C++11 features. Please make sure your compiler
is C++11-compliant before attempting to build RAJA.

CMake System
-----------------

RAJA uses a simple CMake-based system, but you don't need to know much 
about CMake to build RAJA. It does require that you use a CMake version 
greater than 2.8. If needed, you can learn more about CMake and download
it at `<http://www.cmake.org>`_

You can build RAJA like any other CMake project. The simplest way to build 
it is to do the following in the top-level RAJA directory:

.. code-block:: sh

    $ mkdir build
    $ cd build
    $ cmake ../
    $ make

This will build RAJA and the tests it contains using default settings 
based on the system and compiler choices (described below).

Note that the RAJA CMake system does not allow in-source builds so that 
you can easily configure and build for various systems and compilers from 
the same source code.

Configuration Options
----------------------

The RAJA configuration can be set using standard CMake variables along with
RAJA-specific variables. For example, to make a release build with some 
system default GNU compiler and then install the RAJA header files and
libraries in a specific directory location, you could do the following in 
the top-level RAJA directory:

.. code-block:: sh

    $ mkdir build-gnu-release
    $ cd build-gnu-release
    $ cmake \
      -DCMAKE_C_COMPILER=gcc \
      -DCMAKE_CXX_COMPILER=g++ \
      -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_INSTALL_PREFIX=../install-gnu-release ../
    $ make
    $ make install

Following CMake conventions, RAJA supports three build types: 'Release', 
'RelWithDebInfo', and 'Debug'. Depending on which you choose, you will not
have to explicitly add the '-g' compiler flags to generate debugging symbols.
The default compiler flags for optimization, etc. for various compilers can
be found in the `cmake/SetupCompilers.cmake` file.

RAJA uses a variety of custom variables to control how it is compiled. Many 
of these manifest as compilation directives and definitions that appear in 
the RAJA 'config.hxx' file that is generated during the CMake process. The
RAJA configuration header file is included in other RAJA headers as needed
so all options propagate through the build process consistently.

These variables are turned on and off similar to standard CMake variables; 
e.g., to enable RAJA OpenMP functionality, add this CMake option ::

    -DRAJA_ENABLE_OPENMP=On

The following list describes the RAJA CMake variables and their defaults.

  * **Tests**

     The variable that controls whether RAJA tests are compiled with the 
     library is:

      ======================   ======================
      Variable                 Default
      ======================   ======================
      RAJA_ENABLE_TESTS        On 
      ======================   ======================
     
  * **Programming Models**

     Variables that control which RAJA programming model back-ends are enabled
     are (names are descriptive of what they enable):

      ======================   ======================
      Variable                 Default
      ======================   ======================
      RAJA_ENABLE_OPENMP       On 
      RAJA_ENABLE_CUDA         Off 
      RAJA_ENABLE_CILK         Off 
      ======================   ======================

  * **Data Types, Sizes, Alignment Parameters, etc.**

     RAJA provides typedefs that can be used to parametrize floating 
     point types in applications and easily switch between types. Exactly 
     one of these can be on at a time. The defaults should be reasonable 
     for most situations. For more details on these options, please see 
     the header file real_datatypes.hxx.

      ======================   ======================
      Variable                 Default
      ======================   ======================
      RAJA_USE_DOUBLE          On 
      RAJA_USE_FLOAT           Off 
      ======================   ======================

     Similarly, there is a typedef for complex data types that can be enabled 
     if needed.

      ======================   ======================
      Variable                 Default
      ======================   ======================
      RAJA_USE_COMPLEX         Off 
      ======================   ======================

     There are several variables that control RAJA floating-point data
     pointer typedefs. Exactly one of these can be on at a time. When
     RAJA is compiled for CPU execution only, the defaults are:

      =============================   ======================
      Variable                        Default
      =============================   ======================
      RAJA_USE_BARE_PTR               Off
      RAJA_USE_RESTRICT_PTR           On
      RAJA_USE_RESTRICT_ALIGNED_PTR   Off
      RAJA_USE_PTR_CLASS              Off
      =============================   ======================

     When RAJA is compiled with CUDA enabled, the defaults are:

      =============================   ======================
      Variable                        Default
      =============================   ======================
      RAJA_USE_BARE_PTR               On
      RAJA_USE_RESTRICT_PTR           Off
      RAJA_USE_RESTRICT_ALIGNED_PTR   Off
      RAJA_USE_PTR_CLASS              Off
      =============================   ======================

     What these variables mean:

      =============================   ========================================
      Variable                        Meaning
      =============================   ========================================
      RAJA_USE_BARE_PTR               Use standard C-style pointer
      RAJA_USE_RESTRICT_PTR           Use C-style pointer with restrict
                                      qualifier
      RAJA_USE_RESTRICT_ALIGNED_PTR   Use C-style pointer with restrict
                                      qualifier and alignment attribute 
                                      (see RAJA_DATA_ALIGN below)
      RAJA_USE_PTR_CLASS              Use pointer class with overloaded `[]` 
                                      operator that applies restrict and 
                                      alignment intrinsics. This is useful 
                                      when a compiler does not support 
                                      attributes in a typedef.
      =============================   ========================================

     RAJA internally uses parameters to define platform-specific constants 
     for index ranges and data alignment. The variables that control these
     are:

      =============================   ======================
      Variable                        Default
      =============================   ======================
      RAJA_RANGE_ALIGN                4
      RAJA_RANGE_MIN_LENGTH           32
      RAJA_DATA_ALIGN                 64
      RAJA_COHERENCE_BLOCK_SIZE       64
      =============================   ======================

     What these variables mean:

      =============================   ========================================
      Variable                        Meaning
      =============================   ========================================
      RAJA_RANGE_ALIGN                Constrain alignment of begin/end indices 
                                      of range segments generated by index set 
                                      builder methods; i.e., begin and end 
                                      indices of such segments will be 
                                      multiples of this value.
      RAJA_RANGE_MIN_LENGTH           Sets minimum length of range segments 
                                      generated by index set builder methods.
                                      This should be an integer multiple of 
                                      RAJA_RANGE_ALIGN.
      RAJA_DATA_ALIGN                 Specifies data alignment used in 
                                      intrinsics and typedefs; 
                                      units of **bytes**.
      RAJA_COHERENCE_BLOCK_SIZE       Defines thread coherence value for 
                                      shared memory blocks used by RAJA 
                                      reduction objects.
      =============================   ========================================

  * **Timer Options**

     RAJA provides a simple timer class that is used in RAJA example codes
     to determine execution timing and can be used in other apps as well.
     Three variables are available to select the timing mechanism used.
     Exactly one of these can be on at a time.

      ======================   ======================
      Variable                 Default
      ======================   ======================
      RAJA_USE_GETTIME         On 
      RAJA_USE_CLOCK           Off 
      RAJA_USE_CYCLE           Off 
      ======================   ======================

     What these variables mean:

      =============================   ========================================
      Variable                        Meaning
      =============================   ========================================
      RAJA_USE_GETTIME                Use `timespec` from the C standard 
                                      library time.h file
      RAJA_USE_CLOCK                  Use `clock_t` from time.h
      RAJA_USE_CYCLE                  Use `ticks` from the cycle.h file 
                                      borrowed from the FFTW library
      =============================   ========================================

  * **Other RAJA Features**
    
     RAJA contains features that are turned off by default since they may
     not be of interest to all RAJA users. The variables that enable/disable
     these features are described below.

     The RAJA *forallN* nested-loop traversals are controlled with the 
     following variable:
     
      =============================   ========================================
      Variable                        Meaning
      =============================   ========================================
      RAJA_ENABLE_NESTED              Enable/disable nested loop functionality
      =============================   ========================================

     RAJA has an experimental loop-level fault tolerance model which is 
     controlled by the following variables:

      =============================   ========================================
      Variable                        Meaning
      =============================   ========================================
      RAJA_ENABLE_FT                  Enable/disable fault-tolerance mechanism
      RAJA_REPORT_FT                  Enable/disable a report of fault-
                                      tolerance enabled run (e.g., number of 
                                      faults detected, recovered from, 
                                      recovery overhead, etc.)
      =============================   ========================================

Host-Config Files
----------------------

The 'host-configs' directory contains subdirectories with files that define 
configurations for various platforms and compilers at LLNL. These *host-config*
files can be passed to CMake using the '-C' option, which initializes the CMake
cache with the configuration specified in each file.  For example, to use
the host-config file for GNU compiler on LLNL LC Linux systems, one could
do the following from the top-level RAJA directory:

.. code-block:: sh

    $ mkdir my-builds
    $ cd my-builds
    $ mkdir build-gcc-4.9.3-release
    $ cd build-gnu-4.9.3-release
    $ cmake \
      -C ../../host-configs/chaos/gcc_4_9_3.cmake \
      -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_INSTALL_PREFIX=../install-gcc-4.9.3-release \
      ../..
    $ make

The host-config files can be easily modified to suit other configurations 
as desired.

The `scripts` directory contains several bash shell scripts that are set up
to use the host-config files. For example, you can type the following commands
starting at the top-level RAJA directory to build a version of RAJA for 
specific versions of the GNU and Intel compilers in a build subdirectory:

.. code-block:: sh

    $ mkdir my-builds
    $ cd my-builds
    $ ../scripts/gcc-4.9.3.sh 
    $ cd build-gnu-4.9.3-release
    $ make
    $ cd ..
    $ ../scripts/icpc-16.0.109.sh
    $ cd build-icpc-16.0.109-release
    $ make

These scripts serve as useful examples for those who are not fluent in CMake.

Did I build RAJA correctly?
---------------------------

You can verify that RAJA is built correctly with the options you want, you 
can run some unit tests...

.. warning:: Need to add a 'make tests' or 'make check' target that 
             compiles (if needed) and runs some basic tests with sensible 
             output that makes it clear to users that their RAJA build is
             good to go or is not.

