.. ##
.. ## Copyright (c) 2016-18, Lawrence Livermore National Security, LLC.
.. ##
.. ## Produced at the Lawrence Livermore National Laboratory
.. ##
.. ## LLNL-CODE-689114
.. ##
.. ## All rights reserved.
.. ##
.. ## This file is part of RAJA.
.. ##
.. ## For details about use and distribution, please read RAJA/LICENSE.
.. ##

.. _configopt-label:

***********************
Configuration Options
***********************

RAJA uses `BLT <https://github.com/LLNL/blt>`_, a CMake-based build system .
In :ref:`getting_started-label`, we described how to run CMake to configure
RAJA with its default option settings.  In this section, we describe all RAJA
configuration options, their defaults, and how to enable desired features.

=======================
Setting Options
=======================

The RAJA configuration can be set using standard CMake variables along with
BLT and RAJA-specific variables. For example, to make a release build with 
some system default GNU compiler and then install the RAJA header files and
libraries in a specific directory location, you could do the following in 
the top-level RAJA directory::

    $ mkdir build-gnu-release
    $ cd build-gnu-release
    $ cmake \
      -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_C_COMPILER=gcc \
      -DCMAKE_CXX_COMPILER=g++ \
      -DCMAKE_INSTALL_PREFIX=../install-gnu-release ../
    $ make
    $ make install

Following CMake conventions, RAJA supports three build types: 'Release', 
'RelWithDebInfo', and 'Debug'. Similar to other CMake systems, when you
choose a build type that includes debug information, you do not have to specify
the '-g' compiler flag to generate debugging symbols. 

All RAJA options are set like standard CMake variables. For example, to enable 
RAJA OpenMP functionality, pass the following argument to cmake::

    -DENABLE_OPENMP=On

All RAJA settings for default options, compilers, flags for optimization, etc. 
can be found in files in the `RAJA/cmake` directory. Next, we
summarize the available options and their defaults

=================================
Available Options and Defaults
=================================

RAJA uses a variety of custom variables to control how it is compiled. Many 
of these are used internally to control how RAJA gets compiled and do 
not need to be set by users. Others can be turned on or off by users to 
enable or disable certain RAJA features. Most variables get translated to 
compiler directives and definitions in the RAJA 'config.hpp' file that is 
generated when CMake runs. The 'config.hpp' header file is included in other 
RAJA headers as needed so all options propagate consistently through the 
build process for all of the code. Each RAJA variable has a special prefix 
to distinguish it as being specific to RAJA; i.e., it is not a BLT variable
or a standard CMake variable.

The following tables describe which variables set RAJA options and 
and their default settings:

* **Examples, tests, warnings, etc.**

     Variables that control whether RAJA tests and examples are built when
     the library is compiled are:

      ======================   ======================
      Variable                 Default
      ======================   ======================
      ENABLE_TESTS             On 
      ENABLE_EXAMPLES          On 
      ======================   ======================

     RAJA can also be configured to build with compiler warnings reported as
     errors, which may be useful when using RAJA in an application:

      =========================   ======================
      Variable                    Default
      =========================   ======================
      ENABLE_WARNINGS_AS_ERRORS   Off
      =========================   ======================
     
* **Programming models and compilers**

     Variables that control which RAJA programming model back-ends are enabled
     are (names are descriptive of what they enable):

      ======================   ======================
      Variable                 Default
      ======================   ======================
      ENABLE_OPENMP            On 
      ENABLE_TARGET_OPENMP     Off 
      ENABLE_CUDA              Off 
      ENABLE_TBB               Off 
      ======================   ======================

     Other compilation options are available via the following:

      ======================   ======================
      Variable                 Default
      ======================   ======================
      ENABLE_CLANG_CUDA        Off
      ENABLE_CUB               On (when CUDA enabled)
      ======================   ======================

      Turning the 'ENABLE_CLANG_CUDA' variable on will build CUDA code with
      the native support in the Clang compiler. When using it, the 
      'ENABLE_CUDA' variable must also be turned on.

      The 'ENABLE_CUB' variable is used to enable NVIDIA cub library support
      for RAJA CUDA scans. When turned off, NVIDIA thrust is used by default.

.. note:: When using the NVIDIA nvcc compiler for RAJA CUDA functionality, 
          the variable 'RAJA_NVCC_FLAGS' should be used to pass flags to nvcc.

* **Data types, sizes, alignment, etc.**

     RAJA provides type aliases that can be used to parametrize floating 
     point types in applications, which makes it easy to switch between types. 

     The following variables are used to set the type for 'RAJA::Real_type':

      ======================   ======================
      Variable                 Default
      ======================   ======================
      RAJA_USE_DOUBLE          On 
      RAJA_USE_FLOAT           Off 
      ======================   ======================

     Similarly, the 'RAJA::Complex_type' can be enabled to support complex 
     numbers if needed:

      ======================   ======================
      Variable                 Default
      ======================   ======================
      RAJA_USE_COMPLEX         Off 
      ======================   ======================

     When turned on, the RAJA Complex_type is 'std::complex<Real_type>'.

     There are several variables to control the definition of the RAJA 
     floating-point data pointer type 'RAJA::Real_ptr'. The base data type
     is always 'Real_type'. When RAJA is compiled for CPU execution 
     only, the defaults are:

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

     The meaning of these variables is:

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
      =============================   ========================================

     For details on the options in this section are used, please see the 
     header file `RAJA/include/RAJA/util/types.hpp`.

* **Timer Options**

     RAJA provides a simple portable timer class that is used in RAJA
     example codes to determine execution timing and can be used in other apps
     as well. This timer can use any of three internal timers depending on
     your preferences, and one should be selected by setting the `RAJA_TIMER`
     variable. If the `RAJA_CALIPER` variable is turned on (off by default), 
     the timer will also offer caliper-based region annotations.

      ======================   ======================
      Variable                 Values
      ======================   ======================
      RAJA_TIMER               chrono (default)
                               gettime
                               clock
      ======================   ======================

     What these variables mean:

      =============================   ========================================
      Value                           Meaning
      =============================   ========================================
      chrono                          Use the std::chrono library from the STL
      gettime                         Use `timespec` from the C standard 
                                      library time.h file
      clock                           Use `clock_t` from time.h
      =============================   ========================================

* **Other RAJA Features**
   
     RAJA contains some features that are used mainly for development or are 
     not of general interest to RAJA users. These are turned off be default.
     They are described here for reference and completeness.

      =============================   ========================================
      Variable                        Meaning
      =============================   ========================================
      ENABLE_CHAI                     Enable/disable RAJA internal support for
                                      `CHAI <https://github.com/LLNL/CHAI>`_ 
      ENABLE_FT                       Enable/disable RAJA experimental
                                      loop-level fault-tolerance mechanism
      RAJA_REPORT_FT                  Enable/disable a report of fault-
                                      tolerance enabled run (e.g., number of 
                                      faults detected, recovered from, 
                                      recovery overhead, etc.)
      =============================   ========================================

=======================
RAJA Host-Config Files
=======================

The `RAJA/host-configs` directory contains subdirectories with files that 
define configurations for various platforms and compilers at LLNL. These
serve as examples of  *CMake cache files* that can be passed to CMake using 
the '-C' option. This option initializes the CMake cache with the configuration 
specified in each file. For examples of how they are used for specific CMake
configurations, see the files in the `RAJA/scripts` directory.
