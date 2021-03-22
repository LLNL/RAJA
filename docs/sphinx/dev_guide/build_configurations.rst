.. ##
.. ## Copyright (c) 2016-21, Lawrence Livermore National Security, LLC
.. ## and RAJA project contributors. See the RAJA/COPYRIGHT file
.. ## for details.
.. ##
.. ## SPDX-License-Identifier: (BSD-3-Clause)
.. ##

.. host_config:

**************************
RAJA Build Configurations
**************************

RAJA must be built and tested with a wide range of compilers and with 
all of its supported back-ends. The project currently maintains two 
ways to build and test important configurations in a reproducible manner:

  * **Build scripts.** The RAJA source repository contains a collection of
    simple build scripts that are used to generate build configurations 
    for platforms in the Livermore Computing Center primarily.
    
  * **Generated host-config files.** The RAJA repository includes a 
    mechanism to generate *host-config* files (i.e., CMake cache files)
    using `Spack <https://github.com/spack/spack>`_.

Each of these specifies compiler versions and options, a build target 
(Release, Debug, etc.), RAJA features to enable (OpenMP, CUDA, etc.), 
and paths to required tool chains, such as CUDA, ROCm, etc.

They are described briefly in the following sections.


===================
RAJA Build Scripts
===================

The build scripts in the RAJA ``scripts`` directory are used mostly by RAJA 
developers to quickly create a build environment to compile and run tests
during code development. 

Each script can be executed from the top-level RAJA directory. When a script
is run, it creates a uniquely-named build directory in RAJA and runs CMake 
with arguments contained in the script to create a build environment in the
new directory. One then goes into the directory and runs make to build RAJA, 
its tests, example codes, etc.  For example,

.. code-block:: bash

  $ ./scripts/lc-builds/toss3_clang10.0.1.sh
  $ cd build_lc_toss3-clang-10.0.1
  $ make -j
  $ make test

Eventually, these scripts may go away and be superceded by the Spack-based
host-config file generation process when that achieves the level of
compiler coverage that the scripts have.


============================
Generated Host-Config Files
============================

The RAJA repository contains two submodules 
`uberenv <https://github.com/LLNL/uberenv>`_ and
`radiuss-spack-configs <https://github.com/LLNL/radiuss-spack-configs>`_ that 
work together to generate host-config files. These are projects in the 
GitHub LLNL organization and contain utilities shared by various projects. 
The main uberenv script can be used to drive Spack to generate a *host-config* 
file that contains all the information required to define a RAJA build 
environment. The host-config file can then be passed to CMake using the '-C' 
option to create a build configuration. *Spack specs* defining compiler 
configurations are maintained in files in the radiuss-spack-configs 
repository.

RAJA shares its uberenv workflow with other projects. The documentation 
for this is available in `RADIUSS Uberenv Guide <https://radiuss-ci.readthedocs.io/en/latest/uberenv.html#uberenv-guide>`_.


Generating a RAJA host-config file
------------------------------------

This section describes the host-config file generation process for RAJA.

Machine specific configurations
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Compiler configurations for Livermore computer platforms are contained in 
in sub-directories in the RAJA ``scripts/uberenv/spack_configs`` directory:

.. code-block:: bash

  $ ls -c1 ./scripts/uberenv/spack_configs
  blueos_3_ppc64le_ib
  darwin
  toss_3_x86_64_ib
  blueos_3_ppc64le_ib_p9
  config.yaml

To see currently supported configurations, please see the contents of the 
``compilers.yaml`` file in each of these sub-directories.

Generating a host-config file
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The main uberenv python script can be invoked from the top-level RAJA directory
to generate a host-config file for a desired configuration. For example,

.. code-block:: bash

  $ python ./scripts/uberenv/uberenv.py --spec="%gcc@8.1.0"
  $ python ./scripts/uberenv/uberenv.py --spec="%gcc@8.1.0~shared+openmp tests=benchmarks"

Each command generates a corresponding host-config file in the top-level RAJA 
directory. The file name contains the platform and OS to which it applies, and 
the compiler and version. For example,

.. code-block:: bash

  hc-quartz-toss_3_x86_64_ib-gcc@8.1.0-fjcjwd6ec3uen5rh6msdqujydsj74ubf.cmake

Specs that are exercised during the Gitlab CI process are found YAML files in 
the ``RAJA/.gitlab`` directory. See :ref:`vettedspecs-label` for more 
information.

Building RAJA with a generated host-config file
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To build RAJA with one of these host-config files, create a build directory and
run CMake in it by passing the host-config file to CMake using the '-C' option.
Then, run make and RAJA tests, if desired, to make sure the build was done
properly:

.. code-block:: bash

  $ mkdir <build dirname> && cd <build dirname>
  $ cmake -C <path_to>/<host-config>.cmake ..
  $ cmake --build -j .
  $ ctest --output-on-failure -T test

It is also possible to use the configuration with a RAJA CI script outside 
of the normal CI process:

.. code-block:: bash

  $ HOST_CONFIG=<path_to>/<host-config>.cmake ./scripts/gitlab/build_and_test.sh

MacOS
^^^^^

In RAJA, the Spack configuration for MacOS contains the default compiler
corresponding to the OS version (`compilers.yaml`), and a commented section to 
illustrate how to add `CMake` as an external package. You may install CMake 
with `Homebrew <https://brew.sh>`_, for example, and follow the process 
outlined above after it is installed.
