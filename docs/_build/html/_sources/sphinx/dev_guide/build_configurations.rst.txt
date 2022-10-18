.. ##
.. ## Copyright (c) 2016-22, Lawrence Livermore National Security, LLC
.. ## and RAJA project contributors. See the RAJA/LICENSE file
.. ## for details.
.. ##
.. ## SPDX-License-Identifier: (BSD-3-Clause)
.. ##

.. _build_config-label:

**************************
RAJA Build Configurations
**************************

To meet user needs, RAJA is built and tested with a wide range of compilers for 
all of its supported back-ends. Automated continuous integration (CI) testing 
employed by the project is described in :ref:`ci-label`. During day-to-day
development, the project currently maintains two ways to build and test 
configurations in a reproducible manner:

  * **Build scripts.** The RAJA source repository contains a collection of
    simple build scripts that are used to generate build configurations 
    for a variety of platforms, such as Livermore Computing (LC) systems,
    MacOS, and Linux environments.
    
  * **Generated host-config files.** The RAJA repository includes a 
    mechanism to generate *host-config* files (i.e., CMake cache files)
    using `Spack <https://github.com/spack/spack>`_.

The configurations specify compiler versions and options, build targets 
(Release, Debug, etc.), RAJA features to enable (OpenMP, CUDA, HIP, etc.), 
and paths to required tool chains, such as CUDA, ROCm, etc.  
They are described briefly in the following sections.


.. _build_scripts-label:

===================
RAJA Build Scripts
===================

Build scripts mentioned above live in the 
`RAJA/scripts <https://github.com/LLNL/RAJA/tree/develop/scripts>`_ directory. 
Each script is executed from the top-level RAJA directory. The scripts for
CPU-only platforms require an argument that indicates the compiler version.
For example,

.. code-block:: bash

  $ ./scripts/lc-builds/toss3_clang.sh 10.0.1

Scripts for GPU-enabled platforms require three arguments: the device
compiler version, the target compute architecture, and the host
compiler version. For example,

.. code-block:: bash

  $ ./scripts/lc-builds/blueos_nvcc_gcc.sh 10.2.89 sm_70 8.3.1

When a script is run, it creates a build directory named for the configuration
in the top-level RAJA directory and runs CMake with arguments contained in the 
script to create a build environment in the new directory. One then goes into 
that directory and runs 'make' to build RAJA, and depending on options
passed to CMake RAJA tests, example codes, etc.  For example,

.. code-block:: bash

  $ ./scripts/lc-builds/blueos_nvcc_gcc.sh 10.2.89 sm_70 8.3.1
  $ cd build_lc_blueos-nvcc10.2.89-sm_70-gcc8.3.1
  $ make -j
  $ make test

.. _spack_host_config-label:

==================================
Spack-Generated Host-Config Files
==================================

The RAJA repository contains two submodules 
`uberenv <https://github.com/LLNL/uberenv>`_ and
`radiuss-spack-configs <https://github.com/LLNL/radiuss-spack-configs>`_ that 
work together to generate host-config files. These are projects in the 
GitHub LLNL organization and contain utilities shared and maintained by 
various projects. The main uberenv script is used to drive Spack to generate 
a *host-config* file (i.e., a CMake *cache* file) that contains all the 
information required to define a RAJA build environment. The generated file 
can then be passed to CMake using the '-C' option to create a build 
configuration. *Spack specs* defining compiler configurations are maintained 
in files in the radiuss-spack-configs repository.

Additional documentation for this process is available in the
`RADIUSS Uberenv Guide <https://radiuss-ci.readthedocs.io/en/latest/uberenv.html#uberenv-guide>`_.


Generating a RAJA host-config file
------------------------------------

This section describes the host-config file generation process for RAJA.

Platform configurations
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Compiler configurations for Livermore computer platforms are contained
in sub-directories of the ``RAJA/scripts/radiuss-spack-configs`` submodule
directory:

.. code-block:: bash

  $ ls -c1 ./scripts/radiuss-spack-configs
  toss_4_x86_64_ib_cray
  toss_4_x86_64_ib
  toss_3_x86_64_ib
  blueos_3_ppc64le_ib
  darwin
  config.yaml
  blueos_3_ppc64le_ib_p9
  ...

To see available configurations, please see the contents of the 
``compilers.yaml`` and ``packages.yaml`` files in each sub-directory.

Generating a host-config file
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The ``uberenv.py`` python script can be run from the top-level RAJA directory
to generate a host-config file for a desired configuration. For example,

.. code-block:: bash

  $ python3 ./scripts/uberenv/uberenv.py --spec="%gcc@8.1.0"
  $ python3 ./scripts/uberenv/uberenv.py --spec="%gcc@8.1.0~shared+openmp tests=benchmarks"

Each command generates a corresponding host-config file in the top-level RAJA 
directory. The file name contains the platform and OS to which it applies, and 
the compiler and version. For example,

.. code-block:: bash

  hc-quartz-toss_3_x86_64_ib-gcc@8.1.0-fjcjwd6ec3uen5rh6msdqujydsj74ubf.cmake

This process is also used by our Gitlab CI testing effort. 
See :ref:`ci-label` for more information.

Building RAJA with a generated host-config file
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To build RAJA with one of these host-config files, create a build directory and
run CMake in it by passing a host-config file to CMake using the '-C' option.
Then, run 'make' to build RAJA. To ensure the build was successful, you may
want to run the RAJA tests. For example,

.. code-block:: bash

  $ mkdir <build dirname> && cd <build dirname>
  $ cmake -C <path_to>/<host-config>.cmake ..
  $ cmake --build -j .
  $ ctest --output-on-failure -T test

You may also run the RAJA tests with the command

.. code-block:: bash

  $ make test

as an alternative to the 'ctest' command used above.

It is also possible to use the configuration with the RAJA Gitlab CI script 
outside of the Gitlab environment:

.. code-block:: bash

  $ HOST_CONFIG=<path_to>/<host-config>.cmake ./scripts/gitlab/build_and_test.sh

MacOS
^^^^^

In RAJA, the Spack configuration for MacOS contains the default compiler
corresponding to the OS version in the ``compilers.yaml`` file in the 
``RAJA/scripts/radiuss-spack-configs/darwin/`` directory, and a commented 
section to illustrate how to add `CMake` as an external package in the
``packages.yaml`` in the same directory. You may also install CMake 
with `Homebrew <https://brew.sh>`_, for example, and follow the process 
outlined above after it is installed.

.. _docker_local-label:

==================================
Reproducing Docker Builds Locally
==================================

RAJA uses Docker container images that it shares with other LLNL GitHub projects
for Azure CI testing (see :ref:`azure_ci-label` for more information). 
We use Azure Pipelines for Linux, Windows, and MacOS builds.

You can reproduce these builds locally for testing with the following steps if
you have Docker installed.

  #. Run the command to build a local Docker image:

     .. code-block:: bash

       $ DOCKER_BUILDKIT=1 docker build --target ${TARGET} --no-cache

     Here, ``${TARGET}`` is replaced with one of the names following ``AS`` in 
     the `RAJA Dockerfile <https://github.com/LLNL/RAJA/blob/develop/Dockerfile>`_. 


  #. To get dropped into a terminal in the Docker image, run the following:

     .. code-block:: bash
     
       $ docker run -it axom/compilers:${COMPILER} /bin/bash

     Here, ``${COMPILER}`` is replaced with the compiler you want (see the 
     aforementioned Dockerfile).
 
Then, you can build, run tests, edit files, etc. in the Docker image. Note that
the docker command has a ``-v`` argument that you can use to mount a local 
directory in the image. For example

  .. code-block:: bash 

    & docker -v pwd:/opt/RAJA 

will mount your current local directory as ``/opt/RAJA`` in the image.
