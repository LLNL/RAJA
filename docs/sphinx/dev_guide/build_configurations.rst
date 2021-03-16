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

RAJA must be built by a wide variety of compilers and for all of its 
supported back-ends. We maintain two mechanisms to facilitate the use of 
important build configurations for testing and reproducibility:

  * **Build scripts.** We maintain a collection of simple build scripts in
    the RAJA repository that are used to generate build spaces on Livermore 
    Computing platforms primarily.
    
  * **Generated host-config files.** The RAJA repository includes a 
    mechanism to generate *host-config* files (i.e., CMake cache files)
    using `Spack <https://github.com/spack/spack>`_.

Each of these mechanisms specifies compilers and compiler versions, 
compiler options, build target (Release, Debug, etc.), RAJA 
features to enable (OpenMP, CUDA, etc.) and paths to required
tool chains, such as CUDA, ROCm, etc.

They will be describe briefly in the following sections.


===================
RAJA Build Scripts
===================

The build scripts in the RAJA ``scripts`` directory are used mostly by RAJA 
developers to quickly set up a build environment to compile and run tests
during code development. 

Each script can be executed from the top-level RAJA directory. It will 
create a uniquely-named build directory in RAJA and run CMake in it
with arguments contained in the script. You then go into that directory 
and run make to build RAJA. For example,

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
`radiuss-spack-configs <https://github.com/LLNL/radiuss-spack-configs>`_ and 
`uberenv <https://github.com/LLNL/uberenv>`_ which are projects in the 
GitHub LLNL organization. They are shared by various projects. The main 
uberenv script can be used to drive Spack to generate a *host-config* file 
(i.e., CMake cache file) that can be passed to CMake with the '-C' option 
to create a build configuration. The *Spack specs* defining compiler 
configurations are maintained in files in the radiuss-spack-configs 
repository.

RAJA shares its Uberenv workflow with other projects. The documentation 
for this is available in `RADIUSS uberenv guide <https://radiuss-ci.readthedocs.io/en/latest/uberenv.html#uberenv-guide>`_.


Generating a RAJA host-config file
------------------------------------

This section describes key information for generating a host-config file 
for RAJA.

Machine specific configurations
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Compiler configurations for LC platforms are contained in 
in sub-directories in the RAJA ``scripts/uberenv/spack_configs`` 
directory:

.. code-block:: bash

  $ ls -c1 scripts/uberenv/spack_configs
  blueos_3_ppc64le_ib
  darwin
  toss_3_x86_64_ib
  blueos_3_ppc64le_ib_p9
  config.yaml

View the contents of the ``compilers.yaml`` file in each of these 
sub-directories to see the currently supported configurations.

Generating a host-config file
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The main uberenv python script can be invoked from the top-level RAJA
directory to generate a host-config file for a desired configuration.
For example,

.. code-block:: bash

  $ python ./scripts/uberenv/uberenv.py --spec="%gcc@8.1.0"
  $ python ./scripts/uberenv/uberenv.py --spec="%gcc@8.1.0~shared+openmp tests=benchmarks"

Such commands will generate the corresponding host-config file (i.e.,
CMake cache file) in the top-level RAJA directory. For example,

.. code-block:: bash

  hc-quartz-toss_3_x86_64_ib-gcc@8.1.0-fjcjwd6ec3uen5rh6msdqujydsj74ubf.cmake

Building RAJA with a generated host-config file
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

  $ mkdir <build dirname> && cd <build dirname>
  $ cmake -C <path_to>/<host-config>.cmake ..
  $ cmake --build -j .
  $ ctest --output-on-failure -T test

It is also possible to use the configuration with a RAJA CI script outside 
of the normal CI process:

.. code-block:: bash

  $ HOST_CONFIG=<path_to>/<host-config>.cmake ./scripts/gitlab/build_and_test.sh

MacOS case
^^^^^^^^^^

In RAJA, the Spack configuration for MacOS contains the default compilers 
depending on the OS version (`compilers.yaml`), and a commented section to 
illustrate how to add `CMake` as an external package. You may install CMake 
with homebrew, for example, and follow the process outlined above.

Vetted specs
^^^^^^^^^^^^

The *vetted* compiler specs are those which we use during the RAJA Gitlab CI
testing process. These can be viewed by looking at files in the RAJA
``.gitlab`` directory. For example, 

.. code-block:: bash

  $ ls -c1 .gitlab/*jobs.yml
  .gitlab/lassen-jobs.yml
  .gitlab/quartz-jobs.yml

lists the yaml files containing the Gitlab CI jobs for lassen and quartz.

Then, executing a command such as:

.. code-block:: bash

  $ git grep -h "SPEC" .gitlab/quartz-jobs.yml | grep "gcc"
      SPEC: "%gcc@4.9.3"
      SPEC: "%gcc@6.1.0"
      SPEC: "%gcc@7.3.0"
      SPEC: "%gcc@8.1.0"

will list the specs vetted on ``quartz``/``toss_3_x86_64_ib``.
