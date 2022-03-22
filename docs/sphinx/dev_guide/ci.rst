.. ##
.. ## Copyright (c) 2016-22, Lawrence Livermore National Security, LLC
.. ## and RAJA project contributors. See the RAJA/LICENSE file
.. ## for details.
.. ##
.. ## SPDX-License-Identifier: (BSD-3-Clause)
.. ##

.. _ci-label:

************************************
Continuous Integration (CI) Testing
************************************

The RAJA project employs multiple tools to run CI tests for each GitHub
*Pull Request (PR)*. All test checks must pass before the pull request can 
be merged. These tools include:

  * **Azure Pipelines** is used to run builds and tests for Linux, Windows, 
    and MacOS environments using very recent versions of various compilers. 
    While we do GPU builds for CUDA, HIP, and SYCL on Azure, RAJA tests are 
    only run for CPU-only pipelines.

  * **Gitlab** instance in the Livermore Computing (LC) Collaboration Zone (CZ)
    is used to run builds and tests with LC platform and compiler environments
    most important to many RAJA user applications. Execution of RAJA CI 
    pipelines on the LC Gitlab instance has restrictions described below.

For the most part, these tools integrate seamlessly with GitHub. They 
automatically (re)run RAJA builds and tests as changes are pushed to each 
PR branch.

.. important:: * All test checks must pass before the pull request can be 
                 merged.
               * The status (pass/fail and run) for all checks can be viewed by 
                 clicking the appropriate link in the **checks** section of a 
                 GitHub pull request.

Gitlab CI
=========

Gitlab CI testing is under development to expand compiler and version 
coverage, as well as cross-project testing, such as building and running the
`RAJA Performance Suite <https://github.com/LLNL/RAJAPerf>`_ when changes 
are pushed to the RAJA develop branch. 

Constraints
-----------

Running the RAJA Gitlab CI on Livermore Computing (LC) resources is 
constrained by LC security policies. Auto-mirroring of a GitHub project repo
on LC Gitlab requires that all members of the GitHub project are members of 
the LLNL GitHub organization and have two-factor authentication enabled on 
their GitHub accounts. Gitlab CI will run automatically for such projects on 
pull requests that are made by vetted project members. Otherwise, Gitlab CI 
must be run manually by a vetted GitHub user satisfying the constraints just
described.

Gitlab CI (LC CZ) Testing Workflow
--------------------------------------

The next figure provides a high-level overview of the main steps in the 
RAJA Gitlab CI testing workflow. We will describe the individual steps in
more detail later. The main steps are:   

  #. The process starts with the RAJA GitHub repo being *mirrored* to the 
     RAJA project in the LC CZ Gitlab instance. When a PR is made on the RAJA 
     GitHub project or whenever changes are pushed to the source branch of a 
     PR, the mirroring occurs.
  #. Gitlab CI test pipelines are launched. Execution and pass/fail status
     may be viewed and monitored in the Gitlab CI GUI.
  #. For each platform and compiler combination, the 
     `Spack <https://github.com/spack/spack>`_ tool generates a build 
     configuration in the form of a CMake cache or *host-config* file.
  #. Based on the information in each host-config file, the RAJA code and tests
     are built.
  #. Next, the RAJA tests are run.
  #. When a test pipeline completes, results are reported to Gitlab.

In the next section, we will describe the roles that specific files in the 
RAJA repo play in defining these steps.

.. figure:: ./figures/RAJA-Gitlab-Workflow2.png

   The main steps in the RAJA Gitlab CI testing workflow. This process is
   triggered when a developer makes a PR on the GitHub project and is 
   repeated whenever changes are pushed to the source branch of the PR.

Gitlab CI (LC CZ) Testing Files
--------------------------------------

The following figure shows directories and files in the RAJA repo that 
support LC CZ Gitlab CI testing. File names in blue denote that they are 
specific to RAJA and owned by the RAJA team. Red directories and files denote 
that they are part of Git submodules, shared and maintained with other projects.
In the following sections, we discuss how these files are used to define the 
steps in the RAJA Gitlab CI testing process described earlier.

.. figure:: ./figures/RAJA-Gitlab-Files.png

   Directories and files in the RAJA repo that support Gitlab CI testing.
   Files in blue are specific to RAJA and owned by the RAJA team. Red 
   directories and files are part of Git submodules shared with other 
   projects.

Launching CI pipelines (step 2) 
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In **step 2** of the diagram above, Gitlab launches RAJA test pipelines.
The file ``RAJA/.gitlab-ci.yml`` defines high-level testing stages
(resource allocation, build-and-test, and resource deallocation), names of 
scripts that will run, and locations of files that define which jobs will run
in each allocation.

Running a CI build/test pipeline  (steps 3, 4, 5, 6)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The ``RAJA/.gitlab`` directory contains a *platform* and *jobs* file for each 
LC platform where test pipelines will be run. The ``<platform>-templates.yml`` 
file contains shared configuration information for pipelines that will be run 
on the platform. The ``<platform>-jobs.yml`` file defines the build 
specifications that will be use to generate test executables. Specifically, 
they identify Spack *specs*.

The ``scripts/gitlab/build_and_test.sh`` file defines the steps executed
for each build and test run as well as information that will appear in the 
log output for each. The process of executing a test pipeline extends the
information in the script appropriately for each system. First, the script 
runs the ``RAJA/scripts/uberenv/uberenv.py`` script (located in the 
`uberenv <https://github.com/LLNL/uberenv>`_ submodule), which invokes Spack 
to generate a CMake *host-config* file that contains a RAJA configuration 
specification **(step 3)**. 

To generate a *host-config* file, Spack uses information in the RAJA Spack
package file ``RAJA/scripts/spack_packages/raja/package.py`` and items
defined in a given Spack configuration (i.e., *Spack spec*). Available Spack 
configurations are defined in *packages* and *compilers* files in the 
`radiuss-spack-configs <https://github.com/LLNL/radiuss-spack-configs>`_
submodule; located in ``RAJA/scripts/radiuss-spack-configs`` directory.
For each supported system/OS type, you will see files labeled as:
``radiuss-spack-configs/<os-type>/compilers.yaml`` and 
``radiuss-spack-configs/<os-type>/packages.yaml``

After the host-config file is generated, the 
``scripts/gitlab/build_and_test.sh`` script creates a build space directory 
and runs CMake in it, passing the host-config (cache) file. Next, it builds 
the RAJA tests **(step 4)** and runs the tests **(step 5)**. 

Lastly, the script packages the test results in a JUnit XML file, which Gitlab uses for reporting the results in its GUI **(step 6))**.

.. _vettedspecs-label:

Vetted Specs
------------

The Spack specifications we use in the RAJA Gitlab CI workflow should be 
considered by users to be *vetted* in the sense that they are tested
regularly. Specifically, every change pushed to the RAJA main and develop
branches has been run though the build and test process described above.
