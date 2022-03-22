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
*Pull Request (PR)*. 

.. important:: * All CI test checks must pass before a pull request can be
                 merged.
               * The status (pass/fail and run) for all checks can be viewed by
                 clicking the appropriate link in the **checks** section of a
                 GitHub pull request.

The CI tools used by the RAJA project are:

  * **Azure Pipelines** is used to run builds and tests for Linux, Windows, 
    and MacOS environments using very recent versions of various compilers. 
    While we do GPU builds for CUDA, HIP, and SYCL on Azure, RAJA tests are 
    only run for CPU-only pipelines.

  * **Gitlab** instance in the Livermore Computing (LC) Collaboration Zone (CZ)
    is used to run builds and tests with LC resource and compiler environments
    most important to many RAJA user applications. Execution of RAJA CI 
    pipelines on the LC Gitlab instance has restrictions described below.

These tools integrate fairly seamlessly with GitHub. They automatically 
(re)run RAJA builds and tests as changes are pushed to each PR branch and
report status to the GitHub PR.

The following sections describe basic elements of the operation of the CI tools.

Gitlab CI
=========

Gitlab CI testing is under continual development to expand compiler and 
version coverage, as add cross-project testing, such as building and 
running the `RAJA Performance Suite <https://github.com/LLNL/RAJAPerf>`_ 
and gathering performance data when changes are pushed to the RAJA develop 
branch. 

Constraints
-----------

Running RAJA Gitlab CI on Livermore Computing (LC) resources is 
constrained by LC security policies. The policies require that all members of 
a GitHub project be members of the LLNL GitHub organization and have 
two-factor authentication enabled on their GitHub accounts to automatically
trigger Gitlab CI functionality from GitHub. Thus, auto-mirroring of a GitHub 
repo on LC Gitlab is only done when changes are pushed to PRs for branches
in the RAJA repo, not for PRs for a branch on a fork of the repo. Alternatives
we use to account for this are described in :ref:`contributing-label`.

Gitlab CI (LC CZ) Testing Workflow
--------------------------------------

The next figure provides a high-level overview of the main steps in the 
RAJA Gitlab CI testing workflow. The main steps, which we will discuss in more
detail later, are:

  #. The process starts with the RAJA GitHub repo being *mirrored* to the 
     RAJA project in the LC CZ Gitlab instance. When a PR is made on the RAJA 
     GitHub project or whenever changes are pushed to the source branch of a 
     PR, the mirroring is triggered.
  #. Gitlab CI test pipelines are launched. Execution and pass/fail status
     may be viewed and monitored in the Gitlab CI GUI.
  #. For each resource and compiler combination, the 
     `Spack <https://github.com/spack/spack>`_ tool generates a build 
     configuration in the form of a CMake cache or *host-config* file.
  #. A host-config file is passes to CMake, which configures the RAJA build.
     Then, RAJA and its tests are compiled.
  #. Next, the RAJA tests are run.
  #. When a test pipeline completes, final results are reported to Gitlab.

In the next section, we will describe the roles that specific files in the 
RAJA repo play in defining these steps.

.. figure:: ./figures/RAJA-Gitlab-Workflow2.png

   The main steps in the RAJA Gitlab CI testing workflow. This process is
   triggered when a developer makes a PR on the GitHub project and is 
   repeated whenever changes are pushed to the source branch of the PR.

Gitlab CI (LC CZ) Testing Files
--------------------------------------

The following figure shows directories and files in the RAJA repo that 
support LC CZ Gitlab CI testing. Files with names in blue are specific to RAJA 
and owned by the RAJA team. Directories and files with names in red are
in Git submodules, shared and maintained with other projects.
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
The file ``RAJA/.gitlab-ci.yml`` contains high-level testing information, 
such as stages (resource allocation, build-and-test, and resource 
deallocation) and locations of files that define which jobs will run
in each allocation. For example, these items appear in the file as::

  stages:
    - r_allocate_resources
    - r_build_and_test
    - r_release_resources
    - l_build_and_test
    - c_build_and_test
    - multi_project

and:: 

  include:
    - local: .gitlab/ruby-templates.yml
    - local: .gitlab/ruby-jobs.yml
    - local: .gitlab/lassen-templates.yml
    - local: .gitlab/lassen-jobs.yml
    - local: .gitlab/corona-templates.yml
    - local: .gitlab/corona-jobs.yml

In the ``stages`` section above, prefixes like 'r_' and 'l_' refer to resource
names, the machines 'ruby' and 'lassen' in this case. Which jobs will run
in the pipeline(s) for each resource are defined in the files listed in the
``include`` section.

The ``RAJA/.gitlab`` directory contains a *resource* and *jobs* file for each 
LC resource which test pipelines will be run. The ``<resource>-templates.yml`` 
file in that directory contains shared configuration information,
such as script execution for stages on the resource identified in the 
``RAJA/.gitlab-ci.yml``, for pipelines that will be run on the resource. For
example, the ``RAJA/.gitlab/ruby-templates.yml`` file contains the section::

  allocate_resources (on ruby):
    variables:
      GIT_STRATEGY: none
    extends: .on_ruby
    stage: r_allocate_resources
    script:
      - salloc -N 1 -p pdebug -t 45 --no-shell --job-name=${ALLOC_NAME}

which defines the resource allocation stage associated with the 
``r_allocate_resources`` identifier in the ``RAJA/.gitlab-ci.yml`` file. Other
stages are defined similarly in all ``RAJA/.gitlab/<resource>-templates.yml``
files.

Running a CI build/test pipeline  (steps 3, 4, 5, 6)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The ``RAJA/scripts/gitlab/build_and_test.sh`` file defines the steps executed
for each build and test run as well as information that will appear in the 
log output for each step. First, the script runs the 
``RAJA/scripts/uberenv/uberenv.py`` script (located in the 
`uberenv <https://github.com/LLNL/uberenv>`_ submodule), which invokes Spack 
to generate a CMake *host-config* file that contains a RAJA configuration 
specification **(step 3)**. To generate a *host-config* file, Spack uses 
the RAJA Spack package file ``RAJA/scripts/spack_packages/raja/package.py``,
plus *Spack spec* information described next.

The ``RAJA/.gitlab/<resource>-jobs.yml`` file defines the build specifications 
that will be used for the test jobs on the corresponding resource. For example,
in the ``lassen-jobs.yml`` file, you will see entries such as::

  gcc_8_3_1_cuda:
    variables:
      SPEC: "+cuda %gcc@8.3.1 cuda_arch=70 ^cuda@10.1.168"
    extends: .build_and_test_on_lassen

This defines the *Spack spec* for the test job in which CUDA device code will 
be built with the nvcc 10.1.168 compiler and non-device code will be compiled 
with the GNU 8.3.1 compiler. In the Gitlab CI GUI, this pipeline will be 
labeled ``gcc_8_3_1_cuda``. Details for compilers, such as file system paths,
target architecture, etc.  are located in the 
``RAJA/scripts/radiuss-spack-configs/<sys-type>/compilers.yaml`` file for the 
system type associated with the resource. Analogous information for packages 
like CUDA and ROCm (HIP) are located in the corresponding 
``RAJA/scripts/radiuss-spack-configs/<sys-type>/packages.yaml`` file.

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
