.. ##
.. ## Copyright (c) 2016-23, Lawrence Livermore National Security, LLC
.. ## and RAJA project contributors. See the RAJA/LICENSE file
.. ## for details.
.. ##
.. ## SPDX-License-Identifier: (BSD-3-Clause)
.. ##

.. _ci-label:

************************************
Continuous Integration (CI) Testing
************************************

.. important:: * All CI checks must pass before a pull request can be merged.
               * The status (running and pass/fail) for all checks can be 
                 viewed by clicking the appropriate link in the **checks** 
                 section of a GitHub pull request.

The RAJA project uses to CI tools to run its tests:

  * **Azure Pipelines** runs builds and tests for Linux, Windows, and MacOS 
    environments using recent versions of various compilers. While we do some
    GPU builds on Azure, RAJA tests are only run for CPU-only builds. Docker 
    container images we use for the Linux testing are maintained in the 
    `RSE Ops Project <https://github.com/rse-ops/docker-images>`_. Please see 
    the `RAJA Azure DevOps <https://dev.azure.com/llnl/RAJA>`_ project to learn 
    more about our testing there.

  * **GitLab** instance in the Collaboration Zone (CZ) of the Livermore 
    Computing (LC) Center run builds and tests on LC platforms using
    software stacks (compilers, etc.) important to many RAJA user applications.
    Execution of LC GitLab CI on LC resources has restrictions, which are 
    described below. If you have access to LC platforms, you can access 
    information about
    `LC GitLab CI <https://lc.llnl.gov/confluence/display/GITLAB/GitLab+CI>`_

These tools integrate with the RAJA GitHub project and automatically run RAJA 
builds and tests when a PR is created and when changes are pushed to a PR 
branch or one of our protected branches `main` and `develop`.

The following sections describe basic elements of the operation of the CI tools.

.. _gitlab_ci-label:

=========
GitLab CI
=========

The GitLab CI instance used by the RAJA project lives in the Livermore 
Computing (LC) Collaboration Zone (CZ). It runs builds and tests using 
machine and compiler environments important to RAJA user applications at LLNL.

Constraints
-----------

How projects can run GitLab CI on LC platforms is constrained by LC 
security policies. The policies require that all members of a GitHub project 
be members of the LLNL GitHub organization and have two-factor authentication 
enabled on their GitHub accounts. When these requirements are satisfied, 
GitLab on the LC CZ can mirror a GitHub project and trigger GitLab CI when
changes are made to the GitHub repo. If the requirements are not met, LC 
GitLab CI checks will not be run for a project. This implies, for example,
that GitLab CI will not run for a PR made on an LLNL organization project 
from a fork of the project repo by someone not in the LLNL organization. 

For a compliant LLNL GitHub project like RAJA, auto-mirroring of the 
GitHub repo on LC GitLab is done every 30 minutes or so, triggering builds and
tests on new changes pushed to the RAJA GitHub project. If you have access to 
LC platforms, you can learn more about `LC GitLab mirroring <https://lc.llnl.gov/confluence/pages/viewpage.action?pageId=662832265>`_.

.. note:: **GitLab CI will not run for a PR branch on a fork of the RAJA repo.**
           We manually manage contributions made on a fork of the RAJA repo 
           using the procedure described in :ref:`contributing-label`.

.. _gitlab_ci_workflow-label:

GitLab CI (LC) Testing Workflow
--------------------------------------

The figure below shows the high-level steps in the RAJA GitLab CI testing 
process. The main steps, which we will discuss in more detail later, are:

  #. A *mirror* of the RAJA GitHub repo is updated in the RAJA LC CZ GitLab 
     project automatically every 30 minutes approximately.

     .. note:: There may be a delay in the mirroring, since it is not 
               synhronous with changes to the RAJA GitHub project.

  #. GitLab launches CI test pipelines for any new changes made to the 
     ``develop`` or ``main`` branches or any non-fork PR branch. While 
     running, the execution and pass/fail status may be viewed and monitored 
     in the GitLab CI GUI or in the RAJA GitHub project checks section of a PR.

  #. For each platform and compiler combination,
     `Spack <https://github.com/spack/spack>`_ builds RAJA dependencies and
     generates a configuration in the form of a CMake cache file, or 
     *host-config* file.

  #. A host-config file is passed to CMake, which configures a RAJA build 
     space.  Then, RAJA and its tests are compiled.

  #. Next, the RAJA tests are run.

  #. When test pipelines complete, results are reported to GitLab.

  #. Lastly, GitLab reports to GitHub to show the status of checks there.

.. figure:: ./figures/RAJA-Gitlab-Workflow2.png

   The main steps in the RAJA GitLab CI testing workflow are shown in the 
   figure. This process is triggered when a developer makes a PR on the 
   GitHub project or whenever changes are pushed to the source branch of a PR.

Next, we describe the roles that external projects and files in the RAJA repo 
play in the RAJA GitLab CI workflow.

.. _gitlab_ci_depend-label:

GitLab CI Testing Dependencies (specific to LC CZ)
---------------------------------------------------

RAJA GitLab CI testing depends on several other projects that we develop
collaboratively with other projects. These include

  * `RADIUSS Shared CI <https://github.com/LLNL/radiuss-shared-ci>`_,
    a centralized framework for software testing with GitLab CI on LC
    machines. The project is developed on GitHub and is mirrored to the LC 
    CZ GitLab instance.
  * `Spack <https://github.com/spack/spack>`_, a multi-platform package 
    manager that builds and installs HPC software stacks.
  * `Uberenv <https://github.com/LLNL/uberenv>`_, a Python script
    that helps to automate the use of Spack and other tools for building 
    third-party dependencies. Uberenv is a submodule in RAJA that lives in
    ``RAJA/scripts/uberenv/``.
  * `RADIUSS Spack Configs <https://github.com/LLNL/radiuss-spack-configs>`_,
    a collection of Spack compiler and package configurations used by Spack 
    to generate host-config files for CMake. The build configurations are 
    specific to LLNL LC platforms. Spack packages for multiple projects are
    maintained in this project. RADIUSS Spack Configs is a submodule in RAJA 
    that lives in ``RAJA/scripts/radiuss-spack-configs/``.

The relationships among these dependencies in a project that uses them is 
illustrated in the `RADIUSS Shared CI User Guide <https://radiuss-shared-ci.readthedocs.io/en/latest/sphinx/user_guide/index.html>`_. The guide also describes 
how the framework works and how to set up a project to use it.

.. important:: The RAJA Spack package is maintained in the `RADIUSS Spack Configs <https://github.com/LLNL/radiuss-spack-configs>`_ project. When it is updated, it is pushed to the Spack repo on GitHub.

In the rest of the this section, we describe files in the RAJA repo that are
used to configure and customize the shared CI framework specifically for the 
RAJA project.

.. _gitlab_ci_files-label:

GitLab CI Testing Files (specific to LC CZ)
--------------------------------------------

The following figure shows directories and files in the RAJA project that 
support LC GitLab CI testing. 

.. figure:: ./figures/RAJA-Gitlab-Files.png

   The figure shows directories and files in the RAJA repo that support GitLab 
   CI testing. Files in blue are specific to RAJA and are maintained in the 
   RAJA repo. Red directories and files are in Git submodules that are 
   shared and maintained with other projects.

Briefly, these files play the following roles in our GitLab CI testing:

  * The ``RAJA/.gitlab-ci.yml`` file is the root file for GitLab CI 
    configuration. We place jobs is small pipelines described by separate 
    files that are included by this one. Global variables can also be defined 
    here.
  * The ``.uberenv_config.json`` file defines the Spack version we use, where 
    Spack packages live, etc.
  * Files in the ``RAJA/.gitlab`` directory define test pipelines that RAJA
    subscribes to an which are defined in the 
    `RADIUSS Shared CI <https://github.com/LLNL/radiuss-shared-ci>`_ project,
    as well as RAJA-specific jobs, and any job customization that we use,
    such as job time limits, etc. These files are customizations of templates 
    provided by `RADIUSS Shared CI <https://github.com/LLNL/radiuss-shared-ci>`_.
  * The ``RAJA/scripts/gitlab/build_and_test.sh`` file defines the RAJA build 
    and test process and commands that are run during it.

In the following sections, we discuss how these files are used in the 
steps of the RAJA GitLab CI testing process summarized above.

.. _gitlab_ci_pipelines-label:

Launching CI pipelines (step 2) 
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In **step 2** of the diagram above, GitLab launches RAJA test pipelines.
The `RAJA/.gitlab-ci.yml <https://github.com/LLNL/RAJA/tree/develop/.gitlab-ci.yml>`_ file contains high-level testing information that applies to all RAJA
GitLab CI testing pipelines. This includes

  * **GitLab pipeline variables**, such as project name, service user account
    name, etc.

  * **High-level pipeline stages** for build and test, multi-project testing,
    etc.

  * **Build and test sub-pipelines**. Note that this is where the connection 
    is made to the RADIUSS Shared CI project (and version on the LC CZ GitLab 
    instance) and to files in the ``RAJA/.gitlab`` directory that define the 
    Spack specs for build configurations that are run on each machine on
    which RAJA tests are run.

  * **Cross-project test pipelines**, which are triggered when testing 
    certain RAJA branches, mainly the develop branch.

  * **CI subscribed pipelines**, which are defined in the
    RADIUSS Shared CI project. 

.. important:: Variables that define how resources are allocated and job time 
               limits for LC machines that are used to run RAJA CI are defined
               in the ``RAJA/.gilab/custom-jobs-and-variables.yml`` file.

Each job that is run is defined by a Spack spec in one of two places, depending
on whether it is *shared* with other projects or it is specific to RAJA. The 
shared jobs are defined in files named ``<MACHINE>-build-and-test.yml`` in 
the top-level directory of the 
`RADIUSS Shared CI Project <https://github.com/LLNL/radiuss-shared-ci>`_.
RAJA-specific jobs are defined in 
``RAJA/.gitlab/<MACHINE>-build-and-test-extra.yml`` files. 

**Each shared job will be run as-is unless it is overridden** in the RAJA 
'extra' file for the corresponding machine. For example, a shared job for the 
LC ruby machine may appear in the RADIUSS Shared CI file 
``ruby-build-and-test.yml`` as::

  gcc_8_1_0:
    variables:
      SPEC: "${PROJECT_RUBY_VARIANTS} %gcc@8.1.0 ${PROJECT_RUBY_DEPS}"
    extends: .build_and_test_on_ruby

and then may be overridden in the ``RAJA/.gitlab/ruby-build-and-test-extra.yml``
file as::

  gcc_8_1_0:
    variables:
      SPEC: " ${PROJECT_RUBY_VARIANTS} %gcc@8.1.0 ${PROJECT_RUBY_DEPS}"
      RUBY_BUILD_AND_TEST_JOB_ALLOC: "--time=60 --nodes=1"
    extends: .build_and_test_on_ruby

In this example, the Spack build spec is the same, but the job is configured
with a timeout limit and number of nodes appropriate for RAJA testing.

.. important:: A shared job override **must use the same job label as the 
               shared job** defined in the RADIUSS Shared CI project.

RAJA-specific jobs whose configurations are not shared with other projects
are also defined in the 
``RAJA/.gitlab/<MACHINE>-build-and-test-extra.yml`` files. For example::

  clang_10_0_1_gcc_8_3_1_desul_atomics:
    variables:
      SPEC: " ~shared +openmp +tests +desul %clang@10.0.1 cxxflags=--gcc-toolchain=/usr/tce/packages/gcc/gcc-8.3.1 cflags=--gcc-toolchain=/usr/tce/packages/gcc/gcc-8.3.1"
    extends: .build_and_test_on_ruby

defines a RAJA job with desul atomics enabled to be run on the ruby machine.

.. important:: Each base compiler configuration that is used in GitLab CI 
               testing must have a Spack spec defined for it in the appropriate
               file for the machine that it will be tested on in the 
               `RADIUSS Spack Configs <https://github.com/LLNL/radiuss-spack-configs>`_ project.

.. _gitlab_ci_running-label:

Running a CI build and test pipeline  (steps 3, 4, 5, 6)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The `RAJA/scripts/gitlab/build_and_test.sh <https://github.com/LLNL/RAJA/tree/develop/scripts/gitlab/build_and_test.sh>`_ file defines the steps executed
for each build and test pipeline as well as information that will appear in the
log output for each step. 

After some basic set up, the script invokes the 
``RAJA/scripts/uberenv/uberenv.py`` Python script that drives Spack to generate
host-config files::

  ...

  python3 scripts/uberenv/uberenv.py --spec="${spec}" ${prefix_opt}

  ...

Project specific settings related to which Spack version to use, where 
Spack packages live, etc. are located in the 
`RAJA/.uberenv_config.json <https://github.com/LLNL/RAJA/tree/develop/.uberenv_config.json>`_ file.

The Uberenv Python script invokes Spack to generate a CMake *host-config* 
file containing a RAJA build specification **(step 3)**. To generate
a *host-config* file, Spack uses the packages and specs maintained in the 
`RADIUSS Spack Configs project 
<https://github.com/LLNL/radiuss-spack-configs>`_, plus RAJA-specific specs
defined in files in the `RAJA/.gitlab <https://github.com/LLNL/RAJA/tree/develop/.gitlab>`_ directory, as described earlier.

.. note:: Please see :ref:`spack_host_config-label` for more information about
          how to manually generate host-config files and use them for local
          debugging.

After the host-config file is generated, the 
``RAJA/scripts/gitlab/build_and_test.sh`` script creates a build space 
directory and runs CMake in it, passing the host-config (cache) file. Then, 
it builds the RAJA code and tests **(step 4)**::

  ...

  build_dir="${build_root}/build_${hostconfig//.cmake/}"
  install_dir="${build_root}/install_${hostconfig//.cmake/}"

  ...

  date
  echo "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
  echo "~~~~~ Host-config: ${hostconfig_path}"
  echo "~~~~~ Build Dir:   ${build_dir}"
  echo "~~~~~ Project Dir: ${project_dir}"
  echo "~~~~~ Install Dir: ${install_dir}"
  echo "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
  echo ""
  echo "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
  echo "~~~~~ Building RAJA"
  echo "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"

  ..

  rm -rf ${build_dir} 2>/dev/null
  mkdir -p ${build_dir} && cd ${build_dir}

  ...

  $cmake_exe \
      -C ${hostconfig_path} \
      -DCMAKE_INSTALL_PREFIX=${install_dir} \
      ${project_dir}

  ...

  echo "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
  echo "~~~~~ RAJA Built"
  echo "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
  date

Next, it runs the tests **(step 5)**::

  echo "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
  echo "~~~~~ Testing RAJA"
  echo "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"

  ...

  cd ${build_dir}

  ...

  ctest --output-on-failure -T test 2>&1 | tee tests_output.txt

  ...

  echo "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
  echo "~~~~~ RAJA Tests Complete"
  echo "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
  date

Lastly, test results are collected in a JUnit XML file that
GitLab uses for reporting the results in its GUI **(step 6)**. This is
done by the 
`RADIUSS Shared CI Framework <https://github.com/LLNL/radiuss-shared-ci>`_

The commands shown here intermingle with other commands that emit messages,
timing information for various operations, etc. which appear in a log
file that can be viewed in the GitLab GUI.

.. _azure_ci-label:

==================
Azure Pipelines CI
==================

The Azure Pipelines tool builds and tests for Linux, Windows, and MacOS 
environments.  While we do builds for CUDA, HIP, and SYCL RAJA back-ends 
in the Azure Linux environment, RAJA tests are only run for CPU-only pipelines.

Azure Pipelines Testing Workflow
--------------------------------

The Azure Pipelines testing workflow for RAJA is much simpler than the GitLab
testing process described above.

The test jobs we run for each OS environment are specified in the 
`RAJA/azure-pipelines.yml <https://github.com/LLNL/RAJA/blob/develop/azure-pipelines.yml>`_ file. This file defines the job steps, commands,
compilers, etc. for each OS environment in the associated ``- job:`` section.
A summary of the configurations we build are:

  * **Windows.** The ``- job: Windows`` Windows section contains information
    for the Windows test builds. For example, we build and test RAJA as
    a static and shared library. This is indicated in the Windows ``strategy``
    section::
   
      strategy:
        matrix:
          shared:
            ...
          static:
            ...

    We use the Windows/compiler image provided by the Azure application 
    indicated the ``pool`` section; for example::

      pool:
        vmImage: 'windows-2019'

    **MacOS.** The ``- job: Mac`` section contains information for Mac test 
    builds. For example, we build RAJA using the the MacOS/compiler 
    image provided by the Azure application indicated in the ``pool`` section; 
    for example::

      pool:
        vmImage: 'macOS-latest' 

    **Linux.** The ``- job: Docker`` section contains information for Linux
    test builds. We build and test RAJA using Docker container images generated 
    with recent versions of various compilers. The RAJA project shares these 
    images with other open-source LLNL RADIUSS projects and they are maintained
    in the `RES-Ops Docker <https://github.com/rse-ops/docker-images>`_ 
    project on GitHub. The builds we do at any point in time are located in 
    the ``strategy`` block::

      strategy:
        matrix: 
          gccX:
            docker_target: ...
          ...
          clangY:
            docker_target: ...
          ...
          nvccZ:
            docker_target: ...

          ...

    The Linux OS the docker images are run on is indicated in the ``pool`` section; 
    for example::

      pool:
        vmImage: 'ubuntu-latest'

Docker Builds
-------------

For each Linux/Docker pipeline, the base container images, CMake, build, and
test commands are located in `RAJA/Dockerfile <https://github.com/LLNL/RAJA/blob/develop/Dockerfile>`_.

The base container images are built and maintained through the 
`RSE-Ops Docker <https://rse-ops.github.io/>`_ project. A table of the most 
up-to-date containers can be found 
`here <https://rse-ops.github.io/docker-images/>`_. These images are rebuilt 
regularly ensuring that we have the most up to date builds of each 
container and compiler.

.. note:: Please see :ref:`docker_local-label` for more information about
          reproducing Docker builds locally for debugging purposes.

