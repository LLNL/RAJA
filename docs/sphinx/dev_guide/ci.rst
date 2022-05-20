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

.. important:: * All CI test checks must pass before a pull request can be
                 merged.
               * The status (pass/fail and run) for all checks can be viewed by
                 clicking the appropriate link in the **checks** section of a
                 GitHub pull request.

The CI tools used by the RAJA project, and which integrate with GitHub are:

  * **Azure Pipelines** runs builds and tests for Linux, Windows, and MacOS 
    environments using recent versions of various compilers. While we do GPU 
    builds for CUDA, HIP, and SYCL on Azure, RAJA tests are only run for 
    CPU-only pipelines. See the 
    `RAJA Azure DevOps <https://dev.azure.com/llnl/RAJA>`_ project to learn 
    more about our testing there.

  * **Gitlab** instances in the Livermore Computing (LC) Center
    runs builds and tests in LC resource and compiler environments
    important to many RAJA user applications. Execution of RAJA CI 
    pipelines on LC Gitlab resources has restrictions described below. If 
    you have access to LC resources, you can access additional information about
    `LC GitLab CI <https://lc.llnl.gov/confluence/display/GITLAB/GitLab+CI>`_

The tools automatically run RAJA builds and tests when a PR is created and 
when changes are pushed to a PR branch.

The following sections describe basic elements of the operation of the CI tools.

.. _gitlab_ci-label:

=========
Gitlab CI
=========

The GItlab CI instance used by the RAJA project lives in the Livermore 
Computing (LC) Collaboration Zone (CZ). It runs builds and tests in LC 
resource and compiler environments important to RAJA user applications at LLNL.

Constraints
-----------

Running Gitlab CI on Livermore Computing (LC) resources is constrained by LC 
security policies. The policies require that all members of a GitHub project 
be members of the LLNL GitHub organization and have two-factor authentication 
enabled on their GitHub accounts to automatically mirror a GitHub repo and
trigger Gitlab CI functionality from GitHub. For compliant LLNL GitHub projects,
auto-mirroring of the GitHub repo on LC Gitlab is done when changes are pushed 
to PRs for branches in the RAJA repo, but not for PRs for a branch on a fork of
the repo. An alternative procedure we use to handle this is described in 
:ref:`contributing-label`. If you have access to LC resources, you can learn
more about `LC Gitlab mirroring <https://lc.llnl.gov/confluence/pages/viewpage.action?pageId=662832265>`_.

Gitlab CI (LC) Testing Workflow
--------------------------------------

The figure below shows the high-level steps in the RAJA Gitlab CI testing 
process. The main steps, which we will discuss in more detail later, are:

  #. A *mirror* of the RAJA GitHub repo in the RAJA Gitlab project is updated
     whenever the RAJA ``develop`` or ``main`` branches are changed as well
     as when any PR branch in the RAJA GitHub project is changed. 
  #. Gitlab launches CI test pipelines. While running, the execution and 
     pass/fail status may be viewed and monitored in the Gitlab CI GUI.
  #. For each resource and compiler combination,
     `Spack <https://github.com/spack/spack>`_ is used to generate a build 
     configuration in the form of a CMake cache file, or *host-config* file.
  #. A host-config file is passed to CMake, which configures a RAJA build 
     space.  Then, RAJA and its tests are compiled.
  #. Next, the RAJA tests are run.
  #. When a test pipeline completes, final results are reported in Gitlab.

In the next section, we will describe the roles that specific files in the 
RAJA repo play in defining these steps.

.. figure:: ./figures/RAJA-Gitlab-Workflow2.png

   The main steps in the RAJA Gitlab CI testing workflow are shown in the 
   figure. This process is triggered when a developer makes a PR on the 
   GitHub project or whenever changes are pushed to the source branch of a PR.

Gitlab CI (LC) Testing Files
--------------------------------------

The following figure shows directories and files in the RAJA repo that 
support LC Gitlab CI testing. Files with names in blue are specific to RAJA 
and are maintained by the RAJA team. Directories and files with names in red are
in Git submodules, shared and maintained with other projects.

.. figure:: ./figures/RAJA-Gitlab-Files.png

   The figure shows directories and files in the RAJA repo that support Gitlab 
   CI testing. Files in blue are specific to RAJA and owned by the RAJA team. 
   Red directories and files are part of Git submodules shared with other 
   projects.

In the following sections, we discuss how these files are used in the 
steps in the RAJA Gitlab CI testing process summarized above.

Launching CI pipelines (step 2) 
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In **step 2** of the diagram above, Gitlab launches RAJA test pipelines.
The `RAJA/.gitlab-ci.yml <https://github.com/LLNL/RAJA/blob/develop/.gitlab-ci.yml>`_ file contains high-level testing information, 
such as stages (resource allocation, build-and-test, and resource 
deallocation) and locations of files that define which jobs will run
in each pipeline. For example, these items appear in the file as::

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

In the ``stages`` section above, prefixes 'r_', 'l_', and 'c_' refer to 
resources in the LC on which tests are run. Specifically, the machines 'ruby',
'lassen', and 'corona', respectively. Jobs that will run in pipeline(s) on each 
resource are defined in the files listed in the ``include`` section above.
Note that the stage labels above appear on each Gitlab CI run webpage as the
title of a column containing other information about what is run in that stage,
such as build and test jobs.

The `RAJA/.gitlab <https://github.com/LLNL/RAJA/tree/develop/.gitlab>`_ 
directory contains a *templates* and *jobs* file for each LC resource on which 
test pipelines will be run. The ``<resource>-templates.yml`` files contain 
information that is common across jobs that run on the corresponding resource, 
such as commands and scripts that are run for stages identified in the 
``RAJA/.gitlab-ci.yml`` file. For example, the 
``RAJA/.gitlab/ruby-templates.yml`` file contains a section::

  allocate_resources (on ruby):
    variables:
      GIT_STRATEGY: none
    extends: .on_ruby
    stage: r_allocate_resources
    script:
      - salloc -N 1 -p pdebug -t 45 --no-shell --job-name=${ALLOC_NAME}

which contains the resource allocation command associated with the 
``r_allocate_resources`` stage identifier on 'ruby'. Analogous stages are 
defined similarly in other ``RAJA/.gitlab/<resource>-templates.yml`` files.

The ``<resource>-jobs.yml`` files are described in the following sections.

Running a CI build/test pipeline  (steps 3, 4, 5, 6)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The `RAJA/scripts/gitlab/build_and_test.sh <https://github.com/LLNL/RAJA/tree/develop/scripts/gitlab/build_and_test.sh>`_ file defines the steps executed
for each build and test run as well as information that will appear in the
log output for each step. First, the script invokes the 
``RAJA/scripts/uberenv/uberenv.py`` Python script located in the 
`uberenv <https://github.com/LLNL/uberenv>`_ submodule::

  ...

  python3 scripts/uberenv/uberenv.py --spec="${spec}" ${prefix_opt}

  ...

Project specific settings related to which Spack version to use, where 
Spack packages live, etc. are located in the 
`RAJA/.uberenv_config.json <https://github.com/LLNL/RAJA/blob/develop/.uberenv_config.json>`_ file.

The uberenv python script invokes Spack to generate a CMake *host-config* 
file containing a RAJA build specification **(step 3)**. To generate
a *host-config* file, Spack uses the 
`RAJA Spack package <https://github.com/LLNL/RAJA/blob/develop/scripts/spack_packages/raja/package.py>`_, plus *Spack spec* information. 
The ``RAJA/.gitlab/<resource>-jobs.yml`` file defines a build specification
(*Spack spec*) for each job that will be run on the corresponding resource. 
For example, in the ``lassen-jobs.yml`` file, you will see an entry such as::

  gcc_8_3_1_cuda_10_1_168:
    variables:
      SPEC: "+cuda %gcc@8.3.1 cuda_arch=70 ^cuda@10.1.168"
    extends: .build_and_test_on_lassen

This defines the *Spack spec* for the test job in which CUDA device code will 
be built with the nvcc 10.1.168 compiler and non-device code will be compiled 
with the GNU 8.3.1 compiler. In the Gitlab CI GUI, this pipeline will be 
labeled ``gcc_8_3_1_cuda_10_1_168``. Details for compilers, such as file 
system paths, target architecture, etc. are located in the 
``RAJA/scripts/radiuss-spack-configs/<sys-type>/compilers.yaml`` file for the 
system type associated with the resource. Analogous information for packages 
like CUDA and ROCm (HIP) are located in the corresponding 
``RAJA/scripts/radiuss-spack-configs/<sys-type>/packages.yaml`` file.

.. note:: Please see :ref:`spack_host_config-label` for more information about
          Spack-generated host-config files and how to use them for local
          debugging.

After the host-config file is generated, the 
``scripts/gitlab/build_and_test.sh`` script creates a build space directory 
and runs CMake in it, passing the host-config (cache) file. Then, it builds
the RAJA code and tests **(step 4)**::

  ...

  build_dir="${build_root}/build_${hostconfig//.cmake/}"

  ...

  date
  echo "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
  echo "~ Host-config: ${hostconfig_path}"
  echo "~ Build Dir:   ${build_dir}"
  echo "~ Project Dir: ${project_dir}"
  echo "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
  echo ""

  echo "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
  echo "~~~~~ Building RAJA"
  echo "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"

  rm -rf ${build_dir} 2>/dev/null
  mkdir -p ${build_dir} && cd ${build_dir}

  ...

  cmake \
    -C ${hostconfig_path} \
    ${project_dir}  
 
  cmake --build . -j ${core_counts[$truehostname]}

Next, it runs the tests **(step 5)**::

  echo "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
  echo "~~~~~ Testing RAJA"
  echo "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"

  ...

  cd ${build_dir}

  ...

  ctest --output-on-failure -T test 2>&1 | tee tests_output.txt

Lastly, the script packages the test results into a JUnit XML file that
Gitlab uses for reporting the results in its GUI **(step 6)**::

  echo "Copying Testing xml reports for export"
  tree Testing
  xsltproc -o junit.xml ${project_dir}/blt/tests/ctest-to-junit.xsl Testing/*/Test.xml
  mv junit.xml ${project_dir}/junit.xml

The commands shown here intermingle with other commands that emit messages,
timing information for various operations, etc. which appear in a log
file that can be viewed in the Gitlab GUI.

.. _azure_ci-label:

==================
Azure Pipelines CI
==================

The Azure Pipelines tool builds and tests for Linux, Windows, and MacOS 
environments.  While we do builds for CUDA, HIP, and SYCL RAJA back-ends 
in the Azure Linux environment, RAJA tests are only run for CPU-only pipelines.

Azure Pipelines Testing Workflow
--------------------------------

The Azure Pipelines testing workflow for RAJA is much simpler than the Gitlab
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
    builds. For example, we build and test RAJA using the the MacOS/compiler 
    image provided by the Azure application indicated in the ``pool`` section; 
    for example::

      pool:
        vmImage: 'macOS-latest' 

    **Linux.** The ``- job: Docker`` section contains information for Linux
    test builds. We build and test RAJA using Docker container images generated 
    with recent versions of various compilers. The RAJA project shares these 
    images with other open-source LLNL RADIUSS projects and they are maintained
    in the `RES-ops Docker <https://github.com/rse-ops/docker-images>`_ 
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

    The Linux OS image is indicated in the ``pool`` section; 
    for example::

      pool:
        vmImage: 'ubuntu-latest'

For each Linux build and test pipeline, the container images, CMake, build, and
test commands are located in `RAJA/Dockerfile <https://github.com/LLNL/RAJA/blob/develop/Dockerfile>`_.

.. note:: Please see :ref:`docker_local-label` for more information about
          reproducing Docker builds locally for debugging purposes.

