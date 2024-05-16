.. ##
.. ## Copyright (c) 2016-24, Lawrence Livermore National Security, LLC
.. ## and RAJA project contributors. See the RAJA/LICENSE file
.. ## for details.
.. ##
.. ## SPDX-License-Identifier: (BSD-3-Clause)
.. ##

.. _ci_tasks-label:

*****************************************************
Continuous Integration (CI) Testing Maintenance Tasks
*****************************************************

In :ref:`ci-label`, we described RAJA CI workflows. This section describes 
how to perform common RAJA CI testing maintenance tasks.

.. _gitlab_ci_tasks-label:

===============
GitLab CI Tasks
===============

The tasks in this section apply to GitLab CI testing on Livermore 
Computing (LC) platforms. LC folks and others maintain Confluence pages
with a lot of useful information for setting up and maintaining GitLab CI
for a project, mirroring a GitHub to GitLab, etc. Please refer to `LC GitLab CI <https://lc.llnl.gov/confluence/display/GITLAB/GitLab+CI>`_ for such information.

Changing build and test configurations
--------------------------------------

The build for each test we run is defined by a Spack spec in one of two places,
depending on whether it is *shared* with other projects or it is specific to 
RAJA. The details are described in :ref:`gitlab_ci_pipelines-label`.

Remove a configuration
^^^^^^^^^^^^^^^^^^^^^^

To remove a RAJA-specific test configuration, simply delete the entry for it in
the ``RAJA/.gitlab/jobs/<MACHINE>.yml`` file where it is defined. Here,
``MACHINE`` is the name of an LC platform where GitLab CI is run.

To remove a shared configuration, it must be removed from the appropriate
``gitlab/radiuss-jobs/<MACHINE>.yml`` file in the `RADIUSS Spack Configs
<https://github.com/LLNL/radiuss-spack-configs>`_ project.  Create a branch
there, remove the job entry, and create a pull request.

.. important:: The RADIUSS Spack Configs project is used by several other
   projects.  When changing a shared configuration file, please make sure the
   change is OK with those other projects.

Add a configuration
^^^^^^^^^^^^^^^^^^^

To add a RAJA-specific test configuration, add an entry for it to the
``RAJA/.gitlab/jobs/<MACHINE>.yml`` file, where ``MACHINE`` is the name of the
LC platform where it will be run. When adding a test configuration, it is
important to note two items that must be specified properly:

  * A unique **job label**, which identifies it in the machine configuration
    file and also on a web page for a GitLab CI pipeline
  * A build **Spack spec**, which identifies the compiler and version,
    compiler flags, build options, etc.

For example, an entry for a build using the clang 12.0.1 compiler with CUDA 
11.5.0 on the LC lassen machine would be something like this:

.. code-block:: bash

  clang_12_0_1_cuda_11_5_0:
    variables:
      SPEC: " ~shared +openmp +tests +cuda cuda_arch=70 %clang@12.0.1 ^cuda@11.5.0"
    extends: .job_on_lassen

Here, we enable OpenMP and CUDA, both of which must be enabled to test those
RAJA back-ends, and specify the CUDA target architecture 'sm_70'.

To add a shared configuration, it must be added to the appropriate
``gitlab/radiuss-jobs/<MACHINE>.yml`` file in the `RADIUSS Spack Configs
<https://github.com/LLNL/radiuss-spack-configs>`_ project. Create a branch
there, add the job entry, and create a pull request.

.. important:: The RADIUSS Spack Configs project is used by several other
   projects. When changing a shared configuration file, please make sure the
   change is OK with those other projects.

Modifying a configuration
^^^^^^^^^^^^^^^^^^^^^^^^^

To change an existing configuration, change the relevant information in the
configuration in the appropriate ``RAJA/.gitlab/jobs/<MACHINE>.yml`` file. Make
sure to also modify the job label as needed, so it is descriptive of the
configuration (and remains unique!!).

To modify a shared configuration, it must be changed in the appropriate
``gitlab/radiuss-jobs/<MACHINE>.yml`` file in the `RADIUSS Spack Configs
<https://github.com/LLNL/radiuss-spack-configs>`_ project. Create a branch
there, modify the job entry, and create a pull request.

.. important:: Build spec information used in RAJA GitLab CI pipelines must
   exist in the ``compilers.yaml`` file and/or ``packages.yaml`` file for the
   appropriate system type in the `RADIUSS Spack Configs
   <https://github.com/LLNL/radiuss-spack-configs>`_ repo.

   If the desired entry is not there, but exists in a newer version of the RADIUSS
   Spack Configs project, update the RAJA submodule to use the newer version. If
   the information does not exist in any version of the RADIUSS Spack Configs
   project, create a branch there, add the needed spec info, and create a pull
   request. Then, when that PR is merged, update the RAJA submodule for the
   RADIUSS Spack Configs project to the new version.

Changing run parameters
^^^^^^^^^^^^^^^^^^^^^^^

The parameters for each system/scheduler on which we run GitLab CI for
RAJA, such as job time limits, resource allocations, etc. are defined in the 
``RAJA/.gitlab/custom-jobs-and-variables.yml`` file. This information can
remain as is, for the most part, and should not be changed unless absolutely 
necessary.

For example, sometimes a particular job will take longer to build and run than
the default allotted time for jobs on a machine. In this case, the time for the
job can be adjusted in the job entry in the associated
``RAJA/.gitlab/jobs/<MACHINE>.yml`` file. For example:

.. code-block:: bash

  gcc_8_1_0:
  variables:
    SPEC: " ${PROJECT_RUBY_VARIANTS} %gcc@8.1.0 ${PROJECT_RUBY_DEPS}"
    RUBY_BUILD_AND_TEST_JOB_ALLOC: "--time=60 --nodes=1"
  extends: .job_on_ruby

This example sets the build and test allocation time to 60 minutes and the
the run resource to one node.

Allowing failures
^^^^^^^^^^^^^^^^^

Sometimes a shared job configuration is known to fail for RAJA. To allow
the job to fail without the CI check associated with it failing, we can
annotate the job for this. For example:

.. code-block:: bash

  ibm_clang_9_0_0:
    variables:
      SPEC: " ${PROJECT_LASSEN_VARIANTS} %clang@ibm.9.0.0 ${PROJECT_LASSEN_DEPS}"
    extends: .job_on_lassen
    allow_failure: true

.. important:: When a shared job needs to be modified for RAJA specifically, we
   call that "overriding": The job label must be kept the same as in the
   ``gitlab/radiuss-jobs/<MACHINE>.yml`` file in the `RADIUSS Spack Configs
   <https://github.com/LLNL/radiuss-spack-confgs>`_, and the RAJA-specific job
   can be adapted. If you override a shared job, please add a comment to
   describe the change in the ``RAJA/.gitlab/jobs/<MACHINE>.yml`` file where
   the job is overridden.


Building the Intel clang + SYCL HIP compiler for use in CI
----------------------------------------------------------

The SYCL CI tests on corona rely on a custom Intel Clang SYCL compiler that we 
build ourselves. This compiler lives in the ``/usr/workspace/raja-dev/`` folder so 
that it can be accessed by the gitlab CI system. Since the intel compiler does
not do releases in the typical sense (they simply update their repo *every night*), 
it may become necessary to periodically build a new version of the compiler to 
ensure that we are using the most up-to-date version available. The steps for 
building, installing, and running are shown here.

Building the Compiler
^^^^^^^^^^^^^^^^^^^^^

.. important:: Because intel updates their compiler repo daily, there is a nonzero possibility that the head of the sycl branch will fail to build. 
  In the event that it does not build, try checking out a different commit. On the intel/llvm GitHub page, one can see which of their 
  commits builds by checking the status badge next to each commit. Look for a commit that passes. 


#. Load the version of GCC that you want to use. In this case, we are using LC's gcc/10.3.1-magic installation::

    module load gcc/10.3.1-magic

#. Load the version of rocm that you want to use. In this case, we are using 5.7.1::

    module load rocm/5.7.1 

#. Clone the "sycl" branch of intel's llvm compiler fork::

    git clone https://github.com/intel/llvm -b sycl

#. cd into that folder:: 
    
    cd llvm

   In the event that the head of the sycl branch does not build, run ``git checkout <git sha>`` to checkout a version that does build.

#. Build the compiler. 

   Note that in this example, we are using rocm5.7.1, but one can change the version they wish to use simply by changing the paths in the configure step

   a. Configure

     .. code-block:: bash 

        srun -n1 /usr/bin/python3 buildbot/configure.py --hip -o buildrocm5.7.1 \
        --cmake-gen "Unix Makefiles" \
        --cmake-opt=-DSYCL_BUILD_PI_HIP_ROCM_DIR=/opt/rocm-5.7.1 \
        --cmake-opt=-DSYCL_BUILD_PI_HIP_ROCM_INCLUDE_DIR=/opt/rocm-5.7.1/include \
        --cmake-opt=-DSYCL_BUILD_PI_HIP_ROCM_LIB_DIR=/opt/rocm-5.7.1/lib \
        --cmake-opt=-DSYCL_BUILD_PI_HIP_INCLUDE_DIR=/opt/rocm-5.7.1/include \
        --cmake-opt=-DSYCL_BUILD_PI_HIP_HSA_INCLUDE_DIR=/opt/rocm-5.7.1/hsa/include/hsa \
        --cmake-opt=-DSYCL_BUILD_PI_HIP_LIB_DIR=/opt/rocm-5.7.1/lib \
        --cmake-opt=-DUR_HIP_ROCM_DIR=/opt/rocm-5.7.1 \
        --cmake-opt=-DUR_HIP_INCLUDE_DIR=/opt/rocm-5.7.1/include \
        --cmake-opt=-DUR_HIP_HSA_INCLUDE_DIR=/opt/rocm-5.7.1/hsa/include/hsa \
        --cmake-opt=-DUR_HIP_LIB_DIR=/opt/rocm-5.7.1/lib

   b. Build

     .. code-block:: bash

      srun -n1 /usr/bin/python3 buildbot/compile.py -o buildrocm5.7.1

#. Test the compiler

   Follow the steps in the next section "Using the compiler" to test this installation

#. Install

  a. The build step will install the compiler to the folder ``buildrocm<version>/install``. Simply copy this folder to the ``/usr/workspace/raja-dev/`` directory using the naming scheme ``clang_sycl_<git sha>_hip_gcc<version>_rocm<version>``

  #. Set the permissions of the folder, and everything in it to 750::

      chmod 750 /usr/workspace/raja-dev/<foldername>/ -R  

  #. Change the group of the folder and everything in it to raja-dev::

      chgrp raja-dev /usr/workspace/raja-dev/<foldername>/ -R  


Using the compiler
^^^^^^^^^^^^^^^^^^

#. Load the version of rocm that you used when building the compiler::

    module load rocm/5.7.1

#. Navigate to the root of your local checkout space of the RAJA repo::

    cd /path/to/raja

#. Run the test config script::

    ./scripts/lc-builds/corona_sycl.sh /usr/workspace/raja-dev/clang_sycl_2f03ef85fee5_hip_gcc10.3.1_rocm5.7.1

   Note that at the time of writing, the newest compiler we had built was at ``clang_sycl_2f03ef85fee5_hip_gcc10.3.1_rocm5.7.1``

#. cd into the auto generated build directory::

    cd {build directory}

#. Run the tests::

    make -j


==============
Azure CI Tasks
==============

The tasks in this section apply to RAJA Azure Pipelines CI.

Changing Builds/Container Images
--------------------------------

The builds we run in Azure are defined in the `RAJA/azure-pipelines.yml <https://github.com/LLNL/RAJA/blob/develop/azure-pipelines.yml>`_ file.
  
Linux/Docker
^^^^^^^^^^^^

To update or add a new compiler / job to Azure CI we need to edit both ``azure-pipelines.yml`` and ``Dockerfile``.

If we want to add a new Azure pipeline to build with ``compilerX``, then in ``azure-pipelines.yml`` we can add the job like so::

  -job: Docker
    ...
    strategy:
      matrix:
        ...
        compilerX: 
          docker_target: compilerX

Here, ``compilerX:`` defines the name of a job in Azure. ``docker_target: compilerX`` defines a variable ``docker_target``, which is used to determine what part of the ``Dockerfile`` to run.

In the ``Dockerfile`` we will want to add our section that defines the commands for the ``compilerX`` job.::

  FROM ghcr.io/rse-ops/compilerX-ubuntu-20.04:compilerX-XXX AS compilerX
  ENV GTEST_COLOR=1
  COPY . /home/raja/workspace
  WORKDIR /home/raja/workspace/build
  RUN cmake -DCMAKE_CXX_COMPILER=compilerX ... && \
      make -j 6 && \
      ctest -T test --output-on-failure && \
      cd .. && rm -rf build

Each of our docker builds is built up on a base image maintained by RSE-Ops, a table of available base containers can be found `here <https://rse-ops.github.io/docker-images/>`_. We are also able to add target names to each build with ``AS ...``. This target name correlates to the ``docker_target: ...`` defined in ``azure-pipelines.yml``.

The base containers are shared across multiple projects and are regularly rebuilt. If bugs are fixed in the base containers the changes will be automatically propagated to all projects using them in their Docker builds.

Check `here <https://rse-ops.github.io/docker-images/>`_ for a list of all currently available RSE-Ops containers. Please see the `RSE-Ops Containers Project <https://github.com/rse-ops/docker-images>`_ on GitHub to get new containers built that aren't yet available.

Windows / MacOS
^^^^^^^^^^^^^^^

We run our Windows / MacOS builds directly on the Azure virtual machine instances. In order to update the Windows / MacOS instance we can change the ``pool`` under ``-job: Windows`` or ``-job: Mac``::
  
  -job: Windows
    ...
    pool:
      vmImage: 'windows-2019'
    ...
  -job: Mac
    ...
    pool:
      vmImage: 'macOS-latest'

Changing Build/Run Parameters
-----------------------------

Linux/Docker
^^^^^^^^^^^^

We can edit the build and run configurations of each docker build, in the ``RUN`` command. Such as adding CMake options or changing the parallel build value of ``make -j N`` for adjusting throughput.

Each base image is built using `spack <https://github.com/spack/spack>`_. For the most part the container environments are set up to run our CMake and build commands out of the box. However, there are a few exceptions where we need to ``spack load`` specific modules into the path.

  * **Clang** requires us to load LLVM for OpenMP runtime libraries.::

      . /opt/spack/share/spack/setup-env.sh && spack load llvm

    **CUDA** for the cuda runtime.::

      . /opt/spack/share/spack/setup-env.sh && spack load cuda

    **HIP** for the hip runtime and llvm-amdgpu runtime libraries.::

      . /opt/spack/share/spack/setup-env.sh && spack load hip llvm-amdgpu

    **SYCL** requires us to run setupvars.sh::

      source /opt/view/setvars.sh 

Windows / MacOS
^^^^^^^^^^^^^^^

Windows and MacOS build / run parameters can be configured directly in ``azure-pipelines.yml``. CMake options can be configured with ``CMAKE_EXTRA_FLAGS`` for each job. The ``-j`` value can also be edited directly in the Azure ``script`` definitions for each job.

The commands executed to configure, build, and test RAJA for each 
pipeline in Azure are located in the `RAJA/Dockerfile <https://github.com/LLNL/RAJA/blob/develop/Dockerfile>`_ file. 
Each pipeline section begins with a line that ends with ``AS ...`` 
where the ellipses in the name of a build-test pipeline. The name label
matches an entry in the Docker test matrix in the 
``RAJA/azure-pipelines.yml`` file mentioned above.


.. _rajaperf_ci_tasks-label:

===============================
RAJA Performance Suite CI Tasks
===============================

The `RAJA Performance Suite <https://github.com/LLNL/RAJAPerf>`_ project CI
testing processes, directory/file structure, and dependencies are nearly
identical to that for RAJA, which is described in :ref:`ci-label`.
Specifically,

  * The RAJA Performance Suite GitLab CI process is driven by the
    `RAJAPerf/.gitlab-ci.yml
    <https://github.com/LLNL/RAJAPerf/blob/develop/.gitlab-ci.yml>`_ file.
  * The ``custom-jobs-and-variables.yml`` and ``subscribed-pipelines.yml``
    files reside in the `RAJAPerf/.gitlab
    <https://github.com/LLNL/RAJAPerf/tree/develop/.gitlab>`_ directory.
  * The ``build_and_test.sh`` script resides in the `RAJAPerf/scripts/gitlab
    <https://github.com/LLNL/RAJAPerf/tree/develop/scripts/gitlab>`_ directory.
  * The `RAJAPerf/Dockerfile
    <https://github.com/LLNL/RAJAPerf/blob/develop/Dockerfile>`_ drives the
    Azure testing pipelines.

The Performance Suite GitLab CI uses the ``uberenv`` and
``radiuss-spack-configs`` versions located in the RAJA submodule to make the
testing consistent across projects and avoid redundancy. This is reflected in
the `RAJAPerf/.uberenv_config.json
<https://github.com/LLNL/RAJAPerf/blob/develop/.uberenv_config.json>`_ file
which point at the relevant RAJA submodule locations. That is the paths contain
``tpl/RAJA/...``.

Apart from this minor difference, all CI maintenance and development tasks for
the RAJA Performance Suite follow the same pattern that is described in 
:ref:`ci_tasks-label`.
