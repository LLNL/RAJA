.. ##
.. ## Copyright (c) 2016-22, Lawrence Livermore National Security, LLC
.. ## and RAJA project contributors. See the RAJA/LICENSE file
.. ## for details.
.. ##
.. ## SPDX-License-Identifier: (BSD-3-Clause)
.. ##

.. _ci_tasks-label:

******************************************************
Continuous Integration (CI) Testing Maintenance Tasks
******************************************************

In :ref:`ci-label`, we described RAJA CI workflows. This section 
describes common CI testing maintenance tasks for RAJA and how to 
perform them.

.. _gitlab_ci_tasks-label:

=================
GitLab CI Tasks
=================

The tasks in this section apply to GitLab CI running on Livermore
Computing (LC) platforms.

Changing Build Specs
---------------------

The builds for each LC platform on which we run GitLab CI pipelines are
defined in ``<resource>-jobs.yml`` files in the `RAJA/.gitlab <https://github.com/LLNL/RAJA/tree/develop/.gitlab>`_ directory. The key items 
that change when a new build is added are:

  * the unique **label** that identifies the build on a web page for 
    a GitLab CI pipeline, and
  * the build **Spack spec**, which identifies the compiler and version,
    compiler flags, etc.

For example, an entry for a build using a clang compiler with CUDA is:

.. code-block:: bash

  ibm_clang_10_0_1_cuda_10_1_168:
    variables:
      SPEC: "+cuda cuda_arch=70 %clang@ibm.10.0.1 ^cuda@10.1.168"
    extends: .build_and_test_on_lassen

To update, change the corresponding spec item, such as clang compiler
or version, or cuda version. Then, update the label accordingly.

It is important to note that the build spec information must reside in 
the ``compilers.yaml`` and/or ``packages.yaml`` file for the system type
in the `radiuss-spack-configs <https://github.com/LLNL/RAJA/blob/develop/scripts>`_ submodule. If the desired information is not there,
try updating the submodule to a newer version. If the information
is still not available, create a branch in the 
`RADIUSS Spack Configs <https://github.com/LLNL/radiuss-spack-configs>`_ repo, add the needed spec info, and create a pull request.

.. important:: Build spec information used in RAJA GitLab CI pipelines
               must exist in the ``compilers.yaml`` file and/or 
               ``packages.yaml`` file for the appropriate system type in
               the `RADIUSS Spack Configs <https://github.com/LLNL/radiuss-spack-configs>`_ repo.

Changing Build/Run Parameters
------------------------------

The commands executed to acquire resources on each 
system/system-type on which we run GitLab CI are defined in the 
`RAJA/.gitlab-ci.yml <https://github.com/LLNL/RAJA/blob/develop/.gitlab-ci.yml>`_ file. The default execution time for each test pipeline is 
also defined in the file using the variable ``DEFAULT_TIME``. These 
commands and settings can remain as is for the most part. 

However, sometimes a particular pipeline will take longer to build and
run than the default allotted time. In this case, the default time can
be adjusted in the build spec information in the associated 
``<resource>-jobs.yml`` file discussed in the previous section. 
For example:

.. code-block:: bash

  xl_16_1_1_7_cuda:
    variables:
      SPEC: "+cuda %xl@16.1.1.7 cuda_arch=70 ^cuda@10.1.168 ^cmake@3.14.5"
      DEFAULT_TIME: 60
    allow_failure: true
    extends: .build_and_test_on_lassen

This example explicitly sets the build and test allocation time to 60 minutes:
``DEFAULT_TIME: 60``. Note that it also allows the pipeline to fail: 
``allow_failure: true``. We do this in some cases where certain tests are known
to fail regularly. This allows the overall check status to report as passing,
even though the test pipeline annotated this way may fail.


Adding Test Pipelines
---------------------

Adding a test pipeline involves adding a new entry in the 
``RAJA/.gitlab-ci.yml`` file.

.. important:: Build spec information used in RAJA GitLab CI pipelines
               must exist in the ``compilers.yaml`` file and/or 
               ``packages.yaml`` file for the appropriate system type in
               the `RADIUSS Spack Configs <https://github.com/LLNL/radiuss-spack-configs>`_ repo.


.. _azure_ci_tasks-label:

=================
Azure CI Tasks
=================

The tasks in this section apply to RAJA Azure Pipelines CI.

Changing Builds/Container Images
---------------------------------------

The builds we run in Azure are defined in the `RAJA/azure-pipelines.yml <https://github.com/LLNL/RAJA/blob/develop/azure-pipelines.yml>`_ file.
  
Linux/Docker
............

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
      make -j 6 &&\
      ctest -T test --output-on-failure

Each of our docker builds is built up on a base image maintained by RSE-Ops, a table of available base containers can be found `here <https://rse-ops.github.io/docker-images/>`_. We are also able to add target names to each build with ``AS ...``. This target name correlates to the ``docker_target: ...`` defined in ``azure-pipelines.yml``.

The base containers are shared across multiple projects and are regularly rebuilt. If bugs are fixed in the base containers the changes will be automatically propagated to all projects using them in their Docker builds.

Check `here <https://rse-ops.github.io/docker-images/>`_ for a list of all currently available RSE-Ops containers. Please see the `RSE-Ops Containers Project <https://github.com/rse-ops/docker-images>`_ on Github to get new containers built that aren't yet available.

Windows / MacOs
...............

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
............

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
...............

Windows and MacOS build / run parameters can be configured directly in ``azure-pipelines.yml``. CMake options can be configured with ``CMAKE_EXTRA_FLAGS`` for each job. The ``-j`` value can also be edited directly in the Azure ``script`` definitions for each job.

The commands executed to configure, build, and test RAJA for each 
pipeline in Azure are located in the `RAJA/Dockerfile <https://github.com/LLNL/RAJA/blob/develop/Dockerfile>`_ file. 
Each pipeline section begins with a line that ends with ``AS ...`` 
where the ellipses in the name of a build-test pipeline. The name label
matches an entry in the Docker test matrix in the 
``RAJA/azure-pipelines.yml`` file mentioned above.


.. _rajaperf_ci_tasks-label:

================================
RAJA Performance Suite CI Tasks
================================

The `RAJA Performance Suite <https://github.com/LLNL/RAJAPerf>`_ project CI
testing processes, directory/file structure, and dependencies are nearly 
identical to that for RAJA, which is described in :ref:`ci-label`. Specifically,

  * The RAJA Performance Suite GitLab CI process is driven by the 
    `RAJAPerf/.gitlab-ci.yml <https://github.com/LLNL/RAJAPerf/blob/develop/.gitlab-ci.yml>`_ file. 
  * The ``<resource>-jobs.yml`` and ``<resource>-templates.yml`` files reside 
    in the 
    `RAJAPerf/.gitlab <https://github.com/LLNL/RAJAPerf/tree/develop/.gitlab>`_ 
    directory.
  * The ``build_and_test.sh`` script resides in the `RAJAPerf/scripts/gitlab <https://github.com/LLNL/RAJAPerf/tree/develop/scripts/gitlab>`_ directory.
  * The `RAJAPerf/Dockerfile <https://github.com/LLNL/RAJAPerf/blob/develop/Dockerfile>`_ drives the Azure testing pipelines.
  
The main difference is that for GitLab CI, is that the Performance Suite uses 
the RAJA submodules for ``uberenv`` and ``radiuss-spack-configs`` located in 
the RAJA submodule to avoid redundant submodules. This is reflected in the
`RAJAPerf/.uberenv_config.json <https://github.com/LLNL/RAJAPerf/blob/develop/.uberenv_config.json>`_ 
file which point at the relevant RAJA submodule locations.

Apart from this minor difference, all CI maintenance and development tasks for
the RAJA Performance Suite follow the guidance in :ref:`ci_tasks-label`.
