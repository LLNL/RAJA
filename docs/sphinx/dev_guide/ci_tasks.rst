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
Gitlab CI Tasks
=================

The tasks in this section apply to GitLab CI running on Livermore
Computing (LC) resources.

Changing Build Specs
---------------------

The builds for each LC platform on which we run Gitlab CI pipelines are
defined in ``<resource>-jobs.yml`` files in the `RAJA/.gitlab <https://github.com/LLNL/RAJA/tree/develop/.gitlab>`_ directory. The key items 
that change when a new build is added are:

  * the unique **label** that identifies the build on a webpage for 
    a Gitlab CI pipeline, and
  * the build **Spack spec**, which identifies the compiler and version,
    compiler flags, etc.

For example, the file section for build using a clang compiler along 
with cuda:: 

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

.. important:: Build spec information used in RAJA Gitlab CI pipelines
               must exist in the ``compilers.yaml`` file and/or 
               ``packages.yaml`` file for the appropriate system type in
               the `RADIUSS Spack Configs <https://github.com/LLNL/radiuss-spack-configs>`_ repo.

Changing Build/Run Parameters
------------------------------

The commands executed to acquire resources on each 
system/system-type on which we run Gitlab CI are defined in the 
`RAJA/.gitlab-ci.yml <https://github.com/LLNL/RAJA/blob/develop/.gitlab-ci.yml>`_ file. The default execution time for each test pipeline is 
also defined in the file using the variable ``DEFAULT_TIME``. These 
commands and settings can remain as is for the most part. 

However, sometimes a particular pipeline will take longer to build and
run than the default alloted time. In this case, the default time can
be adjusted in the build spec information in the associated 
``<resource>-jobs.yml`` file discussed in the previous section. 
For example::

  xl_16_1_1_7_cuda:
    variables:
      SPEC: "+cuda %xl@16.1.1.7 cuda_arch=70 ^cuda@10.1.168 ^cmake@3.14.5"
      DEFAULT_TIME: 60
    allow_failure: true
    extends: .build_and_test_on_lassen


Adding Test Pipelines
---------------------

.. _azure_ci_tasks-label:

=================
Azure CI Tasks
=================

The tasks in this section apply to RAJA Azure Pipelines CI.

Changing Builds/Container Images
---------------------------------------

The builds we run in Azure are defined in the `RAJA/azure-pipelines.yml <https://github.com/LLNL/RAJA/blob/develop/azure-pipelines.yml>`_ file.
  
Fill in details describing process to update images for new 
compilers/versions, etc. For example, explain how the::
 
  FROM ghcr.io/rse-ops/gcc-ubuntu-20.04:gcc-7.3.0 AS gcc7

stuff works....

Changing Build/Run Parameters
------------------------------

The commands executed to configure, build, and test RAJA for each 
pipeline in Azure are located in the `RAJA/Dockerfile <https://github.com/LLNL/RAJA/blob/develop/Dockerfile>`_ file. 
Each pipeline section begins with a line that ends with ``AS ...`` 
where the ellipses in the name of a build-test pipeline. The name label
matches an entry in the Docker test matrix in the 
``RAJA/azure-pipelines.yml`` file mentioned above.
