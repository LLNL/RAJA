.. ##
.. ## Copyright (c) 2016-21, Lawrence Livermore National Security, LLC
.. ## and RAJA project contributors. See the RAJA/COPYRIGHT file
.. ## for details.
.. ##
.. ## SPDX-License-Identifier: (BSD-3-Clause)
.. ##

.. _ci:

************************************
Continuous Integration (CI) Testing
************************************

The RAJA project employes multiple tools to run its tests for each GitHub
*pull request*, all of which must pass before the pull request can be merged.
These tools include:

  * **Travis CI.** This runs builds for a Linux system for multiple versions of
    the GNU and clang compilers, and one version of the Intel, nvcc (CUDA), and 
    HIP compilers. RAJA tests are run each non-GPU build.
  * **Appveyor.** This runs builds and tests for a Windows environment for two
    versions of the Visual Studio compiler.
  * **Gitlab CI.** Runs builds and tests on platforms in the Livermore
    Computing *Collaboration Zone*. This is a recent addition for RAJA and
    is a work-in-progress to get full coverage of compilers and tests we
    need to exercise.

Travis and Appveyor integrate seamlessly with GitHub. So they will automatically
(re)run RAJA builds and tests as changes are pushed to each PR branch.

Gitlab CI support is still being developed to make it more easy to use with 
GitHub projects. The current state is described below.

.. note:: The status of checks (pass/fail, running status) for each of these 
          tools can be viewed by clicking the appropriate link in the check
          section of a pull request.

Gitlab CI
=========

Due to Livermore Computing security policies, Gitlab CI must be launched 
manually by a *blessed* GitHub user. Specifially, one must be a member of the
LLNL GitHub organization and have tw-factor authentication enabled on your
GitHub account. If you satisfy these requirements, you can initiate Gitlab CI
on a pull request by adding a comment with 'LGTM' in it.

.. note:: Gitlab CI is run on Livermore Computing systems that are heavily used
          and so throughput is less than ideal. To avoid squandering resources,
          **it is important that all RAJA developers only launch Gitlab CI when
          a pull request has passed all other checks and is ready to be 
          merged.** Also, be aware that when a PR is merged, all CI checks will 
          need to be rerun on other PR branches after they are updated with the 
          merged changes. So it's not prudent to have Gitlab CI checks running 
          (or queued to be run) on more than one branch at a time.

It is important to note that RAJA shares its Gitlab CI workflow with 
other projects. See `Shared Gitlab CI Workflow <https://radiuss-ci.readthedocs.io/en/latest/uberenv.html#ci>`_ for more information.

Vetted Specs
------------

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

will list the specs vetted on the quartz platform.

More details to come...
