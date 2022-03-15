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

The RAJA project employs multiple tools to run its tests for each GitHub
*pull request*, all of which must pass before the pull request can be merged.
These tools include:

  * **Azure.** This runs builds for Linux, Windows, and MacOS  environments 
    using a variety of compilers. While we do GPU builds for CUDA, HIP, and
    SYCL on Azure, RAJA tests are run for each non-GPU build.

  * **Appveyor.** This runs builds and tests for a Windows environment for two
    versions of the Visual Studio compiler.

  * **Gitlab CI.** This runs builds and tests on platforms in the Livermore
    Computing *Collaboration Zone*. This is a recent addition for RAJA and
    is a work-in-progress to get full coverage of compilers and tests we
    need to exercise.

These tools integrate seamlessly with GitHub. They will automatically
(re)run RAJA builds and tests as changes are pushed to each PR branch. Gitlab
CI execution on Livermore Computing resources has some restrictions which are
described below.

Gitlab CI support is still being developed to make it more easy to use with 
GitHub projects. The current state is described below.

.. note:: The status of checks (pass/fail, running status) for each of these 
          tools can be viewed by clicking the appropriate link in the check
          section of a pull request.

Gitlab CI
=========

If all memmbers of a GitHub project are members of the LLNL GitHub organization 
and have two-factor authentication enabled on their GitHub accounts, 
auto-mirroring on the Livermore Computing Collaboration Zone Gitlab server is
enabled. Thus, Gitlab CI will run automatically for those projects on pull 
requests that are made by project members. Otherwise, due to Livermore 
Computing security policies, Gitlab CI must be launched manually by a *blessed* 
GitHub user satisfying the constraints described above. To manually initiate
Gitlab CI on a pull request, add a comment with 'LGTM' in it.

It is important to note that RAJA shares its Gitlab CI workflow with 
other projects. See `Shared Gitlab CI Workflow <https://radiuss-ci.readthedocs.io/en/latest/uberenv.html#ci>`_ for more information.


.. _vettedspecs-label:

Vetted Specs
------------

The *vetted* compiler specs are those which we use during the RAJA Gitlab CI
testing process. These can be viewed by looking at files in the RAJA
``.gitlab`` directory. For example,

.. code-block:: bash

  $ ls -c1 .gitlab/*jobs.yml
  .gitlab/lassen-jobs.yml
  .gitlab/ruby-jobs.yml

lists the yaml files containing the Gitlab CI jobs for the lassen and ruby 
machines.

Then, executing a command such as:

.. code-block:: bash

  $ git grep -h "SPEC" .gitlab/ruby-jobs.yml | grep "gcc"
      SPEC: "%gcc@4.9.3"
      SPEC: "%gcc@6.1.0"
      SPEC: "%gcc@7.3.0"
      SPEC: "%gcc@8.1.0"

will list the specs vetted on the ruby platform.

More details to come...
