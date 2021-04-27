.. ##
.. ## Copyright (c) 2016-21, Lawrence Livermore National Security, LLC
.. ## and RAJA project contributors. See the RAJA/COPYRIGHT file
.. ## for details.
.. ##
.. ## SPDX-License-Identifier: (BSD-3-Clause)
.. ##

.. _contributing-label:

*********************
Contributing to RAJA
*********************

Since RAJA is a collaborative open source software project, we embrace 
contributions from anyone who wants to add features or improve its existing 
capabilities. This section describes basic processes to follow
for individuals outside of the core RAJA team to contribute new features or 
bugfixes to RAJA. It assumes you are familiar with 
`Git <https://git-scm.com/>`_, which we use for source code version control,
and `GitHub <https://github.com/>`_, which is where our project is hosted. 

This section describes development processes, such as:

  * Making a fork of the RAJA repository 
  * Creating a branch for development
  * Creating a pull request (PR)
  * Tests that your PR must pass before it can be merged into RAJA

============
Forking RAJA
============

If you are not a member of the LLNL organization on GitHub and of 
the core RAJA team of developers, then you do not have permission to create 
a branch in the RAJA repository. This is due to the policy adopted by the LLNL
organization on GitHub in which the RAJA project resides. Fortunately, you may 
still contribute to RAJA by `forking the RAJA repo 
<https://github.com/LLNL/RAJA/fork>`_. Forking creates a copy of the RAJA 
repository that you own. You can push code changes on that copy to GitHub and 
create a pull request in the RAJA project.

.. note:: A contributor who is not a member of the LLNL GitHub organization 
          and the core team of RAJA developers cannot create a
          branch in the RAJA repo. However, anyone can create a fork of the 
          RAJA project and create a pull request in the RAJA project.

=========================
Developing RAJA Code
=========================

New features, bugfixes, and other changes are developed on a **feature branch.**
Each such branch should be based on the RAJA ``develop`` branch. For more 
information on the branch development model used in RAJA, please see
:ref:`branching-label`. When you want to make a contribution, first ensure 
you have an up-to-date copy of the ``develop`` branch locally:

.. code-block:: bash

    $ git checkout develop
    $ git pull origin develop

----------------------
Developing a Feature
----------------------

Assuming you are on the develop branch in your local copy of the RAJA repo,
and the branch is up-to-date, the first step toward developing a RAJA feature
is to create a new branch on which to perform your development. For example:

.. code-block:: bash

    $ git checkout -b feature/<name-of-feature>

Proceed to modify your branch by committing changes with reasonably-sized 
work portions (i.e., *atomic commits*), and add tests that will exercise your 
new code. If you are creating new functionality, please add documentation to 
the appropriate section of the `RAJA User Guide <https://readthedocs.org/projects/raja/>`_. The source files for the RAJA documentation are maintained in 
the ``RAJA/docs`` directory.

After your new code is complete, you've tested it, and developed appropriate
documentation, you can push your branch to GitHub and create a PR in the RAJA
project. It will be reviewed by members of the RAJA team, who will provide 
comments, suggestions, etc. After it is approved and all CI checks pass, your 
contribution will be merged into the RAJA repository.

.. important:: When creating a branch that you intend to be merged into the 
               RAJA repo, please give it a succinct name that clearly describes 
               the contribution.  For example, **feature/<name-of-feature>** 
               for a new feature, **bugfix/<fixed-issue>** for a bugfix, etc.

--------------------
Developing a Bug Fix
--------------------

Contributing a bugfix follows the same process as described above. Be sure to
indicate in the name of your branch that it is for a bugfix; for example:

.. code-block:: bash

    $ git checkout -b bugfix/<fixed-issue>

We recommend that you add a test that reproduces the issue you have found
and demonstrates that the issue is resolved. To verify that you have done
this properly, build the code for your branch and then run ``make test`` to 
ensure that your new test passes.

When you are done, push your branch to GitHub, then create a PR in the RAJA
project.

-----------------------
Creating a Pull Request
-----------------------

You can create a pull request (PR) 
`here <https://github.com/LLNL/RAJA/compare>`_. GitHub has a good 
`PR guide <https://help.github.com/articles/about-pull-requests/>`_ on
PR basics if you want more information. Ensure that the base branch for your 
PR is the ``develop`` branch of RAJA.

When you create a RAJA PR, you must enter a description of the contents of the 
PR. We have a *PR template* for this purpose for you to fill in. Be sure to add
a descriptive title explaining the bug you fixed or the feature you have added
and any other relevant details that will assist the RAJA team in reviewing your
contribution.

When a PR is created in RAJA, it will be run through our automated testing
processes and be reviewed by RAJA team members. When the PR passes all 
tests and it is approved, a member of the RAJA team will merge it.

.. note:: Before a PR can be merged into RAJA, all CI checks must pass and
          the PR must be approved by a member of the core team. 

-----
Tests
-----

RAJA uses multiple continuous integration (CI) tools to test every pull
request. See :ref:`ci-label` for more information. 

All RAJA tests are in the ``RAJA/test`` directory and are split into 
*unit tests* and *functional tests*. Unit tests are intended to test basic
interfaces and features of individual classes, methods, etc. Functional tests
are used to test combinations of RAJA features. We have organized our 
tests to make it easy to see what is being tested and easy to add new tests.
For example, tests for each programming model back-end are exercised using
the same common, parameterized test code to ensure back-end support is
consistent.

.. important:: Please follow the sub-directory structure and code implementation
               pattern for existing tests in the ``RAJA/test`` directory when 
               adding or modifying tests. 

.. _prfromfork-label::

-----------------------------------------------------------
Testing Pull Requests from Branches in Forked Repositories
-----------------------------------------------------------

Due to LLNL security policies and RAJA project policies, only a PR created
by someone on the RAJA core development team will be run automatically
through all RAJA CI tools. In particular, a PR made from branch on a forked 
repository will not trigger Gitlab and Travis CI checks. Gitlab CI on internal 
LLNL platforms and Travis CI will only be run on PRs that are made from 
branches in the GitHub RAJA repository.

.. note:: **RAJA core team members:**

          To facilitate testing contributions in PRs from forked repositories, 
          we maintain a script to pull a PR branch from a forked repo into the 
          RAJA repo. First, identify the number of the PR. Then, run the 
          script from the top-level RAJA directory::

            $ ./scripts/make_local_branch_from_fork_pr -b <PR #>

          If successful, this will create a branch in your local copy of the
          RAJA repo labeled ``pr-from-fork/<PR #>`` and you will be on that
          local branch in your checkout space. To verify this, you can run
          the following command after you run the script::

            $ git branch

          You will see the new branch in the listing of branches and the branch
          you are on will be starred.

          You can push the new branch to the RAJA repo on GitHub::

            $ git push origin <branch-name>

          and make a PR for the new branch. It is good practice to reference 
          the original PR in the description of the new PR to track the 
          original PR discussion and reviews.

          All CI checks will be triggered to run on the new PR made in the
          RAJA repo. When everything passes and the PR is approved, it may 
          be merged. When it is merged, the original PR from the forked repo 
          will be closed and marked as merged unless it is referenced 
          elsewhere, such as in a GitHub issue. If this is the case, then the 
          original PR (from the forked repo) must be closed manually.

