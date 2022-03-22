.. ##
.. ## Copyright (c) 2016-22, Lawrence Livermore National Security, LLC
.. ## and RAJA project contributors. See the RAJA/LICENSE file
.. ## for details.
.. ##
.. ## SPDX-License-Identifier: (BSD-3-Clause)
.. ##

.. _contributing-label:

*********************
Contributing to RAJA
*********************

RAJA is a collaborative open source software project and we encourage 
contributions from anyone who wants to add features or improve its
capabilities. This section describes the following:

  * GitHub project access
  * How to develop a RAJA *pull request* (PR) contribution.
  * Requirements that must be met for a PR to be merged.

We assume contributors are familiar with 
`Git <https://git-scm.com/>`_, which we use for source code version control,
and `GitHub <https://github.com/>`_, which is where our project is hosted. 

.. important:: * Before a PR can be merged into RAJA, all test checks must pass 
                 and the PR must be approved by at least one member of the 
                 core RAJA team.
               * Each RAJA contribution (feature, bugfix, etc.) must include 
                 adequate tests, documentation, and code examples. What is
                 *adequate* is determined via PR reviews using the professional
                 judgment of RAJA team members and contributors.

=======================
GitHub Project Access
=======================

RAJA maintains three levels of project access on it GitHub project:

  * **Core team member.** Individuals considered to be on the core RAJA team. 
    They participate in project meetings, discussions, etc. and are members of 
    the LLNL GitHub organization and the ``RAJA-core`` GitHub team. They
    have privileges to create branches, push code changes to the RAJA repo,
    make PRs and merge them when they are approved and all checks have passed. 
  * **Regular contributors.** Individuals who are members of the LLNL
    GitHub organization and the ``RAJA-contrib`` GitHub team are considered 
    sufficiently frequent contributors that they have been given permissions 
    to create branches, push code changes to the RAJA repo, and make PRs. 
    However, and this is mainly due to the way GitHub structures its project 
    access levels, these individuals cannot merge PRs.
  * **Everyone else.** Anyone with a GitHub account is welcome to contribute
    to the RAJA project. Individuals outside of the previous two groups can 
    make PRs in the RAJA project, but must do so from a branch on a *fork* of 
    the RAJA repo. This is described below.

=======================
Pull Request Basics
=======================

The following figure shows the basic elements of the RAJA PR contribution 
workflow. Some details vary depending on RAJA GitHub project access level 
of the contributor. The process involves four main steps:

  #. A RAJA contributor makes a PR on the RAJA GitHub project to merge a
     branch on which she has developed a contribution into another RAJA branch.
     Most often, this is the develop branch.
  #. Then, GitHub triggers Azure and Gitlab CI checks. Running and pass/fail
     status is reported back to GitHub where it can be viewed and monitored.
  #. Meanwhile RAJA team members review the PR, suggesting changes and/or
     approving when they think it is ready to merge.
  #. When all checks pass and the PR is approved, it PR may be merged.

.. figure:: ./figures/PR-Workflow.png

   The four main steps in the RAJA pull request (PR) process, which are
   common practices for many software projects.

If you want more information about the pull request (PR) process, GitHub has 
a good `PR guide <https://help.github.com/articles/about-pull-requests/>`_ on
PR basics.

When you create a RAJA PR, you must enter a description of the contents of the 
PR. The project has a GitHub *PR template* for this purpose for you to fill in.
Be sure to add a descriptive title explaining the bug you fixed or the feature 
you have added and any other relevant details that will assist others in 
reviewing your contribution.

When a PR is created in RAJA, it will be run through our automated testing
processes and be reviewed by RAJA team members. When the PR passes all 
tests and it is approved, a member of the RAJA core team will merge it.

============
Forking RAJA
============

If you are not a member of the core RAJA development team, or a recognized
RAJA contributor, then you do not have permission to create a branch in the 
RAJA GitHub repository. This is due to the policy adopted by the LLNL
organization on GitHub in which the RAJA project resides. Fortunately, you may 
still contribute to RAJA by `forking the RAJA repo 
<https://github.com/LLNL/RAJA/fork>`_. Forking creates a copy of the RAJA 
repository that you own. You can push code changes on that copy to GitHub and 
create a pull request in the RAJA project.

.. note:: A contributor who is not a member of the core RAJA development team,
          or a recognized RAJA contributor, cannot create a branch in the RAJA 
          GitHub repo. However, anyone can create a fork of the 
          RAJA project and create a pull request in the RAJA project.

=========================
Developing RAJA Code
=========================

New features, bugfixes, and other changes are developed on a **feature branch.**
Each such branch should be based on the most current RAJA ``develop`` branch. 
For more information on the branch development model used in RAJA, please see
:ref:`branching-label`. When you want to make a contribution, first ensure 
you have an up-to-date copy of the ``develop`` branch locally by running the
following commands:

.. code-block:: bash

    $ git checkout develop
    $ git pull origin develop
    $ git submodule update --init --recursive

Then, in your local space, you will be on the current version of develop branch
with all RAJA submodules synchronized with that. 

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
new code, as needed. If you are creating new functionality, please add 
documentation to the appropriate section of the `RAJA User Guide <https://readthedocs.org/projects/raja/>`_. The source files for the RAJA documentation are 
maintained in the ``RAJA/docs`` directory. Also, consider adding example
code(s) that illustrate usage of the new features you develop. These should
be placed in the ``RAJA/examples`` directory and referenced in the RAJA User
Guide as needed.

After your new code is complete, you've tested it, and developed appropriate
documentation, you can push your branch to GitHub and create a PR in the RAJA
project. It will be reviewed by members of the RAJA team, who will provide 
comments, suggestions, etc. 

Note that not all required :ref:`ci-label` can be run on a PR made from a branch
in a fork of the RAJA repo. When the RAJA team is comfortable with your PR,
it will be pulled into the RAJA GitHub repo (see :ref:`prfromfork-label`).
Then, it will run through all required testing and receive final reviews. 
After it is approved and all CI testing checks pass, your contribution will 
be merged into the RAJA repository, most likely the develop branch.

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

.. _prfromfork-label:

===========================================================
Testing Pull Requests from Branches in Forked Repositories
===========================================================

Due to LLNL security policies, some RAJA pull requests will not be able to
be run through all RAJA CI tools. The Livermore Computing (LC) 
Collaboration Zone (CZ) Gitlab instance restricts which GitHub PRs may 
automatically run through its CI test pipelines. 
In particular, a PR made from branch on a forked repository will not trigger 
Gitlab CI checks. Gitlab CI on internal LLNL platforms will only be run on PRs 
that are made from branches in the GitHub RAJA repository. 
See :ref:`ci-label` for more information about RAJA PR testing.

.. note:: **RAJA team process for accepting PR contributions from forked repos:**

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

