.. ##
.. ## Copyright (c) 2016-21, Lawrence Livermore National Security, LLC
.. ## and RAJA project contributors. See the RAJA/COPYRIGHT file
.. ## for details.
.. ##
.. ## SPDX-License-Identifier: (BSD-3-Clause)
.. ##

.. _release-label:

*******************************************
RAJA Release Process
*******************************************

The RAJA release process typically involves the following sequence of steps:

  #. Identify all work (features in development, outstanding PRs, etc.) to be 
     merged into the develop branch to be included in the release.
  #. Merge all necessary PRs into the develop branch.
  #. Make a **release candidate** branch for the release and finalize 
     the release by completing remaining release tasks on that branch. See
     :ref:`rcbranch-labl` for more information.
  #. When release candidate branch is complete, make a PR for it to be **merged
     into the main branch.** When that is approved and all CI checks pass,
     merge the release candidate branch into the RAJA main branch.
  #. On GitHub, make a new release with a tag for the release. Following our
     convention, the tag label should have the format ``vxx.yy.zz``. In the
     GitHub release description, note key features, bugfixes, etc. in the 
     release. These can be summarized from the ``RELEASE_NOTES.md`` file in 
     the RAJA repo. Also, add a note to the release description to download 
     the gzipped tarfile for the release rather than one of the assets GitHub 
     creates. This GitHub-created assets do not contain the RAJA submodules.

     Please follow the description of previous RAJA releases for consistency.
  #. Check out the main branch locally and pull in all the latest changes to it.     Then, generate the release tarfile for the release by running the script 
     ``.scripts/make_release_tarball.sh`` from the top-level RAJA directory. 
     If this is successful, a gzipped tarfile whose name includes the release 
     tag **with no extraneous SHA-1 hash information** will be in the top-level
     RAJA directory.
  #. Edit the release in GitHub and upload the tarfile to the release.
  #. Make a PR to merge the main branch into the develop branch. After it 
     passes all CI checks and is approved, merge the PR. This will ensure that
     all changes done to finalize the release will not be lost in future
     changes to the develop branch.

_rcbranch-label:

===========================
Release Candidate Branches
===========================

*Release candidate* branches are an important temporary branch type in Gitflow.
When the team has determined which features, documentation, bug fixes, etc. 
to include in a release and those items are complete and merged into the 
develop branch, a release candidate branch is made off of develop to finalize 
the release. Typically, a release candidate branch is named **rc-<release #>.**

.. note:: Creating a release candidate branch starts the next release cycle 
          on the develop branch whereby new work being performed on 
          feature branches can be merged into the develop branch.

.. important:: **No new feature development is done a release branch. Only bug 
               fixes, release documentation, and other release-oriented changes
               go into a release branch.**

Please see :ref:`release-label` for a description of how a release candidate
branch is used in the process. 

Preparation of a RAJA release candidate branch involves the following sequence 
of steps:

  #. Fill in list...

===========================
Hotfix Branches
===========================

Describe...
