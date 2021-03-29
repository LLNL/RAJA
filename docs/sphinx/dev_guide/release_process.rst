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
  #. Make a **release candidate** branch from develop for the release and 
     finalize the release by completing remaining release tasks on that branch.
     See :ref:`rcbranch-labl` for more information.
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
  #. Check out the main branch locally and make sure it is up-to-date.     
     Then, generate the release tarfile for the release by running the script 
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

_hotfixbranch-label:

===========================
Hotfix Branches
===========================

*Hotfix* branches are used in the (hopefully!) rare case when a bug is found
shortly after a release that has the potential for negative impact on RAJA
users. A hotfix branch is used to address the issue and make a new release
containing only the fixed code. 

A hotfix branch is *made from main* with the name **hotfix/<issue>**. The 
issue is fixed (hopefully quickly!) and the release notes file is updated on 
the hotfix branch for the pending bugfix release. The branch is tested, against 
user code if necessary, to make sure the issue is resolved. Then, a PR is made 
to merge into main and when it is approved and passes CI checks, it is merged 
into the main branch. Lastly, a new release is made in a fashion similar to the
process described in :ref:`release-label`. For completeness, here are the
main steps for performing a hotfix release:

  #. Make a **hotfix** branch for the release (hotfix/<issue>), fix the
     issue and verify if a user is involved, and update the release notes
     file as needed.
  #. When hotfix branch is ready, make a PR for it to be **merged
     into the main branch.** When that is approved and all CI checks pass,
     merge it into the RAJA main branch.
  #. On GitHub, make a new release with a tag for the release. Following our
     convention, the tag label should have the format ``vxx.yy.zz``. In the
     GitHub release description, note that the release is a bugfix release
     and describe the issue that is resolved. Also, add a note to the release 
     description to download the gzipped tarfile for the release rather than 
     one of the assets GitHub creates as in a normal release.
  #. Check out the main branch locally and make sure it is up-to-date.     
     Then, generate the release tarfile for the release by running the script 
     ``.scripts/make_release_tarball.sh`` from the top-level RAJA directory. 
     If this is successful, a gzipped tarfile whose name includes the release 
     tag **with no extraneous SHA-1 hash information** will be in the top-level
     RAJA directory.
  #. Make a PR to merge the main branch into the develop branch. After it 
     passes all CI checks and is approved, merge the PR. This will ensure that
     changes for the bugfix will exist in future development.
