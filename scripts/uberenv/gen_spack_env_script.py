###############################################################################
# Copyright (c) 2015-2019, Lawrence Livermore National Security, LLC.
#
# Produced at the Lawrence Livermore National Laboratory
#
# LLNL-CODE-716457
#
# All rights reserved.
#
# This file is part of Ascent.
#
# For details, see: http://ascent.readthedocs.io/.
#
# Please also read ascent/LICENSE
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice,
#   this list of conditions and the disclaimer below.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the disclaimer (as noted below) in the
#   documentation and/or other materials provided with the distribution.
#
# * Neither the name of the LLNS/LLNL nor the names of its contributors may
#   be used to endorse or promote products derived from this software without
#   specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL LAWRENCE LIVERMORE NATIONAL SECURITY,
# LLC, THE U.S. DEPARTMENT OF ENERGY OR CONTRIBUTORS BE LIABLE FOR ANY
# DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES  (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS
# OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
# HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
# STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING
# IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
#
###############################################################################
import os
import sys
import subprocess

from os.path import join as pjoin

# if you have bad luck with spack load, this
# script is for you!
#
# Looks for subdir: spack or uberenv_libs/spack
# queries spack for given package names and
# creates a bash script that adds those to your path
#
#
# usage:
# python gen_spack_env_script.py [spack_pkg_1 spack_pkg_2 ...]
#

def sexe(cmd,ret_output=False,echo = True):
    """ Helper for executing shell commands. """
    if echo:
        print("[exe: {}]".format(cmd))
    if ret_output:
        p = subprocess.Popen(cmd,
                             shell=True,
                             stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT)
        res = p.communicate()[0]
        res = res.decode('utf8')
        return p.returncode,res
    else:
        return subprocess.call(cmd,shell=True)


def spack_exe(spath=None):
    if spath is None:
        to_try = [pjoin("uberenv_libs","spack"), "spack"]
        for p in to_try:
            abs_p = os.path.abspath(p)
            print("[looking for spack directory at: {}]".format(abs_p))
            if os.path.isdir(abs_p):
                print("[FOUND spack directory at: {}]".format(abs_p))
                return os.path.abspath(pjoin(abs_p,"bin","spack"))
        print("[ERROR: failed to find spack directory!]")
        sys.exit(-1)
    else:
        spack_exe = os.path.abspath(spath,"bin","spack")
        if not os.path.isfile(spack_exec):
            print("[ERROR: failed to find spack directory at spath={}]").format(spath)
            sys.exit(-1)
        return spack_exe

def find_pkg(pkg_name):
    r,rout = sexe(spack_exe() + " find -p " + pkg_name,ret_output = True)
    print(rout)
    for l in rout.split("\n"):
        print(l)
        lstrip = l.strip()
        if not lstrip == "" and \
           not lstrip.startswith("==>") and  \
           not lstrip.startswith("--"):
            return {"name": pkg_name, "path": l.split()[-1]}
    print("[ERROR: failed to find package named '{}']".format(pkg_name))
    sys.exit(-1)

def path_cmd(pkg):
    return('export PATH={}:$PATH\n'.format((pjoin(pkg["path"],"bin"))))

def write_env_script(pkgs):
    ofile = open("s_env.sh","w")
    for p in pkgs:
        print("[found {} at {}]".format(p["name"],p["path"]))
        ofile.write("# {}\n".format(p["name"]))
        ofile.write(path_cmd(p))
    print("[created {}]".format(os.path.abspath("s_env.sh")))

def main():
    pkgs = [find_pkg(pkg) for pkg in sys.argv[1:]]
    if len(pkgs) > 0:
        write_env_script(pkgs)
    else:
        print("usage: python gen_spack_env_script.py spack_pkg_1 spack_pkg_2 ...")

if __name__ == "__main__":
    main()
