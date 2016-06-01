#! /usr/bin/env python

import os, sys, argparse, glob, time
import fnmatch

from subprocess import call

#if using pypar run this command as mpirun -np 16 --hostfile hostfile.txt -mca btl tcp,sm,self python  script.py

class ClangFormat :
  cmd = "clang-format"  # assumes you've done a use clang-3.8.0 or equivalent which has clang-format built/installed
  pypar_avail = False
  pypar = None
    
  def __init__(self):
    try:
      #import pypar as pypar
      self.pypar_avail = True
      #self.pypar = pypar
    except:
      pass

  def getScriptPath(self):
    return os.path.dirname(os.path.realpath(sys.argv[0]))

  def process(self,input_directory,cmd_prefix):
    files = []
    self.pypar_avail = False # Turn off mpi processing for now
    self.cmd = cmd_prefix + self.cmd
    print self.cmd + " is processing " + input_directory
    for root, dirs, ff in os.walk(input_directory):
      for name in ff:
        #print name
        if name.endswith(('.hxx','.cxx','.cpp','.cc','.c','.h')):
          files.append(os.path.join(root,name))

    print "Num to Process = " + str(len(files))
    numToProcess = len(files)
    if numToProcess > 0:
      if self.pypar_avail == True:
        proc = self.pypar.size()                                # Number of processes as specified by mpirun
        myid = self.pypar.rank()                                # Id of this process (myid in [0, proc-1]) 
        for ii in range(0,fpas):
          if ii % proc == myid:
            fname = files[ii]
            print "node " + str(myid) + " is processing " + fname
            call([self.cmd,"-i",fname])
            self.pypar.barrier()
            self.pypar.finalize() 
      else:
        for ii in range(0,numToProcess):
          fname = files[ii]
          print "Processing " + fname
          call([self.cmd,"-i",fname])

def main():
  parser = argparse.ArgumentParser(description='clang-format directory tree [.hxx,.cxx] : default search path is scriptdirectory/..')
  parser.add_argument('--input_directory',default="",help='Input Directory to start recursive auto-format')
  parser.add_argument('--cmdpath',default="",help='Path to clang-format')
  args = parser.parse_args()
  cf = ClangFormat()
  input_directory = args.input_directory
  if input_directory == "":
    input_directory = cf.getScriptPath()
    input_directory += '/..'
  cmd_prefix = args.cmdpath
  if cmd_prefix != "":
    cmd_prefix += '/'
  cf.process(input_directory,cmd_prefix)

if __name__ == '__main__':
  main()

