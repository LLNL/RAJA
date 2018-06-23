FROM rajaorg/compiler:ubuntu-clang-base

LABEL maintainer="Tom Scogland <scogland1@llnl.gov>"

RUN mkdir /home/raja/intel
ADD silent.cfg /home/raja/intel/silent.cfg
ADD ./parallel_studio_xe_2018_update3_professional_edition /home/raja/intel/parallel_studio

# apparently cpio is required or the install fails as though it ran out of disk... yay...
RUN sudo apt-get install cpio
RUN cd /home/raja/intel \
 && mkdir downloads tmp \
 && sudo /home/raja/intel/parallel_studio/install.sh -s /home/raja/intel/silent.cfg -D /home/raja/intel/downloads -t /home/raja/intel/tmp \
 && sudo rm -rf /home/raja/intel /opt/intel/ism/db /opt/intel/licenses

# last line removes the license and serial references,
# this is REQUIRED, and since the installer likes to echo
# the serial into log files, nuking everything this way
# is basically the way to go

# usage requires a directory with a valid license file to
# be volume mounted to /opt/intel/licenses

