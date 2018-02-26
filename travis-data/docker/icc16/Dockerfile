FROM rajaorg/compiler:ubuntu-clang-base

LABEL maintainer="Tom Scogland <scogland1@llnl.gov>"

RUN mkdir /home/raja/intel
ADD silent.cfg /home/raja/intel/silent.cfg

ENV NAME=parallel_studio_xe_2016_update3_online.sh
# apparently cpio is required or the install fails as though it ran out of disk... yay...
RUN cd intel \
 && sudo apt-get install cpio \
 && wget -q -O ${NAME} 'http://registrationcenter-download.intel.com/akdlm/irc_nas/9061/parallel_studio_xe_2016_update3_online.sh' \
 && chmod +x ${NAME} \
 && mkdir downloads tmp \
 && sudo ./${NAME} -s silent.cfg -D downloads -t tmp \
 && sudo rm -rf /home/raja/intel /opt/intel/ism/db /opt/intel/licenses
# last line removes the license and serial references,
# this is REQUIRED, and since the installer likes to echo
# the serial into log files, nuking everything this way
# is basically the way to go

# usage requires a directory with a valid license file to
# be volume mounted to /opt/intel/licenses
