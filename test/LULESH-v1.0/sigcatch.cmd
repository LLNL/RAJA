#!/bin/bash
#SBATCH -p pdebug
#SBATCH --signal=USR2@120
#SBATCH -t 5:00

echo -n 'The start time is: ';date
# trap "echo trap" SIGUSR1
# On rzmerl, % sbatch sigcatch.cmd
# % scancel -s USR2  <job_id>     do this repeatedly to simulate faults
srun /g/g19/keasler/RAJA_RICH/raja/test/LULESH-v1.0/lulesh-RAJA-parallel-ft.exe
echo -n 'The completion time is: ';date

