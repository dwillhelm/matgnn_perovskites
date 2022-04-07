#!/bin/bash

echo "Loading Modules"

module purge
## load bash profile
. ~/.bash_profile

## load git
module load GCCcore/10.3.0 
module load git/2.32.0-nodocs

## load CUDA and other modules
module load CUDA/11.3.1

## end cmmds 
source activate graphDL
module list



# module load GCC/10.2.0
# module load PyTorch/1.7.1
# module load OpenMPI/4.0.5
