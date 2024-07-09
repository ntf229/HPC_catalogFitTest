#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=160:00:00
#SBATCH --mem=30GB
#SBATCH --job-name=flexFit
#SBATCH --mail-type=END
#SBATCH --output=slurm_out/slurm_%x.out
#SBATCH --mail-user=ntf229@nyu.edu
#SBATCH --array=31

module purge

# Final sample (65)
galaxies=( 'g1.88e10' 'g1.89e10' 'g1.90e10' 'g2.34e10' 'g2.63e10' 'g2.64e10' 'g2.80e10' 'g2.83e10'
                'g2.94e10' 'g3.44e10' 'g3.67e10' 'g3.93e10' 'g4.27e10' 'g4.48e10' 'g4.86e10' 'g4.94e10'
                'g4.99e10' 'g5.05e10' 'g6.12e10' 'g6.37e10' 'g6.77e10' 'g6.91e10' 'g7.12e10' 'g8.89e10' 'g9.59e10'
                'g1.05e11' 'g1.08e11' 'g1.37e11' 'g1.52e11' 'g1.57e11' 'g1.59e11' 'g1.64e11' 'g2.04e11' 'g2.19e11'
                'g2.39e11' 'g2.41e11' 'g2.42e11' 'g2.54e11' 'g3.06e11' 'g3.21e11' 'g3.23e11' 'g3.49e11' 'g3.55e11'
                'g3.59e11' 'g3.71e11' 'g4.90e11' 'g5.02e11' 'g5.31e11' 'g5.36e11' 'g5.38e11' 'g5.46e11' 'g5.55e11'
                'g6.96e11' 'g7.08e11' 'g7.44e11' 'g7.55e11' 'g7.66e11' 'g8.06e11' 'g8.13e11' 'g8.26e11' 'g8.28e11'
                'g1.12e12' 'g1.77e12' 'g1.92e12' 'g2.79e12' )

cd /home/ntf229/containers

singularity exec --overlay overlay-15GB-500K.ext3:ro \
	    /scratch/work/public/singularity/cuda11.0-cudnn8-devel-ubuntu18.04.sif \
/bin/bash -c "source /ext3/env.sh; 
python /home/ntf229/catalogFitTest/bin/flexFit.py \
--method=SFH --dust=True --dust_type=power_law --fitType=DustPedia \
--sampling=dynesty --nwalkers=1024 --niter=10000 --galaxy=${galaxies[$SLURM_ARRAY_TASK_ID]}"



