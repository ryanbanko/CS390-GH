#!/bin/bash

#SBATCH --job-name=fmri_prep_${SUBJECT}
#SBATCH --output=fmri_prep_%j.out
#SBATCH --error=fmri_prep_%j.err
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --partition=compsci
#SBATCH --mail-user=rb446@duke.edu
#SBATCH --mail-type=END,FAIL


SCRATCH_DIR="/tmp/fmri_prep_$SLURM_JOB_ID"
mkdir -p $SCRATCH_DIR

trap "rm -rf $SCRATCH_DIR" EXIT

echo "Processing subject: sub-${SUBJECT}"

apptainer exec \
  --bind /home/users/rb446/license.txt:/freesurfer/license.txt \
  --bind /home/users/rb446/mri_data:/data \
  /home/users/rb446/fmriprep_latest.sif \
  fmriprep /data /data/output participant \
  --skip-bids-validation \
  --md-only-boilerplate \
  --fs-license-file /freesurfer/license.txt \
  --participant-label=${SUBJECT} \
  --fs-no-reconall \
  --output-spaces MNI152NLin2009cAsym:res-2 \
  --nthreads 16 \
  --stop-on-first-crash \
  -w $SCRATCH_DIR
