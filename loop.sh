#!/bin/bash
subjects=("30" "31" "32" "33" "34" "35" "36" "37" "38" "39")

for sub in "${subjects[@]}"; do
  echo "Submitting job for subject: sub-${sub}"
  sbatch --export=SUBJECT=${sub} final.sbatch
done
