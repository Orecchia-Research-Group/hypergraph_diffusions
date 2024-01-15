#!/bin/bash
#
#SBATCH --mail-user=kameranis@uchicago.edu
#SBATCH --mail-type=ALL
#SBATCH --output=/scratch/kameranis/hypergraph_diffusion/slurm.out
#SBATCH --error=/scratch/kameranis/hypergraph_diffusion/slurm.err
#SBATCH --chdir=/scratch/kameranis/hypergraph_diffusion/
#SBATCH --partition=fast
#SBATCH --job-name=run_uci
# #SBATCH --exclusive
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task 1
#SBATCH --mem=25000
#SBATCH --time=20:00:00
# #SBATCH --array=0-440

graphs=(zoo mushroom covertype45 covertype67 newsgroups)
minimum_samples=(20 20 20 20 20)
step=(5 20 20 20 20)
maximum_samples=(50 200 200 200 200)

for i in {0..4}
do
  python submodular_cut_fns.py -g data/Paper_datasets/$((graphs[$i])) --minimum-samples=$((minimum_samples[$i])) --step=$((step[$i])) --maximum-samples=$((maximum_samples[$i])) -T 200 --repeats=50 -f Paper_results/uci_runs.csv
done