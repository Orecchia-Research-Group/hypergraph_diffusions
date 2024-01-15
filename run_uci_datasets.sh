#!/bin/bash
#
#SBATCH --mail-user=kameranis@uchicago.edu
#SBATCH --mail-type=ALL
#SBATCH --output=/scratch/kameranis/hypergraph_diffusions/slurm.out
#SBATCH --error=/scratch/kameranis/hypergraph_diffusions/slurm.err
#SBATCH --chdir=/scratch/kameranis/hypergraph_diffusions/
#SBATCH --partition=fast
#SBATCH --job-name=run_uci
# #SBATCH --exclusive
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task 8
#SBATCH --mem=25000
#SBATCH --time=20:00:00
# #SBATCH --array=0-440

graphs=(zoo mushroom covertype45 covertype67 newsgroups)
minimum_samples=(0 20 20 20 20 20)
step=(0 5 20 20 20 20)
maximum_samples=(0 50 200 200 200 200)

#for i in {1..5}
#do
i=$1
echo "python submodular_cut_fns.py -g data/Paper_datasets/${graphs[$i]} --minimum-samples=$((minimum_samples[$i])) --step=$((step[$i])) --maximum-samples=$((maximum_samples[$i])) -T 200 --repeats=50 -f data/Paper_results/uci_runs.csv"
python submodular_cut_fns.py -g data/Paper_datasets/${graphs[$i]} --minimum-samples=$((minimum_samples[$i])) --step=$((step[$i])) --maximum-samples=$((maximum_samples[$i])) -T 200 --repeats=50 -f data/Paper_results/uci_runs_${graphs[$i]}.csv
#done
