#!/bin/bash
#----------------------------------------------------
# Distributed Training Job Script
# for TACC Lonestar6 AMD Milan nodes
#
# *** MPI Training Job in Development Queue ***
# 
#----------------------------------------------------

#SBATCH -J train_model        # Job name
#SBATCH -o train_model.o%j    # Name of stdout output file
#SBATCH -e train_model.e%j    # Name of stderr error file
#SBATCH -p normal              # Queue (partition) name
#SBATCH -N 4                  # Total # of nodes (4 nodes = 512 cores)
#SBATCH -n 4                  # Total # of MPI tasks (1 per node)
#SBATCH -t 02:00:00           # Run time (hh:mm:ss)
#SBATCH --mail-type=all       # Send email at begin and end of job
#SBATCH -A <YOUR_PROJECT_ID>  # Project/Allocation name (REPLACE THIS)

# Any other commands must follow all #SBATCH directives...
module list
pwd
date

# Load required modules
module load python3

# Create a directory on each node's /tmp space
WORKDIR=/tmp/${USER}_${SLURM_JOB_ID}
mkdir -p $WORKDIR

# Set the percentage of data to process per node (25% per node when using 4 nodes)
export DATA_PERCENTAGE=25

# Get the node ID (0-based)
NODE_ID=$SLURM_PROCID

# Calculate the start percentage for this node
START_PERCENT=$(($NODE_ID * $DATA_PERCENTAGE))

# Launch distributed training
ibrun -np 1 python3 train_node.py -d $SCRATCH/training_data -m $SCRATCH/model_node${NODE_ID} -v -p $START_PERCENT -t $DATA_PERCENTAGE

# Wait for all processes to complete
wait

# If this is the first node (rank 0), combine the models
if [ "$NODE_ID" -eq 0 ]; then
  echo "Combining models from all nodes..."
  python3 combine_models.py -m $SCRATCH/model_node0,$SCRATCH/model_node1,$SCRATCH/model_node2,$SCRATCH/model_node3 -o $SCRATCH/model
fi

# Clean up
rm -rf $WORKDIR
