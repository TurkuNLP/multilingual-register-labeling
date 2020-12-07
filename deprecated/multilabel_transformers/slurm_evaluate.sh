#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=64G
#SBATCH -p gputest
#SBATCH -t 00:15:00
#SBATCH --gres=gpu:v100:1
#SBATCH --ntasks-per-node=1
#SBATCH --account=Project_2002026
#SBATCH -o logs/%j.out
#SBATCH -e logs/%j.err

echo "START: $(date)"

rm logs/current.err
rm logs/current.out
ln -s $SLURM_JOBID.err logs/current.err
ln -s $SLURM_JOBID.out logs/current.out

#source /scratch/project_2002026/multilabel_bert/neuro_classifier/multilabel/VENV2/bin/activate
#python
#import keras_bert
module purge
module load tensorflow
source /scratch/project_2002026/multilabel_bert/neuro_classifier/multilabel/VENV3/bin/activate


#module load keras
#module load keras_bert
#from keras import bert

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MODEL_DIR=/scratch/project_2002026/bert/cased_L-12_H-768_A-12

export SOURCE_DIR=/scratch/project_2002026/multilabel_bert/neuro_classifier/multilabel

export DATA_DIR=data
export OUTPUT_DIR=output

mkdir -p "$OUTPUT_DIR"

LR=0.0
EPOCHS=1
BS=6
#COMMENT="win20"
MODEL_SUFFIX="_nblocks3-ep10-2"
#MODEL_SUFFIX=""

for f in 100;
do srun python bert_fine_tune_multigpu.py \
  --train $DATA_DIR/train_dummy-processed.jsonl.gz \
  --dev $DATA_DIR/dev_nblocks3-processed.jsonl.gz \
  --load_model $OUTPUT_DIR/model$MODEL_SUFFIX.h5 \
  --lr $LR \
  --seq_len 512 \
  --epochs $EPOCHS \
  --batch_size $BS \
  --output_file /dev/null
done;

#rm -rf "$OUTPUT_DIR"
#mkdir -p "$OUTPUT_DIR"

#echo -e "$SLURM_JOBID\t$LR\t$EPOCHS\t$BS\t$DATA_SUFFIX\t$COMMENT" >> experiments.log

echo "END: $(date)"
