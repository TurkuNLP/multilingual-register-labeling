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

module purge
module load tensorflow/2.0.0
source VENV-tf2.0-transformers3.4/bin/activate



export PYTHONPATH=/scratch/project_2002026/samuel/multilingual-register-labeling/multilabel_transformers/VENV-tf2.0-transformers3.4/lib/python3.7/site-packages:$PYTHONPATH
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
#export MODEL_DIR=/scratch/project_2002026/bert/cased_L-12_H-768_A-12
export MODEL_DIR=/scratch/project_2002026/bert/cased_L-24_H-1024_A-16

export SOURCE_DIR=/scratch/project_2002026/multilabel_bert/neuro_classifier/multilabel

export DATA_DIR=data
export OUTPUT_DIR=output

mkdir -p "$OUTPUT_DIR"

#1-2
LR=7e-6
#3e-5
EPOCHS=4  #3-5
BS=7 #7
#6
COMMENT="bert-large,cls,amsgrad,warmup0.1,devBeg"
DATA_SUFFIX=""

#echo -e "$SLURM_JOBID\t$LR\t$EPOCHS\t$BS\t$DATA_SUFFIX\t$COMMENT" >> experiments.log



for f in 1 2 3;
do srun python bert_fine_tune_multigpu.py \
  --train $DATA_DIR/train$DATA_SUFFIX-processed.jsonl.gz \
  --dev $DATA_DIR/dev-processed.jsonl.gz \
  --init_checkpoint $MODEL_DIR/bert_model.ckpt \
  --bert_config $MODEL_DIR/bert_config.json \
  --lr $LR \
  --seq_len 512 \
  --epochs $EPOCHS \
  --batch_size $BS \
  --output_file $OUTPUT_DIR/model$DATA_SUFFIX-$SLURM_JOBID-$f.h5
done;
#  --load_model $OUTPUT_DIR/model_nblocks3-ep10-2.h5 \

#--load_model $OUTPUT_DIR/model.h5 \
#rm -rf "$OUTPUT_DIR"
#mkdir -p "$OUTPUT_DIR"


echo "END: $(date)"
