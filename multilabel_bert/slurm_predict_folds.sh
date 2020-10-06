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

#DATA_SUFFIX="_win100"
DATA_SUFFIX="_fulldoc"
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MODEL_DIR=/scratch/project_2002026/bert/cased_L-12_H-768_A-12
export SOURCE_DIR=/scratch/project_2002026/multilabel_bert/neuro_classifier/multilabel

export DATA_DIR=data/train-10-fold-prep
export OUTPUT_DIR=output

mkdir -p "$OUTPUT_DIR"


srun python3 $SOURCE_DIR/bert_predict.py --test data/dev-processed.jsonl.gz --model $OUTPUT_DIR/model$DATA_SUFFIX.h5 --labels data/labels.json --output_file $OUTPUT_DIR/pred_fulldoc-train_beg-dev.txt --clip_value 1e-4 --seq_len 512 --eval_batch_size 6
#srun python3 $SOURCE_DIR/bert_predict.py --test data/fulldoc_dev-processed.jsonl.gz --model $OUTPUT_DIR/model$DATA_SUFFIX.h5 --labels data/labels.json --output_file $OUTPUT_DIR/pred_fulldoc_dev.txt --clip_value 1e-4 --seq_len 512 --eval_batch_size 6
#srun python3 $SOURCE_DIR/bert_predict.py --test data/fulldoc_test-processed.jsonl.gz --model $OUTPUT_DIR/model$DATA_SUFFIX.h5 --labels data/labels.json --output_file $OUTPUT_DIR/pred_fulldoc_test.txt --clip_value 1e-4 --seq_len 512 --eval_batch_size 6
for f in {8,9,10}; do
  DATA_SUFFIX="fulldoc-fold-$f"
  #srun python3 $SOURCE_DIR/bert_predict.py --test $DATA_DIR/$DATA_SUFFIX-test-processed.jsonl.gz --model $OUTPUT_DIR/model_$DATA_SUFFIX.h5 --labels data/labels.json --output_file $OUTPUT_DIR/pred_$DATA_SUFFIX.txt --clip_value 1e-4 --seq_len 512 --eval_batch_size 6
done

# srun python3 bert_fine_tune_multigpu.py --train train_bert-processed.gz  --dev dev_bert-processed.gz --init_checkpoint $MODEL_DIR/bert_model.ckpt --bert_config $MODEL_DIR/bert_config.json --lr $f --seq_len 512 --epochs 5 --batch_size 6
# srun python3 bert_fine_tune_multigpu.py --train train_bert-processed.gz --dev dev_bert-processed.gz --init_checkpoint $MODEL_DIR/bert_model.ckpt --bert_config $MODEL_DIR/bert_config.json --lr $f --seq_len 512 --epochs 5 --batch_size 6 ; done

#rm -rf "$OUTPUT_DIR"
#mkdir -p "$OUTPUT_DIR"


echo "END: $(date)"
