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

export DATA_DIR=/scratch/project_2002026/multilabel_bert/neuro_classifier/multilabel/CORE-final
export OUTPUT_DIR=data/train-10-fold-prep
#export OUTPUT_DIR=/users/repoliin/test

mkdir -p "$OUTPUT_DIR"
POS="0"
export DATA_SUFFIX=""
#cp -n $DATA_DIR/test.tsv $OUTPUT_DIR/test$DATA_SUFFIX.tsv
#cp -n $DATA_DIR/dev.tsv $OUTPUT_DIR/dev$DATA_SUFFIX.tsv
#cp -n $DATA_DIR/train.tsv $OUTPUT_DIR/train$DATA_SUFFIX.tsv

for f in {9,10}; do
  ## Prepare data for BERT fine-tuning
  export DATA_PREFIX="fold-$f"
  #srun python preprocess_data.py -i $OUTPUT_DIR/$DATA_PREFIX-test.tsv -v $MODEL_DIR/vocab.txt -t eng -l $OUTPUT_DIR/labels.json -L data/labels.json -w $POS
  #srun python preprocess_data.py -i $OUTPUT_DIR/$DATA_PREFIX-train.tsv -v $MODEL_DIR/vocab.txt -t eng -L data/labels.json -w $POS
  ## Prepare data for BERT prediction over full documents
  export DATA_PREFIX="fulldoc-fold-$f"
  #srun python preprocess_data.py -i $OUTPUT_DIR/$DATA_PREFIX-test.tsv -v $MODEL_DIR/vocab.txt -t eng -l $OUTPUT_DIR/labels.json -L data/labels.json --all_blocks True
  srun python preprocess_data.py -i $OUTPUT_DIR/$DATA_PREFIX-train.tsv -v $MODEL_DIR/vocab.txt -t eng -l $OUTPUT_DIR/labels.json -L data/labels.json --all_blocks True
done
#srun python preprocess_data.py -i $OUTPUT_DIR/fold-3-train.tsv -v $MODEL_DIR/vocab.txt -t eng -L data/labels.json -w $POS

ln -s test.tsv data/fulldoc_test.tsv
#srun python preprocess_data.py -i data/fulldoc_test.tsv -v $MODEL_DIR/vocab.txt -t eng -L data/labels.json --all_blocks True
ln -s dev.tsv data/fulldoc_dev.tsv
#srun python preprocess_data.py -i data/fulldoc_dev.tsv -v $MODEL_DIR/vocab.txt -t eng -L data/labels.json --all_blocks True

NLABELS=56
# Fix example counts in files before predicting
#ls data/train-10-fold-prep/fulldoc-fold-*-test-processed.jsonl.gz|while read FILE; do NLINES=$(zcat $FILE|tail -n +2|grep -c ""); echo $FILE $NLINES; echo "[$NLINES, $NLABELS]" > _tmp_; zcat $FILE|tail -n +2>>_tmp_;cat _tmp_|gzip -9> $FILE;done
#ls data/train-10-fold-prep/fulldoc-fold-*-train-processed.jsonl.gz|while read FILE; do NLINES=$(zcat $FILE|tail -n +2|grep -c ""); echo $FILE $NLINES; echo "[$NLINES, $NLABELS]" > _tmp_; zcat $FILE|tail -n +2>>_tmp_;cat _tmp_|gzip -9> $FILE;done
#ls data/fulldoc*jsonl.gz|while read FILE; do NLINES=$(zcat $FILE|tail -n +2|grep -c ""); echo $FILE $NLINES; echo "[$NLINES, $NLABELS]" > _tmp_; zcat $FILE|tail -n +2>>_tmp_;cat _tmp_|gzip -9> $FILE;done
#rm _tmp_

#srun python3 bert_fine_tune_multigpu.py --train CORE-final/train-processed.gz --dev CORE-final/dev-processed.gz --init_checkpoint $MODEL_DIR/bert_model.ckpt --bert_config $MODEL_DIR/bert_config.json --lr 5e-5 --seq_len 512 --epochs 1 --batch_size 6

rm -f labels.json
#rm -rf "$OUTPUT_DIR"
#mkdir -p "$OUTPUT_DIR"


echo "END: $(date)"
