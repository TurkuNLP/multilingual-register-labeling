#!/bin/sh
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=64G
#SBATCH -p gputest
#SBATCH -t 00:15:00
#SBATCH --gres=gpu:v100:1
#SBATCH --ntasks-per-node=1
#SBATCH --account=Project_2002026
#SBATCH -o /scratch/project_2003075/ve/bert-%j.out
#SBATCH -e /scratch/project_2003075/ve/bert-%j.err

echo "START: $(date)"

module purge
module load tensorflow
source /scratch/project_2002026/multilabel_bert/neuro_classifier/multilabel/VENV3/bin/activate

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OUTPUT_DIR=/scratch/project_2003075/ve/output

mkdir -p "$OUTPUT_DIR"

export EVALSET=dev    # set to "test" for final experiments, otherwise dev


set -euo pipefail

export SCRIPTDIR=/scratch/project_2003075/ve

export WVDIR="$SCRIPTDIR/wordvecs"
export DATADIR=/scratch/project_2003075/ve


export MODELDIR="$SCRIPTDIR/models"
export RESULTDIR="$SCRIPTDIR/results"

# Parameters
EPOCHS=15
MAX_WORDS=100000
WORD_VECS="fi:$WVDIR/wiki.multi.fi.vec" #,"sv:$WVDIR/wiki.multi.sv.vec"

export WORD_VECS="fi:$WVDIR/wiki.multi.fi.vec" #,"sv:$WVDIR/wiki.multi.sv.vec"


mkdir -p "$MODELDIR"
mkdir -p "$RESULTDIR"


# Finnish w/subsets
#total=$(wc -l < "$DATADIR/fi-train.ft")
#for p in `seq 10 10 100`; do
#    lines=$((p*total/100));
#    shuf "$DATADIR/fi-train.ft" | head -n $lines \
#        > "$DATADIR/fi-train-${p}p.ft" || true

#    python3 "$SCRIPTDIR/trainmlcnn.py" \#
#	    --epochs "$EPOCHS" \
#	    --limit "$MAX_WORDS" \#
#	    --word-vectors "$WORD_VECS" \
#	    --input "fi:$DATADIR/fi-train-${p}p.ft" \
#	    --output "$MODELDIR/fi-${p}p.model"

#    python3 "$SCRIPTDIR/testmlcnn.py" \
#	    "$MODELDIR/fi-${p}p.model" \
#	    "fi:$DATADIR/fi-${EVALSET}.ft" \
#	| tee -a "$RESULTDIR/fi-${p}p.txt"
#done


#for p in `seq 10 10 100`; do
#    python3 "$SCRIPTDIR/trainmlcnn.py" \
#	    --epochs "$EPOCHS" \#
#	    --limit "$MAX_WORDS" \
#	    --word-vectors "$WORD_VECS" \
#	    --input "fi:$DATADIR/fi-train-${p}p.ft,en:$DATADIR/en-train.ft" \
#	    --output "$MODELDIR/combo-${p}p.model"

#    python3 "$SCRIPTDIR/testmlcnn.py" \
#	    "$MODELDIR/combo-${p}p.model" \
#	    "fi:$DATADIR/fi-${EVALSET}.ft" \
#	| tee -a "$RESULTDIR/combo-fi-${p}p.txt"
#done

# Finnish
#python3 "$SCRIPTDIR/trainmlcnn.py" \
#	--epochs "$EPOCHS" \
#	--limit "$MAX_WORDS" \
#	--word-vectors "$WORD_VECS" \
#	--input "fi:$DATADIR/fi-train.ft" \
#	--output "$MODELDIR/fi.model"

#python3 "$SCRIPTDIR/testmlcnn.py" \
#	"$MODELDIR/fi.model" \
#	"fi:$DATADIR/fi-${EVALSET}.ft" \
#    | tee -a "$RESULTDIR/fi.txt"


# English
for l in {0.0001,0.0005}; #0.001,0.003,0.005,0.006,0.01}; 
do
for t in {0.4,0.5} #0.6}

do python3 "$SCRIPTDIR/trainml_multilabel_cnn.py" \
    --epochs "$EPOCHS" \
    --limit "$MAX_WORDS" \
    --learning_rate $l \
    --word-vectors "$WORD_VECS" \
    --input "fi:$DATADIR/train.tsv" \
    --output "$MODELDIR/fi.model" \
    --validation "fi:$DATADIR/dev.tsv" \
    --predictions "$DATADIR/predictions.txt" \
    --threshold $t \
    --kernel-size 2
done
done
#        -v "en:$DATADIR/dev.tsv"



#python3 "$SCRIPTDIR/testmlcnn.py" \
#	"$MODELDIR/en.model" \
#	"en:$DATADIR/${EVALSET}.tsv" \
#    | tee -a "$RESULTDIR/en.txt"


# English -> Finnish
#python3 "$SCRIPTDIR/testmlcnn.py" \
#	"$MODELDIR/en.model" \
#	"fi:$DATADIR/fi-${EVALSET}.ft" \
 #   | tee -a "$RESULTDIR/en-fi.txt"


# English+Finnish -> Finnish
#python3 "$SCRIPTDIR/trainmlcnn.py" \
#	--epochs "$EPOCHS" \
#	--limit "$MAX_WORDS" \
#	--word-vectors "$WORD_VECS" \
#	--input "fi:$DATADIR/fi-train.ft,en:$DATADIR/en-train.ft" \
#	--output "$MODELDIR/combo.model"

#python3 "$SCRIPTDIR/testmlcnn.py" \
#	"$MODELDIR/combo.model" \
#	"fi:$DATADIR/fi-${EVALSET}.ft" \
#    | tee -a "$RESULTDIR/combo-fi.txt"
