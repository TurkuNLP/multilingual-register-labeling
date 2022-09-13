mkdir logs

module purge
module load tensorflow/2.2-hvd
python -m venv VENV-tf2.2-transformers3.4
source VENV-tf2.2-transformers3.4/bin/activate
python -m pip install -r requirements.txt
