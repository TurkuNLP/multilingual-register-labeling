# Multilingual, multilabel modeling of registers

This repository includes the following code and data:
* Transformers Classifier (using Huggingface model repository and TF2/keras)
* CNN
* Modeling utilities
* (Deprecated: Multilabel BERT (using Keras BERT))
* (Deprecated: Multilabel Transformers (same as above, adapted to Huggingface library; not functional))
* EACL21_SRW: Contains the French FrCORE and Swedish SvCORE data and code introduced in the paper 'Beyond the English Web: Zero-Shot Cross-Lingual and Lightweight Monolingual Classification of Registers'.
* Code for training on multiple corpora available [here](https://github.com/sronnqvist/transformers-classifier) and the trained master model [here](http://dl.turkunlp.org/register/nodalida21/model_xlmrL0.5_en%2Bfi%2Bfr%2Bsv-common-8e-7-100.h5.gz) (as reported in Rönnqvist et al. 2021, please cite as below).

```
@inproceedings{ronnqvist2021multilingual,
  title={Multilingual and Zero-Shot is Closing in on Monolingual Web Register Classification},
  author={Samuel R\"onnqvist and Valtteri Skantsi and Miika Oinonen and Veronika Laippala},
  booktitle={Proceedings of the 23rd Nordic Conference on Computational Linguistics (NoDaLiDa’21).},
  year={2021}
}
```
