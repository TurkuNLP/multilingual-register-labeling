# Multilingual modeling of registers

This repository includes the following folders, including annotations and code used in the papers described below. Please cite these papers if you use these resources!

#### A transformers Classifier 
Using Huggingface model repository and TF2/keras, developed by Rönnqvist et al. (2021)

#### EACL21-code 
Code used to develop the register identification systems applied in Rönnqvist et al. (2021)

#### Multilingual register annotations
* FreCORE and SvCORE: register annotations in French and Swedish, developed by Repo et al. (2020) and applied by Rönnqvist et al. (2022)
* Register annotations in eight further languages, developed by Laippala et al. (2022)

#### Text quality annotations
Applied and developed by Laippala et al. (2022) and Salmela (2022).


* Modeling utilities
* (Deprecated: Multilabel BERT (using Keras BERT))
* (Deprecated: Multilabel Transformers (same as above, adapted to Huggingface library; not functional))
* EACL21_SRW: Contains the French FrCORE and Swedish SvCORE data and code introduced in the paper 'Beyond the English Web: Zero-Shot Cross-Lingual and Lightweight Monolingual Classification of Registers'.
* Code for training on multiple corpora available [here](https://github.com/sronnqvist/transformers-classifier) and the trained master model [here](http://dl.turkunlp.org/register/nodalida21/model_xlmrL0.5_en%2Bfi%2Bfr%2Bsv-common-8e-7-100.h5.gz) (as reported in Rönnqvist et al. 2021, please cite as below).

```
@inproceedings{laippala2022-registeroscar,
title={Towards better structured and less noisy Web data: Oscar with Register annotations}
author={Veronika Laippala and Anna Salmela and Samuel R\"onnqvist and Alham Fikri Aji and Li-Hsin Chang and Asma Dhifallah and Larissa Goulart and Henna Kortelainen and Marc P\`amies and Deise Prina Dutra and Valtteri Skantsi and Lintang Sutawika and Sampo Pyysalo}
booktitle={Proceedings of the Eight Workshop on Noisy User-generated Text (W-NUT 2022)}
year={2022}
}
```

```
@inproceedings{ronnqvist2021multilingual,
  title={Multilingual and Zero-Shot is Closing in on Monolingual Web Register Classification},
  author={Samuel R\"onnqvist and Valtteri Skantsi and Miika Oinonen and Veronika Laippala},
  booktitle={Proceedings of the 23rd Nordic Conference on Computational Linguistics (NoDaLiDa’21).},
  year={2021}
}
```
