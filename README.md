## Multilingual modeling of registers

This repository presents resources for the multilingual modeling of web registers, i.e., text varieties such as news, blogs or how-to pages. 

Please cite the below papers if you use these resources!

### Register scheme

We apply the register scheme originally developed for the <a href="https://github.com/TurkuNLP/CORE-corpus">Corpus of Online Registers of English</a> (CORE) by Jesse Egbert, Douglas Biber and Mark Davies.

The scheme is hierarchical with eight main registers: Spoken, Lyrical, Informational Description, Informational Persuasion, Opinion, Narrative, How-to and Interactive Discussion. 

Additionally, the scheme has a technical category Machine-translated / Generated.

The main registers can be further described with sub-register labels, such as News Article, Opinion Blog or Description of a Thing. 

All the register categories are described in detail at https://turkunlp.org/register-annotation-docs/.

### Data

#### Manual multilingual register annotations
* <a href="https://github.com/TurkuNLP/multilingual-register-labeling/tree/master/register-annotations/FrCORE">FreCORE</a> and <a href="https://github.com/TurkuNLP/multilingual-register-labeling/tree/master/register-annotations/SvCORE">SvCORE:</a> register annotations in French and Swedish, developed by Repo et al. (2020) and applied by Rönnqvist et al. (2022)
* <a href="https://github.com/TurkuNLP/CORE-corpus:">CORE corpus:</a> register annotations in English, developed by Biber and Egbert (2018) and applied by Laippala et al. (2022)
* <a href="https://github.com/TurkuNLP/FinCORE_full/releases/tag/v1.0">FinCORE:</a> register annotations in Finnish, developed by Skantsi et al. (2023)
* <a href="https://github.com/TurkuNLP/multilingual-register-labeling/tree/master/register-annotations">Evaluation sets in eight further languages</a>, developed by Laippala et al. (2022)

#### Automatically labeled datasets
* Register annotations for Oscar, available at https://huggingface.co/datasets/TurkuNLP/register_oscar, described in Laippala et al. (2022)

#### Text quality annotations
* Applied and developed by Laippala et al. (2022) and Salmela (2022).

### Code

##### A transformers Classifier 
* Using Huggingface model repository and TF2/keras, developed by Rönnqvist et al. (2021)

##### EACL21-code 
* Code used to develop the register identification systems applied in Rönnqvist et al. (2021)

### Demo
* Demo available at https://live.european-language-grid.eu/catalogue/tool-service/20212

### Funding
We thank the Academy of Finland, Emil Aaltonen Foundation and Fulbright Finland for funding.

### References
```
@article{skantsi_laippala_2023, title={Analyzing the unrestricted web: The Finnish corpus of online registers}, 
DOI={10.1017/S0332586523000021}, journal={Nordic Journal of Linguistics}, 
publisher={Cambridge University Press}, 
author={Skantsi, Valtteri and Laippala, Veronika}, year={2023}, pages={1–31}}
```
```
@inproceedings{laippala2022-registeroscar,
title={Towards better structured and less noisy Web data: Oscar with Register annotations}
author={Veronika Laippala and Anna Salmela and Samuel R\"onnqvist and Alham Fikri Aji 
and Li-Hsin Chang and Asma Dhifallah and Larissa Goulart and Henna Kortelainen and Marc P\`amies and Deise Prina Dutra and Valtteri Skantsi 
and Lintang Sutawika and Sampo Pyysalo}
booktitle={Proceedings of the Eight Workshop on Noisy User-generated Text (W-NUT 2022)}
year={2022}
}

```
```

@article{laippala2022corecorpus,
author={Veronika Laippala and Samuel R{\"o}nnqvist and Miika Oinonen and Aki-Juhani Kyr{\"o}l{\"a}inen 
and Anna Salmela and Douglas Biber and Jesse Egbert and Sampo Pyysalo},
title={Register identification from the unrestricted open Web using the Corpus of Online Registers of English},
year=2022,
journal={Language Resources and Evaluation},
doi={ https://doi.org/10.1007/s10579-022-09624-1}
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
```
@inproceedings{repo-etal-2021-beyond,
    title = "Beyond the {E}nglish Web: Zero-Shot Cross-Lingual and Lightweight Monolingual Classification of Registers",
    author = {Repo, Liina  and Skantsi, Valtteri and R{\"o}nnqvist, Samuel and Hellstr{\"o}m, Saara and Oinonen, Miika 
    and Salmela, Anna and Biber, Douglas and Egbert, Jesse and Pyysalo, Sampo and Laippala, Veronika},
    booktitle = "Proceedings of the 16th Conference of the European Chapter of the Association for Computational Linguistics: Student Research Workshop",
    month = apr,
    year = "2021",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2021.eacl-srw.24",
    doi = "10.18653/v1/2021.eacl-srw.24",
    pages = "183--191",
}
```
