## Multilingual modeling of registers

This repository presents resources for the multilingual modeling of web registers, i.e., text varieties such as news, blogs or how-to pages. The register taxonomy we use is described at https://turkunlp.org/register-annotation-docs/.

Please cite the below papers if you use these resources!

### Data

#### Manual multilingual register annotations
* FreCORE and SvCORE: register annotations in French and Swedish, developed by Repo et al. (2020) and applied by Rönnqvist et al. (2022)
* CORE: register annotations in English, developed by Biber and Egbert (2018) and applied by Laippala et al. (2022)
* FinCORE: register annotations in Finnish, developed by Skantsi et al. (2023)
* Evaluation sets in eight further languages, developed by Laippala et al. (2022)

#### Automatically labeled datasets
* Register annotations for Oscar, available at https://huggingface.co/datasets/TurkuNLP/register_oscar, described in Laippala et al. (2022)

#### Text quality annotations
Applied and developed by Laippala et al. (2022) and Salmela (2022).

### Code

##### A transformers Classifier 
Using Huggingface model repository and TF2/keras, developed by Rönnqvist et al. (2021)

##### EACL21-code 
Code used to develop the register identification systems applied in Rönnqvist et al. (2021)

### Demo
* Demo available at https://live.european-language-grid.eu/catalogue/tool-service/20212

#### References:
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
