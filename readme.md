#  Mask-guided BERT for Few Shot Text Classification Neroucomputing 2024 paper

## 1. software environment

```
environment.yml
```

## 2.download pre-trained language model (PLM) and dataset

Download bert-base-cased and save in ./save_models: https://huggingface.co/bert-base-cased

AG_news: Already provided in directory https://huggingface.co/datasets/fancyzhx/ag_news

dbpedia14: https://huggingface.co/datasets/dbpedia_14

snippets: Already provided in directory ./data/snippets/

symptoms: Already provided in directory ./data/symptoms/

nicta: Already provided in directory ./data/nicta/

PubMed20k: https://github.com/Franck-Dernoncourt/pubmed-rct


## Few-shot learning
```
python mask_bert.py
```
