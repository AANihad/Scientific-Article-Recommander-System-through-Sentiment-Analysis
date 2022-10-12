# Scientific Article Recommander System through Sentiment Analysis

A scientific article recommander system based on citation sentiment analysis using [Huggingface](https://huggingface.co) finetuned models on [Athar](https://cl.awaisathar.com/citation-sentiment-corpus/) (2011) Citation Sentiment Corpus and [Streamlit](https://github.com/streamlit/streamlit) as deployement method.
The fine-tuned transformers used are:

- SiEBERT Sentiment RoBERTa Large English
- Allenai Scibert Scivocab Uncased
- XLNet Large & Base Cased
- BERT Large & BERT Base Uncased

![Streamlit interface](Mémoire/ui.png?raw=true)

The results from the best models are displayed in the following table:
| Transformer | Accuracy | AUROC P | AUROC O | AUROC N | Micro F score | Macro F-score |
| ----------- | -------- | ------- | ------- | ------- | ------------- | ------------- |
| XLNet Large | 0,72 | 0,743 | 0,827 | 0,965 | 0,55 | 0,4 |
| XLNet Base | 0,818 | 0,829 | 0,852 | 0,948 | 0,73 | 0,51 |
| SIEBERT | 0,826 | 0,867 | 0,851 | 0,927 | 0,74 | 0,52 |
| SCIBERT | 0,933 | 0,98 | 0,959 | 0,997 | 0,9 | 0,71 |
| BERT Large | 0,842 | 0,861 | 0,877 | 0,963 | 0,76 | 0,52 |
| BERT Base | 0,889 | 0,932 | 0,928 | 0,984 | 0,83 | 0,61 |

---

Le but de ce projet est d’implémenter un système de recommandation pour les articles scientifiques basés sur l’analyse des sentiments des citations en utilisant une méthode d’apprentissage automatique supervisée basée sur l’apprentissage profond. Notre but n’est pas d’implémenter ces réseaux du zéro mais d’utiliser les modèles état de l’art dans le traitement du langage naturel et l’analyse des sentiments qui sont les Transformers et les modèles pré-entraînés.
L’approche proposée est l’entraînement de plusieurs modèles d’analyse de sentiments et en choisir les meilleurs. Dans le but d’avoir de meilleurs résultats on utilise des modèles pré-entraînés de Transformers, principalement depuis la plat-forme Huggingface et on fait l’apprentissage sur nos propres données.
Les modèles pré-entraînés utilisés sont :

- SiEBERT Sentiment RoBERTa Large English
- Allenai Scibert Scivocab Uncased
- XLNet Large et Base Cased
- BERT Large et BERT Base Uncased
  Ce système de recommandation est un système basé contenu textuel. L’éxtraction des citations a été faite par le biais d’expressions régulières.
