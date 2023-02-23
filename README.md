# SamsungCard Customer Feedback Classifier Competition

## Contributor
|박지훈|박인창|이헌득|
|:---:|:---:|:---:|
![image](https://user-images.githubusercontent.com/73874591/213502466-64a5def9-f685-48f4-8255-a289d165f6ff.png)|![image](https://user-images.githubusercontent.com/73874591/213502466-64a5def9-f685-48f4-8255-a289d165f6ff.png)|<img src="https://user-images.githubusercontent.com/97590480/205299457-5292caeb-22eb-49d2-a52e-6e69da593d6f.jpeg">|
|[Github](https://github.com/hundredeuk2)|[Github](https://github.com/inchang0507)|[Github](https://github.com/hundredeuk2)|

## Result
※ 전체 3위 (정확도 기준 1위 - 약 96%)

# 1. Introduction
* RoBERTa, KoBERT, KoELECTRA benchmark
* Framework
1. Preprocessing with KoSpacing, py-hanspell
2. Oversampling to Data Augument
3. Ensemble

# Project Tree
```
DocVQA
├─ configs
│  └─ baseline.yaml
├─ dataloader
│  └─ custom_dataloader.py
├─ model
│  ├─ custom_model_old.py
│  └─ custom_model.py
├─ trainer
│  └─ BaselineTrainer.py
├─ utils
│  ├─ metric.py
│  ├─ seed.py
│  └─ wandb.py
├─ .gitignore
├─ requirements.txt
├─ train.py
└─ inference.py
```
# Reference
* KoELECTRA
* KoBERT
* Klue/RoBERTa
* PyKoSpacing
* py-hanspell
