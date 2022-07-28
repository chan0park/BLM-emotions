# BLM-emotions
Official code and data of the paper "An Analysis of Emotions and the Prominence of Positivity in #BlackLivesMatter Tweets" (PNAS 2022)

We release 1) the BERT model pre-trained with our BLBM tweets with a masked language modeling  objective and 2) emotion recognition models trained with our human-annotated data. You can find them from this [gdrive folder](https://drive.google.com/drive/folders/1OnmZsWJwAJknsf1xGwRWGAW1vPn6wemG?usp=sharing)

However, we do not make the raw data freely available to preserve anonymity and privacy as much as possible. Still, we will make tweet ids and annotated data available only for academic research upon request. Please fill out [this form](https://forms.gle/upc8M5eQH5VnQs1C8) to request the data.

# Prerequisite
- change `path_root` in `src_model/configs.py`
- install the following python packages: torch, tensorboard, pandas, transformers, emoji, wordsegment, scikit-learn

# How can I train new emotion recognition models?
1. Prepare GoEmotions and HurricaneEmo dataset
```
bash scripts/download_emotion_datasets.sh
python src_model/preprocess_emotion_data.py
```
Executing the commands above will create a processed HurricaneEmo and GoEmotions  dataset under `data/processed-emotions`, which can be used to train emotion recognition models.
The two data sets use different emotion categories, but here we map both data sets into Ekman basic emotions using the mappings we defined in `data/emo-mapping`.

2. Train emotion detection models
```
bash scripts/train_binary_model.sh $target_emotion $bert_model
```

- **$target_emotion**: one of six Ekman emotions (disgust, fear, anger, sadness, surprise, joy)
- **$bert_model**: {*blm*, *none*}. *blm* indicates the bert-base-uncased model from HuggingFace *transformers* further pre-trained with our BLM tweets.  If you do not provide anything to the bert_model argument (i.e., *none*) the code will use the base pre-trained bert model (bert-base-uncased).

# How can I generate emotion labels given input texts?
```
bash scripts/generate_binary_emotion_labels.sh $target_emotion $trained_model_path $path_input_txt_file $path_output
```
- **$target_emotion**: {disgust, fear, anger, sadness, surprise, joy}
- **$trained_model_path**: path to a trained model you want to generate labels from, e.g., `blm_joy.pt`.
- **$path_input_txt_file**: .txt file separated by lines
- **$path_output**: path to save the output and their predictions
