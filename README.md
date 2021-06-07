# Social-Media-Fake-News-Detection

## INTRODUCTION
Fake news and hoaxes have been there since before the advent of the Internet. The widely accepted definition of Internet fake news is: fictitious articles deliberately fabricated to deceive readers”. Social media and news outlets publish fake news to increase readership or as part of psychological warfare.

The purpose of the work is to come up with a solution that can be utilized by users to detect and filter out sites containing false and misleading information. We use carefully selected features to accurately identify fake posts. Our project aims at classifying the given news articles as fake or true based on the content using few shot learning models with less data. To begin with, we have used pre-trained BERT models as our baseline

## PREPROCESSING AND FEATURE ENGINEERING
### DATASET:
**Smaller Dataset:**
https://drive.google.com/file/d/1d10zW73C_Vd5JF5PKBXw9vmjV5MSS6C2/view?usp=sharing

This is the first dataset containing id & content of fake news. We have trained our model for 322
samples.

**Larger Dataset :**
https://drive.google.com/file/d/1WnoSO6U0RSxxgFlG5Sijnq6C1j2GsKQu/view?usp=sharing

Dataset is loaded and preprocessed as follows:

● The dataset is loaded using pandas library as a dataframe.

● train_test _split package is imported from sklearn library and the data is split into training
and test dataframes.

● Data encoding is so as to feed to the BERT (base) layer.

● To encode the data, firstly every text/row in the dataframe is split into tokens, masks and
segments.

## MODEL DESCRIPTION
### BERT Model:
BERT is one of the best easily available and popular language models. Here we have used the base architecture of BERT and there are a total of 12 layers and each of these layer outputs can be used as word embedding.

We have implemented the BERT model by using custom helper functions like bert_encode() and build_model(). The custom encode function splits each sentence into words(tokens) and a [CLS] token is inserted at the beginning of the sentence and [SEP] token is inserted at the end of each sentence. These split tokens are converted to ids using convert_tokens_to_ids() method imported from the tokenization library. The output of this function returns tokens, masks and
segments. The length of the segment depends on the max_len parameter that is set to 160 in our case.

Once the data is encoded , build_model() function is called and trained for ~500 samples. We decreased the labelled data for training and evaluated the performance of the BERT model.

### Fine Tuned BERT Model:
In this, instead of relying upon pre-trained BERT, we have fine-tuned the BERT model according to the few-shot setting and dataset. We have trained the following models on 500 samples of data.

**Pre-trained BERT + BiLSTM:**
Here, we applied Bi-directional LSTM on top of the pre-trained BERT and modified the word embedding in such a way that the output of the LSTM layer captures relevant information from the BERT embedding layer for classifying the tex.

**Pre-trained BERT + BiGRU:**
Here the implementation is similar to the above method. Howere, instead of BiLSTM we have used BiGRU.
