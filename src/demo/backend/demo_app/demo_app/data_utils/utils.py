import spacy
import torch
import pandas as pd
import os
import time

from src.RNN_model import FCModel
from sklearn.linear_model import LogisticRegression as LR
from transformers import BertTokenizer
from src.transformer_model import PersonalityClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

PRE_TRAINED_MODEL_NAME = 'bert-base-cased'


def get_average_word_embeddings(text):
    t = time.time()
    nlp = spacy.load("en_core_web_lg")
    tokens = nlp(text)
    embedding_sum = 0
    counter = 0
    for token in tokens:
        embedding_sum += token.vector
        counter += 1
    print('Average embedding time:', time.time() - t)
    return embedding_sum / counter


def predict_with_fc(average_word_embeddings, trait):
    param_file_path = f'demo_app/data_utils/fc_params/{str.lower(trait)}_model.pt'
    model = FCModel(300, 150)
    model.load_state_dict(torch.load(param_file_path))

    outputs = model.forward(torch.from_numpy(average_word_embeddings))
    logit = outputs  # torch.sigmoid(outputs)
    return logit.item()


def train_lr(df_train, vectorizer, personality, lr_kwargs={"solver": "liblinear"}): #"max_iter": 1000, "solver": "lbfgs"}):
    """
    Receives the train set `df_train` as pd.DataFrame and extracts lemma n-grams
    with their correspoding labels (news type).
    The text is vectorized and used to train a logistic regression with
    training arguments passed as `lr_kwargs`.
    Returns the fitted model.
    """
    vectorizer.set_params(max_df=df_train.shape[0])
    X=vectorizer.fit_transform(df_train["TEXT"])
    model=LR(**lr_kwargs)
    model.fit(X, df_train[[personality]])
    return model


def predict_with_lr(text, trait, df_train, vectorizer):
    # essays_df = pd.read_csv("../../../../data/essays_expanded.csv")
    # vectorizer = TfidfVectorizer()
    # df_train, df_test = train_test_split(
    #     essays_df[["TEXT", "cEXT", "cNEU", "cAGR", "cCON", "cOPN"]], test_size=0.2, random_state=42
    # )

    model = train_lr(df_train, vectorizer, trait)
    vectorized_text = vectorizer.transform([text])
    prediction = model.predict_proba(vectorized_text)

    return prediction[0][1]


def predict_with_transformer(text, trait):
    tokenizer = BertTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME)
    tokens = tokenizer.tokenize(text)
    token_ids = tokenizer.convert_tokens_to_ids(tokens)
    encoding = tokenizer.encode_plus(
        text,
        max_length=128,
        add_special_tokens=True,  # Add '[CLS]' and '[SEP]'
        return_token_type_ids=False,
        pad_to_max_length=True,
        return_attention_mask=True,
        return_tensors='pt',  # Return PyTorch tensors
    )

    param_file_path = f'demo_app/data_utils/transformer_params/model_{str.lower(trait)}.pt'
    model = PersonalityClassifier(2)
    device = torch.device('cpu')
    model.load_state_dict(torch.load(param_file_path, map_location=device))

    result = model.forward(
        input_ids=encoding['input_ids'].flatten(),
        attention_mask=encoding['attention_mask'].flatten()
    )

    return result



