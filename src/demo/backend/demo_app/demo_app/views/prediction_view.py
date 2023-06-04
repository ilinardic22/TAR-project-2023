import json
import time
import pandas as pd

from django.http import HttpResponse
from django.views import View
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from src.demo.backend.demo_app.demo_app.data_utils.utils import predict_with_fc, predict_with_lr, get_average_word_embeddings, train_lr


class PredictionView(View):
    def post(self, request):
        text = json.loads(request.body).get('text', '')
        if text == '':
            return HttpResponse('')

        personality_traits = ['AGR', 'CON', 'EXT', 'OPN', 'NEU']
        response = {}

        # AGR, CON, EXT, OPN from FC
        # average_word_embeddings = get_average_word_embeddings(text)
        # t = time.time()
        # for trait in personality_traits[0:4]:
        #     # print(trait)
        #     logit = predict_with_fc(average_word_embeddings, trait)
        #     response[trait] = logit
        # print('FC prediction time:', time.time() - t)

        essays_df = pd.read_csv("../../../../data/essays_expanded.csv")
        vectorizer = TfidfVectorizer()
        df_train, df_test = train_test_split(
            essays_df[["TEXT", "cEXT", "cNEU", "cAGR", "cCON", "cOPN"]], test_size=0.2, random_state=42
        )

        # NEU from LR
        t = time.time()
        for trait in personality_traits:
            response[trait] = predict_with_lr(text, f'c{trait}', df_train, vectorizer)
        print('LR prediction time:', time.time() - t)

        return HttpResponse(json.dumps(response))

