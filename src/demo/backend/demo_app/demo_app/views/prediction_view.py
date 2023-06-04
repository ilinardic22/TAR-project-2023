import json

from django.http import HttpResponse
from django.views import View


class PredictionView(View):
    def post(self, request):
        text = json.loads(request.body).get('text', '')
        if text == '':
            return HttpResponse('')



        dict_to_return = {
            'aaa': '111',
            'bbb': '222',
            'text': text
        }
        return HttpResponse(json.dumps(dict_to_return))

