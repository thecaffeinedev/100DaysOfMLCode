from django.urls import path, re_path
from api.views import PredictSentimentView


urlpatterns = [
	re_path(r'^get_sentiment/', PredictSentimentView.as_view()),
]





