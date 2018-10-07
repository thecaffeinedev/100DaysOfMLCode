from django.shortcuts import render
from django.views.generic.base import View 
from django.http.response import HttpResponse, JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.utils.decorators import method_decorator
import json
import tensorflow as tf

from models.wrapper import SentimentAnalysisModel


## Auxillary function
def load_model():
	""" Load pre-trained sentiment classifier """
	## TODO: read in from config
	weights_file = "trained_models/2018_06_15_10_weights.h5"
	params_file = "trained_models/2018_06_15_10_params.json"
	preprocessor_file = "trained_models/2018_06_15_10_preprocessor.pkl" 
	sentiment_model = SentimentAnalysisModel.load(weights_file, params_file, preprocessor_file)
	return sentiment_model

MODEL = load_model()



class PredictSentimentView(View):
	""" API endpoint to predict sentiment of text """

	def get(self, request, *args, **kwargs):
		""" GET requests returns a warning message"""
		return JsonResponse({"response":"Use the post method","status": 400})

	def post(self, request, *args, **kwargs):
		""" POST requests take in text and return sentiment"""
		try:
			json_object = json.loads(request.body.decode("utf-8"))
			text = json_object['input']
		except:
			return JsonResponse({"response": "json object does not contain 'input'", "status": 400})

		if text!="":
			## Inference on loaded model
			sent_class = MODEL.predict([text])
			## TODO: Asynchronous Celery call to process model prediction? Not needed now. 
			return JsonResponse({"text": text, "sentiment_score": sent_class, "response": "Successful", "status": 200})
		else:
			return JsonResponse({"text": "Please enter a comment", "sentiment_score": None, "response": "Successful", "status": 200})

	@method_decorator(csrf_exempt)
	def dispatch(self, request, *args, **kwargs):
		""" Handle Cross-Site Request Forgery """
		return View.dispatch(self, request, *args, **kwargs)
