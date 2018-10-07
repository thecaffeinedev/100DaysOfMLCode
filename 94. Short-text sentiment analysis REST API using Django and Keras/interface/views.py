import numpy as np
from django.shortcuts import render
from django.views.generic.base import View 

EXAMPLE_SENTENCES = ["I love Cape Town!", "I hate this movie", "Great acting.... not!"]


class IndexView(View):
	template_name = "interface/01_index.html"

	def get(self, request):
		""" Returns input form """
		example_sentence = np.random.choice(EXAMPLE_SENTENCES)
		context = {"example_sentence": example_sentence}
		return render(request, self.template_name,context)






















