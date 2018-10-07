from django.urls import path, re_path
from interface.views import IndexView


urlpatterns = [
	path('', IndexView.as_view()),
]




















