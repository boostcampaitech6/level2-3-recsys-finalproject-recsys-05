from django.urls import path
from .views import index, get_account_by_summoner_name, recommend_ai, riot_txt

urlpatterns = [
    path("riot.txt", riot_txt),
    path("", index, name="home"),
    path("summoner/", get_account_by_summoner_name, name="summoner"),
    path("recommend-ai/", recommend_ai, name="recommend-ai"),
]