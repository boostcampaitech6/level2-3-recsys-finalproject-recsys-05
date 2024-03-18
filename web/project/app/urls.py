from django.urls import path, include

from project import settings
from .views import (
    index,
    get_account_by_summoner_name,
    recommend_ai,
    recommend_result,
    riot_txt,
    search_summoners_by_name,
    save_summoner,
)

urlpatterns = [
    path("riot.txt", riot_txt),
    path("", index, name="home"),
    path("summoner/", get_account_by_summoner_name, name="summoner"),
    path("recommend-ai/", recommend_ai, name="recommend-ai"),
    path("recommend-result/", recommend_result, name="recommend-result"),
    path(
        "summoners/search/", search_summoners_by_name, name="search-summoners-by-name"
    ),
    path("profile/summoner", save_summoner, name="save-summoner-info"),
]

if settings.DEBUG:
    import debug_toolbar

    urlpatterns = [path("__debug__/", include(debug_toolbar.urls))] + urlpatterns
