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
    terms_of_service,
    privacy_policy,
    inference,
    get_duo_match,
    get_duo_match_history,
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
    path("recommend/inference", inference, name="inference"),
    path("accounts/", include("allauth.urls")),
    path("match/duo", get_duo_match, name="get-duo-match"),
    path("match/duo/history", get_duo_match_history, name="get-duo-match-history"),
    path("tos", terms_of_service, name="terms-of-service"),
    path("privacy", privacy_policy, name="privacy-policy"),
]

if settings.DEBUG:
    import debug_toolbar

    urlpatterns = [path("__debug__/", include(debug_toolbar.urls))] + urlpatterns
