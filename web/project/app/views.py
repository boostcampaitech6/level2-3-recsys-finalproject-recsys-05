import os
import uuid
from django.http import JsonResponse
from django.shortcuts import redirect, render
from django.core.handlers.wsgi import WSGIRequest
import requests
from django.contrib.auth.decorators import login_required

from users.models import SummonerInfo
from rest_framework.decorators import api_view
from app.riot_client import get_client
from app.riot_assets import get_riot_assets
from rest_framework import viewsets
from app.serializers import SummonerSerializer, LeagueEntrySerializer
from app.models import Summoner, LeagueEntry


def riot_txt(request: WSGIRequest):
    return render(request, "riot.txt")


def index(request: WSGIRequest):
    return render(
        request,
        "index.html",
        {"user": request.user, "login_url": "http://localhost:8000/accounts/login"},
    )


def authorize(request: WSGIRequest):
    provider = "https://auth.riotgames.com"

    redirect_uri = "http://localhost:8000/oauth2-callback"
    client_id = ""
    response_type = "code"
    scope = "openid"

    url = f"{provider}/authorize?client_id={client_id}&redirect_uri={redirect_uri}&response_type={response_type}&scope={scope}"

    return redirect(url)


def get_account_by_summoner_name(request: WSGIRequest):
    summoner_name = request.GET.get("summoner_name")
    if SummonerInfo.objects.filter(name=summoner_name).exists():
        summoner_info = SummonerInfo.objects.get(name=summoner_name)

        return render(request, "summoner/index.html", {"summoner": summoner_info})

    client = get_client()

    response = client.get_account_by_summoner_name(summoner_name)

    if response.status_code == 200:
        data = response.json()

        summoner_info = SummonerInfo(
            id=data["id"],
            account_id=data["accountId"],
            profile_icon_id=data["profileIconId"],
            revision_date=data["revisionDate"],
            name=data["name"],
            puuid=data["puuid"],
            summoner_level=data["summonerLevel"],
        )

        summoner_info.save()

        return render(request, "summoner/index.html", {"summoner": summoner_info})

    return render(request, "summoner/index.html", {"error": response.json()})


def recommend_ai(request: WSGIRequest):
    return render(request, "recommend/ai.html", {"user": request.user})


@login_required
def recommend_result(request: WSGIRequest):
    return render(
        request,
        "recommend/result.html",
        {"user": request.user},
    )


@api_view(["GET"])
def search_summoners_by_name(request: WSGIRequest):
    name = request.GET.get("name")
    count = request.GET.get("count")
    if not name or not count:
        return JsonResponse({"error": "name, count are required"}, status=400)
    
    try:
        count = int(count)
    except ValueError:
        return JsonResponse({"error": "count must be an integer"}, status=400)
    
    if count > 10:
        return JsonResponse({"error": "count must be less than or equal to 10"}, status=400)
    
    queryset = Summoner.objects.filter(name__startswith=name)[:count]
    serializer = SummonerSerializer(queryset, many=True)
    
    summoners = serializer.data
    
    summoner_ids = [
        summoner["id"]
        for summoner in serializer.data
    ]
    
    queryset = LeagueEntry.objects.filter(summoner_id__in=summoner_ids)
    serializer = LeagueEntrySerializer(queryset, many=True)
    
    league_entries = serializer.data
    
    result = [
        {
            "summoner": summoner,
            "league_entries": [
                league_entry
                for league_entry in league_entries
                if league_entry["summoner"] == summoner["id"]
            ]
        }
        for summoner in summoners
    ]
    
    return JsonResponse(result, safe=False)