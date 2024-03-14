import os
import uuid
from django.shortcuts import redirect, render
from django.core.handlers.wsgi import WSGIRequest
import requests
from django.contrib.auth.decorators import login_required

from users.models import SummonerInfo
from app.riot_client import get_client
from app.riot_assets import get_riot_assets


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
