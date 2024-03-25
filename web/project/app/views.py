import datetime
from django.http import JsonResponse
from django.shortcuts import redirect, render
from django.core.handlers.wsgi import WSGIRequest
from django.contrib.auth.decorators import login_required

from app.models import DuoMatch, Summoner
from rest_framework.decorators import api_view
from app.riot_client import get_client
from app.serializers import SummonerSerializer, LeagueEntrySerializer
from app.models import LeagueEntry
from users.models import AppUser


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
    if Summoner.objects.filter(name=summoner_name).exists():
        summoner = Summoner.objects.get(name=summoner_name)

        return render(request, "summoner/index.html", {"summoner": summoner})

    client = get_client()

    response = client.get_account_by_summoner_name(summoner_name)

    if response.status_code == 200:
        data = response.json()

        summoner = Summoner(
            id=data["id"],
            account_id=data["accountId"],
            profile_icon_id=data["profileIconId"],
            revision_date=data["revisionDate"],
            name=data["name"],
            puuid=data["puuid"],
            summoner_level=data["summonerLevel"],
        )

        summoner.save()

        return render(request, "summoner/index.html", {"summoner": summoner})

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
        return JsonResponse(
            {"error": "count must be less than or equal to 10"}, status=400
        )

    queryset = Summoner.objects.filter(name__startswith=name)[:count]
    serializer = SummonerSerializer(queryset, many=True)

    summoners = serializer.data

    summoner_ids = [summoner["id"] for summoner in serializer.data]

    queryset = LeagueEntry.objects.filter(summoner_id__in=summoner_ids)
    serializer = LeagueEntrySerializer(queryset, many=True)

    league_entries = serializer.data

    for summoner in summoners:
        if summoner["puuid"] is None:
            new_summoner = get_client().get_summoner_by_encrypted_summoner_id(
                summoner["id"]
            )

            if new_summoner.status_code == 200:
                summoner["puuid"] = new_summoner.json()["puuid"]
                summoner["account_id"] = new_summoner.json()["accountId"]
                summoner["profile_icon_id"] = new_summoner.json()["profileIconId"]
                summoner["revision_date"] = new_summoner.json()["revisionDate"]
                summoner["summoner_level"] = new_summoner.json()["summonerLevel"]

                Summoner.objects.filter(id=summoner["id"]).update(
                    puuid=summoner["puuid"],
                    account_id=summoner["account_id"],
                    profile_icon_id=summoner["profile_icon_id"],
                    revision_date=summoner["revision_date"],
                    summoner_level=summoner["summoner_level"],
                )

    result = [
        {
            "summoner": summoner,
            "league_entries": [
                league_entry
                for league_entry in league_entries
                if league_entry["summoner"] == summoner["id"]
            ],
        }
        for summoner in summoners
    ]

    return JsonResponse(result, safe=False)


@api_view(["POST"])
def save_summoner(request: WSGIRequest):
    me = request.user

    summoner_id = request.data["summoner_id"]

    if not summoner_id:
        return JsonResponse({"error": "id is required"}, status=400)

    summoner = Summoner.objects.get(id=summoner_id)

    AppUser.objects.filter(id=me.id).update(summoner=summoner)

    return JsonResponse(
        {
            "name": summoner.name,
            "puuid": summoner.puuid,
            "level": summoner.summoner_level,
            "account_id": summoner.account_id,
            "profile_icon_id": summoner.profile_icon_id,
            "revision_date": summoner.revision_date,
        },
        safe=False,
    )


def terms_of_service(request: WSGIRequest):
    return render(request, "terms_of_service.html")


def privacy_policy(request: WSGIRequest):
    return render(request, "privacy_policy.html")


@api_view(["POST"])
def inference(request: WSGIRequest):
    me = request.user

    # Fake User

    user = AppUser.objects.get(id=2)

    result = DuoMatch.objects.create(user1=me, user2=user)
    result.save()

    return JsonResponse(
        {
            "duo_match_id": result.id,
            "user1": me.username,
            "user2": user.username,
        },
        safe=False,
    )


def get_duo_match(request: WSGIRequest):
    duo_match_id = request.GET.get("duo_match_id")

    if not duo_match_id:
        return JsonResponse({"error": "duo_match_id is required"}, status=400)

    duo_match = DuoMatch.objects.get(id=duo_match_id)

    summoner2: Summoner = duo_match.user2.summoner

    summoner2_rank = LeagueEntry.objects.get(summoner=summoner2)

    return render(
        request,
        "recommend/duo_match.html",
        {
            "user": request.user,
            "summoner1": duo_match.user1.summoner,
            "summoner2": duo_match.user2.summoner,
            "summoner2_rank": summoner2_rank,
        },
    )


def _n_days_ago(target_date: datetime.datetime):
    timezone = datetime.timezone(datetime.timedelta(hours=9))
    today = datetime.datetime.now().astimezone(timezone)

    diff = today - target_date.astimezone(timezone)

    if diff.days == 0 and diff.seconds < 3600:
        return f"{diff.seconds // 60}분 전"

    if diff.days == 0:
        return f"{diff.seconds // 3600}시간 전"

    if diff.days < 7:
        return f"{diff.days}일 전"

    return f"{diff.days}일 전"


@login_required(login_url="/accounts/discord/login/")
def get_duo_match_history(request: WSGIRequest):
    me = request.user
    timezone = datetime.timezone(datetime.timedelta(hours=9))

    duo_matches = [
        {
            "id": duo_match.id,
            "target_username": duo_match.user2.summoner.name,
            "target_profile_icon_id": duo_match.user2.summoner.profile_icon_id,
            "target_summoner_tier": LeagueEntry.objects.get(
                summoner=duo_match.user2.summoner
            ).tier,
            "target_summoner_rank": LeagueEntry.objects.get(
                summoner=duo_match.user2.summoner
            ).rank,
            "created_at": duo_match.created_at.astimezone(timezone).strftime(
                "%Y-%m-%d %H:%M:%S"
            ),
            "n_days_ago": _n_days_ago(duo_match.created_at),
        }
        for duo_match in DuoMatch.objects.filter(user1=me).order_by("-created_at")
    ]

    return render(
        request,
        "recommend/history.html",
        {"user": me, "duo_matches": duo_matches},
    )
