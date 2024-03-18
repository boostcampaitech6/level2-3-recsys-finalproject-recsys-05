from app.models import Summoner, LeagueEntry
from rest_framework import serializers


class SummonerSerializer(serializers.ModelSerializer):
    class Meta:
        model = Summoner
        fields = "__all__"


class LeagueEntrySerializer(serializers.ModelSerializer):
    class Meta:
        model = LeagueEntry
        fields = "__all__"
