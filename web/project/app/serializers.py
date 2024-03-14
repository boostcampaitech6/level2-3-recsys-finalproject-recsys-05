from app.models import Summoner
from rest_framework import serializers

class SummonerSerializer(serializers.ModelSerializer):
    class Meta:
        model = Summoner
        fields = "__all__"
