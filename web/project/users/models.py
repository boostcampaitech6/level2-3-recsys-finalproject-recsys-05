from django.contrib.auth.models import AbstractUser
from django.db import models


class AppUser(AbstractUser):
    summoner_info = models.OneToOneField(
        "SummonerInfo", on_delete=models.CASCADE, blank=True, null=True
    )


class SummonerInfo(models.Model):
    id = models.CharField(max_length=100, blank=False, null=False, primary_key=True)
    account_id = models.CharField(max_length=100, blank=False, null=False)
    profile_icon_id = models.IntegerField(blank=False, null=False)
    revision_date = models.BigIntegerField(blank=False, null=False)
    name = models.CharField(max_length=100, blank=False, null=False)
    puuid = models.CharField(max_length=100, blank=False, null=False)
    summoner_level = models.IntegerField(blank=False, null=False)
