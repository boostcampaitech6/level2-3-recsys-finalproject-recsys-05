# Create your models here.

from users.models import AppUser as User
from django.db import models


class UserActivity(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    path = models.CharField(max_length=255)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.user.username} - {self.path}"


class Champion(models.Model):
    version = models.CharField(max_length=255)
    id = models.CharField(max_length=255, primary_key=True)
    key = models.CharField(max_length=255)
    name = models.CharField(max_length=255)
    title = models.CharField(max_length=255)
    blurb = models.TextField()
    info = models.JSONField()
    image = models.JSONField()
    tags = models.JSONField()
    partype = models.CharField(max_length=255)
    stats = models.JSONField()


class Summoner(models.Model):
    id = models.CharField(max_length=100, blank=False, null=False, primary_key=True)
    name = models.CharField(max_length=100, blank=False, null=False)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        return self.name


class LeagueEntry(models.Model):
    summoner = models.ForeignKey(Summoner, on_delete=models.CASCADE)
    tier = models.CharField(max_length=255)
    rank = models.CharField(max_length=255)
    league_points = models.IntegerField()
    wins = models.IntegerField()
    losses = models.IntegerField()
    veteran = models.BooleanField()
    inactive = models.BooleanField()
    fresh_blood = models.BooleanField()
    hot_streak = models.BooleanField()
    queue_type = models.CharField(max_length=255)
    league_id = models.CharField(max_length=255)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        return f"{self.summoner.name} - {self.tier} {self.rank}"
