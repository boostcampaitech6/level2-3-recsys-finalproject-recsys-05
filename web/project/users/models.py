from django.contrib.auth.models import AbstractUser
from django.db import models


class AppUser(AbstractUser):

    summoner = models.OneToOneField(
        "app.Summoner", on_delete=models.CASCADE, blank=True, null=True
    )
