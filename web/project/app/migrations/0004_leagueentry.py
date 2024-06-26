# Generated by Django 5.0.2 on 2024-03-14 05:49

import django.db.models.deletion
from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ("app", "0003_summoner"),
    ]

    operations = [
        migrations.CreateModel(
            name="LeagueEntry",
            fields=[
                (
                    "id",
                    models.BigAutoField(
                        auto_created=True,
                        primary_key=True,
                        serialize=False,
                        verbose_name="ID",
                    ),
                ),
                ("tier", models.CharField(max_length=255)),
                ("rank", models.CharField(max_length=255)),
                ("league_points", models.IntegerField()),
                ("wins", models.IntegerField()),
                ("losses", models.IntegerField()),
                ("veteran", models.BooleanField()),
                ("inactive", models.BooleanField()),
                ("fresh_blood", models.BooleanField()),
                ("hot_streak", models.BooleanField()),
                ("queue_type", models.CharField(max_length=255)),
                ("league_id", models.CharField(max_length=255)),
                ("created_at", models.DateTimeField(auto_now_add=True)),
                ("updated_at", models.DateTimeField(auto_now=True)),
                (
                    "summoner",
                    models.ForeignKey(
                        on_delete=django.db.models.deletion.CASCADE, to="app.summoner"
                    ),
                ),
            ],
        ),
    ]
