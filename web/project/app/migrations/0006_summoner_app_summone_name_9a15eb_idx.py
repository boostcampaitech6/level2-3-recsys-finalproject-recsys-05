# Generated by Django 5.0.2 on 2024-03-14 08:25

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ("app", "0005_leagueentry_unique_league_entry"),
    ]

    operations = [
        migrations.AddIndex(
            model_name="summoner",
            index=models.Index(fields=["name"], name="app_summone_name_9a15eb_idx"),
        ),
    ]