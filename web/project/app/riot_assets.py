import requests
from app.models import Champion


class RiotAssets:
    def __init__(self):
        pass

    def get_champion(self, champion_id: str) -> dict:
        try:
            champion = Champion.objects.get(key=champion_id)
        except Champion.DoesNotExist:
            champion = None

        if champion is None:
            self._upsert_champions()
            champion = Champion.objects.get(key=champion_id)

        return champion

    def _fetch_all_champions(self) -> list:
        champions = requests.get(
            "https://ddragon.leagueoflegends.com/cdn/10.6.1/data/ko_KR/champion.json"
        ).json()["data"]
        return champions

    def _upsert_champions(self) -> None:
        champions = self._fetch_all_champions()

        for champion in champions:
            data = champions[champion]
            Champion.objects.update_or_create(
                version=data["version"],
                id=data["id"],
                key=data["key"],
                name=data["name"],
                title=data["title"],
                blurb=data["blurb"],
                info=data["info"],
                image=data["image"],
                tags=data["tags"],
                partype=data["partype"],
                stats=data["stats"],
            )

        return None


riot_assets = RiotAssets()


def get_riot_assets():
    return riot_assets
