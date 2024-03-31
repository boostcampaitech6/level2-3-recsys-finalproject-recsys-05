import os
import requests


KR_API_HOST = "https://kr.api.riotgames.com"
ASIA_API_HOST = "https://asia.api.riotgames.com"


class RiotClient:
    def __init__(self):
        self.api_key = os.environ.get("RIOT_API_KEY")

    def get_account_by_summoner_name(self, summoner_name) -> requests.Response:
        url = f"{KR_API_HOST}/lol/summoner/v4/summoners/by-name/{summoner_name}"
        response = requests.get(url, headers={"X-Riot-Token": self.api_key})
        return response

    def get_match_list(self, puuid: str) -> requests.Response:
        url = f"{ASIA_API_HOST}/lol/match/v5/matches/by-puuid/{puuid}/ids"
        response = requests.get(url, headers={"X-Riot-Token": self.api_key})
        return response

    def get_match(self, match_id: str) -> requests.Response:
        url = f"{ASIA_API_HOST}/lol/match/v5/matches/{match_id}"
        response = requests.get(url, headers={"X-Riot-Token": self.api_key})
        return response

    def get_champion_masteries_by_puuid_top(self, puuid: str) -> requests.Response:
        url = f"{KR_API_HOST}/lol/champion-mastery/v4/champion-masteries/by-puuid/{puuid}/top"
        response = requests.get(url, headers={"X-Riot-Token": self.api_key})
        return response

    def get_summoner_by_encrypted_summoner_id(
        self, encrypted_summoner_id: str
    ) -> requests.Response:
        url = f"{KR_API_HOST}/lol/summoner/v4/summoners/{encrypted_summoner_id}"
        response = requests.get(url, headers={"X-Riot-Token": self.api_key})
        return response


client = RiotClient()


def get_client():
    return client
