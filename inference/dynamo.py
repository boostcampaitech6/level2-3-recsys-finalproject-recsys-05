from enum import Enum
import boto3
from pydantic import BaseModel
from config import Config, get_config


class MostChampions(BaseModel):
    summoner_id: str
    champions: list["Champion"]

    class Champion(BaseModel):
        summoner_name: str
        champion_id: int
        total_kills: int
        total_deaths: int
        total_assists: int
        total_played: int
        win_count: int
        loss_count: int

    @classmethod
    def from_dict(cls, d: dict):
        return cls(
            summoner_id=d["summoner_id"],
            champions=[
                cls.Champion(
                    summoner_name=champion["summoner_name"],
                    champion_id=champion["champion_id"],
                    total_kills=champion["total_kills"],
                    total_deaths=champion["total_deaths"],
                    total_assists=champion["total_assists"],
                    total_played=champion["total_played"],
                    win_count=champion["win_count"],
                    loss_count=champion["loss_count"],
                )
                for champion in d["champions"]
            ],
        )


class DynamoTableEnum(str, Enum):
    most_champions = "most_champions"


class DynamoClient:
    def __init__(self, config: Config):
        self.client = boto3.client(
            "dynamodb",
            region_name="ap-northeast-2",
            aws_access_key_id=config.dynamodb_aws_access_key_id,
            aws_secret_access_key=config.dynamodb_aws_secret_access_key,
        )

    def parse_item(self, item: dict):
        if isinstance(item, dict):
            if "S" in item:
                return item["S"]
            elif "N" in item:
                return int(item["N"]) if item["N"].isdigit() else float(item["N"])
            elif "L" in item:
                return [self.parse_item(elem) for elem in item["L"]]
            elif "M" in item:
                return {key: self.parse_item(value) for key, value in item["M"].items()}
            else:
                return {key: self.parse_item(value) for key, value in item.items()}
        elif isinstance(item, list):
            return [self.parse_item(elem) for elem in item]
        else:
            return item

    def get_most3_champions(self, summoner_id: str) -> MostChampions | None:
        response = self.client.get_item(
            TableName=DynamoTableEnum.most_champions,
            Key={"summoner_id": {"S": summoner_id}},
        )

        if "Item" not in response:
            return None

        return MostChampions.from_dict(self.parse_item(response["Item"]))


dynamo_client = DynamoClient(get_config())


def get_dynamo_client():
    return dynamo_client
