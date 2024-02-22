import asyncio
import os
import random
from typing import Generator
import aiohttp
import inquirer
from loguru import logger
from tqdm import tqdm
import orjson as json
import ijson

logger.remove()
logger.add("user_summary.log", format="{time} {level} {message}", level="INFO")

API_HOST = "https://op.gg"
USER_AGENT = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_14_4) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/75.0.3770.100 Safari/537.36"


def get_uri(region, summoner_id):
    return (
        f"{API_HOST}/api/v1.0/internal/bypass/summoners/{region}/{summoner_id}/summary"
    )


def user_generator(file_path: str) -> Generator[str, None, None]:
    with open(file_path, "rb") as file:
        for obj in ijson.items(file, "item.summoner_id"):
            yield obj


def save_jsonl(dir_name: str, file_name: str, data: dict):
    full_path = os.path.join(dir_name, f"{file_name}l")

    os.makedirs(os.path.dirname(full_path), exist_ok=True)

    with open(full_path, "a", encoding="utf-8") as jsonl_file:
        jsonl_file.write(json.dumps(data).decode("utf-8"))
        jsonl_file.write("\n")


class UserCrawler:
    def __init__(
        self,
        summoner_file_paths: list[str],
        output_dir: str = "./summoner_info",
        last_summonor_id: str = "",
    ):
        self.session = aiohttp.ClientSession()
        self.summoner_file_paths = summoner_file_paths
        self.output_dir = output_dir
        self.last_summoner_id = last_summonor_id
        self.snapshot_mode = last_summonor_id != ""

    async def _get(self, url) -> dict | None:
        headers = {"User-Agent": USER_AGENT}
        timeout = aiohttp.ClientTimeout(total=0)

        while True:
            try:
                async with self.session.get(
                    url, headers=headers, timeout=timeout
                ) as response:
                    if response.status == 429:  # Rate limit exceeded
                        sleep_time = random.uniform(5, 10)
                        logger.info(
                            f"Rate limit exceeded. Retrying after {sleep_time} seconds."
                        )
                        await asyncio.sleep(sleep_time)
                    elif response.status == 404:  # Not found
                        logger.info("Not found")
                        return None
                    elif response.status != 200:  # Other errors
                        logger.error(f"Error: {response.status}")
                        return None
                    elif response.status == 200:
                        return await response.json()
            except Exception as e:
                logger.error(f"Error: {e}")
                return None

    async def start(self):
        for summoner_file_path in self.summoner_file_paths:
            logger.info(f"Processing {summoner_file_path}")
            for summoner_id in tqdm(user_generator(summoner_file_path)):
                if self.snapshot_mode and summoner_id == self.last_summoner_id:
                    self.snapshot_mode = False
                    logger.info(f"Resuming from {summoner_id}")
                elif self.snapshot_mode:
                    continue

                logger.info(f"Group: {summoner_file_path}, Summoner ID: {summoner_id}")
                response = await self._get(get_uri("KR", summoner_id))
                if response is not None:
                    continue

                save_jsonl(self.output_dir, summoner_file_path, response["data"])
                self.last_summoner_id = summoner_id
                asyncio.sleep(random.uniform(0.5, 1))

        await self.session.close()


async def new_user_crawler(
    summoner_file_paths: list[str],
    output_dir: str = "./summoner_info",
    last_summonor_id: str = "",
):
    return UserCrawler(
        summoner_file_paths,
        output_dir,
        last_summonor_id,
    )


async def start(
    summoner_file_paths: list[str],
    output_dir: str = "./summoner_info",
    last_summonor_id: str = "",
):
    user_crawler = await new_user_crawler(
        summoner_file_paths,
        output_dir,
        last_summonor_id,
    )
    await user_crawler.start()


if __name__ == "__main__":
    questions = [
        inquirer.Text(
            "file_path",
            message="디렉토리를 선택해주세요",
        ),
    ]

    file_path = inquirer.prompt(questions)["file_path"]

    if os.path.exists(file_path) is False:
        logger.error(f"File not found: {file_path}")
        raise FileNotFoundError("File not found")

    file_list = os.listdir(file_path)
    file_list = [file for file in file_list if file.endswith(".json")]

    if len(file_list) == 0:
        logger.error(f"No JSON file found in {file_path}")
        raise FileNotFoundError("No JSON file found")

    questions = [
        inquirer.Checkbox(
            "input_files",
            message="변환할 파일을 선택해주세요",
            choices=file_list,
        ),
        inquirer.Text(
            "output_dir_path",
            message="저장할 디렉토리의 경로를 입력해주세요",
            default="./summoner_info",
        ),
        inquirer.Text(
            "last_summoner_id",
            message="마지막으로 크롤링한 소환사 ID를 입력해주세요",
        ),
    ]
    answers = inquirer.prompt(questions)

    asyncio.run(
        start(
            answers["input_files"],
            answers["output_dir_path"],
            answers["last_summoner_id"],
        )
    )
