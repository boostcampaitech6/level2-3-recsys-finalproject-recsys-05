import asyncio
import os
import asyncpg
import aiohttp


class LeagueEntryCrawler:
    def __init__(self):
        self.session = aiohttp.ClientSession()

        self.base_url = (
            "https://kr.api.riotgames.com/lol/league/v4/entries/RANKED_SOLO_5x5"
        )
        self.api_key = os.environ.get("RIOT_API_KEY")
        self.tiers = ["IRON", "BRONZE", "SILVER", "GOLD", "PLATINUM", "DIAMOND"]
        self.divisions = ["I", "II", "III", "IV"]

    async def run(self):
        # for tier in self.tiers:
        #     for division in self.divisions:
        #         await self.get_league_entries(tier, division)

        await self.get_league_entries(self.tiers[0], self.divisions[0])

        await self.session.close()

    async def get_league_entries(self, tier: str, division: str):
        url = f"{self.base_url}/{tier}/{division}?api_key={self.api_key}"
        page = 1

        league_entries = await self._fetch(url + f"&page={page}")

        summoners = []

        while league_entries:
            summoners.extend(league_entries)
            print(f"Fetching page {page}")

            page += 1
            league_entries = await self._fetch(url + f"&page={page}")

        print(summoners)

    async def _fetch(self, url: str) -> list:
        async with self.session.get(url) as response:
            response = await self.session.get(url)

            while response.status == 429:
                retry_after = response.headers.get("Retry-After")
                print(f"Retrying after {retry_after} seconds")

                await asyncio.sleep(int(retry_after))
                response = await self.session.get(url)

            return await response.json()


async def main():
    # crawler = LeagueEntryCrawler()
    # await crawler.run()

    db_user = os.environ.get("POSTGRES_USER")
    db_password = os.environ.get("POSTGRES_PASSWORD")
    db_host = "localhost"
    db_port = os.environ.get("POSTGRES_PORT")
    db_name = os.environ.get("POSTGRES_WEB_DB")

    db_url = f"postgresql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"

    print(f"Connecting to database: {db_url}")

    pool = await asyncpg.create_pool(db_url)

    print("Connected to database")

    await pool.close()


loop = asyncio.get_event_loop()
loop.run_until_complete(main())
loop.close()
