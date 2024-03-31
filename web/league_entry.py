import asyncio
import datetime
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

        self.db_user = os.environ.get("POSTGRES_USER")
        self.db_password = os.environ.get("POSTGRES_PASSWORD")
        self.db_host = "localhost"
        self.db_port = os.environ.get("POSTGRES_PORT")
        self.db_name = os.environ.get("POSTGRES_WEB_DB")

        self.pool = None

    async def create_db_pool(self):
        self.pool = await asyncpg.create_pool(
            user=self.db_user,
            password=self.db_password,
            host=self.db_host,
            port=self.db_port,
            database=self.db_name,
        )

    async def run(self):
        await self.create_db_pool()

        for tier in self.tiers:
            for division in self.divisions:
                await self.get_league_entries(tier, division)

        await self.session.close()
        await self.pool.close()

    async def get_league_entries(self, tier: str, division: str):
        url = f"{self.base_url}/{tier}/{division}?api_key={self.api_key}"
        page = 1

        league_entries = await self._fetch(url + f"&page={page}")

        while league_entries:
            print(
                f"Fetching {tier} {division} page {page}... entries: {len(league_entries)}"
            )
            summoners = [
                (
                    league_entry["summonerId"],
                    league_entry["summonerName"],
                    datetime.datetime.now(),
                    datetime.datetime.now(),
                )
                for league_entry in league_entries
            ]

            async with self.pool.acquire() as connection:
                async with connection.transaction():
                    await connection.executemany(
                        "INSERT INTO app_summoner (id, name, created_at, updated_at)"
                        "VALUES ($1, $2, $3, $4)"
                        "ON CONFLICT (id)"
                        """
                        DO UPDATE SET 
                            name = EXCLUDED.name,
                            updated_at = EXCLUDED.updated_at
                        """,
                        summoners,
                    )

            entries = [
                (
                    league_entry["tier"],
                    league_entry["rank"],
                    league_entry["leaguePoints"],
                    league_entry["wins"],
                    league_entry["losses"],
                    league_entry["veteran"],
                    league_entry["inactive"],
                    league_entry["freshBlood"],
                    league_entry["hotStreak"],
                    league_entry["queueType"],
                    league_entry["leagueId"],
                    league_entry["summonerId"],
                    datetime.datetime.now(),
                    datetime.datetime.now(),
                )
                for league_entry in league_entries
            ]
            async with self.pool.acquire() as connection:
                async with connection.transaction():
                    await connection.executemany(
                        "INSERT INTO app_leagueentry (tier, rank, league_points, wins, losses, veteran, inactive, fresh_blood, hot_streak, queue_type, league_id, summoner_id, created_at, updated_at)"
                        "VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14)"
                        "ON CONFLICT (summoner_id, queue_type)"
                        """
                        DO UPDATE SET
                            tier = EXCLUDED.tier,
                            rank = EXCLUDED.rank,
                            league_points = EXCLUDED.league_points,
                            wins = EXCLUDED.wins,
                            losses = EXCLUDED.losses,
                            veteran = EXCLUDED.veteran,
                            inactive = EXCLUDED.inactive,
                            fresh_blood = EXCLUDED.fresh_blood,
                            hot_streak = EXCLUDED.hot_streak,
                            queue_type = EXCLUDED.queue_type,
                            league_id = EXCLUDED.league_id,
                            summoner_id = EXCLUDED.summoner_id,
                            updated_at = EXCLUDED.updated_at
                        """,
                        entries,
                    )

            page += 1
            league_entries = await self._fetch(url + f"&page={page}")

        return []

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
    crawler = LeagueEntryCrawler()
    await crawler.run()


loop = asyncio.get_event_loop()
loop.run_until_complete(main())
loop.close()
