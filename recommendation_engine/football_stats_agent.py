from bs4 import BeautifulSoup
import time
import pandas as pd
from typing import List, Dict, Tuple
import logging
import requests
import re
from datetime import datetime, timedelta
import asyncio
from playwright.async_api import async_playwright
import aiohttp
from urllib.parse import urljoin
import json
import os
from pathlib import Path
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
from webdriver_manager.chrome import ChromeDriverManager


# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def extract_team_id(url: str) -> str:
    """Extract team ID from fbref URL"""
    match = re.search(r'/squads/([a-f0-9]+)/', url)
    return match.group(1) if match else None

def clean_team_name(name: str) -> str:
    """Remove country/nation identifiers from team names"""
    # Remove country codes and names
    patterns = [
        r'^[a-z]{2,3}\s+',  # Two or three letter codes at start (e.g., "br Palmeiras")
        r'^[A-Z]{2,3}\s+',  # Two or three letter codes in caps at start
        r'^\([a-z]{2,3}\)\s+',  # Two or three letter codes in parentheses at start
        r'^\([A-Z]{2,3}\)\s+',  # Two or three letter codes in caps in parentheses at start
        r'\s+[a-z]{2,3}$',  # Two or three letter codes at end (e.g., "Inter Miami us")
        r'\s+[A-Z]{2,3}$',  # Two or three letter codes in caps at end
        r'\s+\([a-z]{2,3}\)$',  # Two or three letter codes in parentheses at end
        r'\s+\([A-Z]{2,3}\)$',  # Two or three letter codes in caps in parentheses at end
    ]
    
    cleaned_name = name
    for pattern in patterns:
        cleaned_name = re.sub(pattern, '', cleaned_name, flags=re.IGNORECASE)
    
    return cleaned_name.strip()

class FootballStatsAgent:
    def __init__(self):
        # Remove Selenium Options, only keep cache setup
        # self.options = Options()
        # self.options.add_argument("--headless")
        # self.options.add_argument("--disable-gpu")
        # self.options.add_argument("--no-sandbox")
        # self.options.add_argument("--disable-dev-shm-usage")
        # self.options.add_argument("--window-size=1920,1080")
        # self.options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")
        # Set up cache directory
        self.cache_dir = Path("cache")
        self.cache_dir.mkdir(exist_ok=True)
        self.fixtures_cache_file = self.cache_dir / "fixtures_cache.json"

    def clear_cache(self):
        """Clear the fixtures cache"""
        try:
            if self.fixtures_cache_file.exists():
                self.fixtures_cache_file.unlink()
                logger.info("Cache cleared successfully")
            else:
                logger.info("No cache file found to clear")
        except Exception as e:
            logger.error(f"Error clearing cache: {str(e)}")

    def load_cached_fixtures(self) -> Tuple[List[Dict], bool]:
        """Load fixtures from cache if available and not expired"""
        try:
            if not self.fixtures_cache_file.exists():
                return [], False

            with open(self.fixtures_cache_file, 'r') as f:
                cache_data = json.load(f)
                
            # Check if cache is expired (24 hours)
            cache_time = datetime.fromisoformat(cache_data['timestamp'])
            if datetime.now() - cache_time > timedelta(hours=24):
                logger.info("Cache expired, will fetch new fixtures")
                return [], False
                
            logger.info("Using cached fixtures")
            return cache_data['fixtures'], True
            
        except Exception as e:
            logger.error(f"Error loading cache: {str(e)}")
            return [], False

    def save_fixtures_to_cache(self, fixtures: List[Dict]):
        """Save fixtures to cache with timestamp"""
        try:
            cache_data = {
                'timestamp': datetime.now().isoformat(),
                'fixtures': fixtures
            }
            
            with open(self.fixtures_cache_file, 'w') as f:
                json.dump(cache_data, f, indent=2)
                
            logger.info("Fixtures saved to cache")
            
        except Exception as e:
            logger.error(f"Error saving to cache: {str(e)}")

    async def get_fixtures(self) -> List[Dict]:
        """Scrape fixtures from fctables.com/livescore/ using Playwright or load from cache"""
        cached_fixtures, cache_valid = self.load_cached_fixtures()
        if cache_valid:
            df = pd.DataFrame(cached_fixtures)
            df.to_csv("live_fixtures_with_h2h.csv", index=False)
            return cached_fixtures

        logger.info("Fetching fresh fixtures from fctables.com/ ...")
        try:
            options = Options()
            options.headless = True
            driver = webdriver.Chrome(options=options)
            driver.get("https://www.fctables.com/")

            WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.ID, "live-games-box"))
            )

            time.sleep(2)  # Allow extra JS content to load
            soup = BeautifulSoup(driver.page_source, 'html.parser')
            driver.quit()

            fixtures = []
            live_games = soup.find("div", {"id": "live-games-box"})
            if not live_games:
                logger.warning("No live games found on fctables.com/")
                return []

            game_divs = live_games.find_all("div", class_="live-game")
            for game in game_divs:
                try:
                    home = game.select_one(".home a").text.strip()
                    away = game.select_one(".away a").text.strip()
                    score = game.select_one(".score strong").text.strip() if game.select_one(".score strong") else ''
                    h2h_url = game.select_one("a.btn-info")
                    h2h_url = h2h_url["href"] if h2h_url else None
                    if h2h_url and not h2h_url.startswith("http"):
                        h2h_url = "https://www.fctables.com" + h2h_url
                    fixture = {
                        'Home': home,
                        'Score': score,
                        'Away': away,
                        'League': '',
                        'H2H_URL': h2h_url
                    }
                    fixtures.append(fixture)
                    logger.info(f"Found fixture: {fixture}")
                except Exception as e:
                    logger.error(f"Error processing a game: {e}")
                    continue
            if not fixtures:
                logger.warning("No fixtures found in live-games-box.")
                return []
            self.save_fixtures_to_cache(fixtures)
            df = pd.DataFrame(fixtures)
            df.to_csv("live_fixtures_with_h2h.csv", index=False)

            return fixtures
        except Exception as e:
            logger.error(f"Failed to fetch fixtures: {str(e)}")
            return []

  

    def get_h2h_stats_from_url(self, url: str) -> Dict:
        """Extract H2H stats from the given fctables H2H URL"""
        options = Options()
        options.headless = True
        driver = webdriver.Chrome(options=options)
        driver.get(url)
        try:
            WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.ID, "team-profile-stats"))
            )
        except Exception as e:
            logger.warning(f"H2H table not found for {url}: {e}")
        time.sleep(2)
        soup = BeautifulSoup(driver.page_source, 'html.parser')
        driver.quit()

        # Use the extractor function to get team stats
        team_stats = self.extract_team_stats_from_h2h(soup)
        return team_stats

    async def get_team_elo_rating(self, team_name: str) -> int:
        """Fetch the current Elo rating for a team from clubelo.com"""
        import re
        import aiohttp
        import unicodedata
        # Normalize team name for URL
        def normalize_name(name):
            name = unicodedata.normalize('NFKD', name).encode('ascii', 'ignore').decode('ascii')
            name = re.sub(r'[^a-zA-Z0-9 ]', '', name)
            name = name.replace(' ', '')
            return name
        url = f"http://clubelo.com/{normalize_name(team_name)}"
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url) as resp:
                    html = await resp.text()
            # Find Elo rating in the HTML
            match = re.search(r'Elo\s*:\s*([0-9]{3,5})', html)
            if match:
                return int(match.group(1))
            # Fallback: try to find the first 4-digit number in a table cell
            match = re.search(r'<td[^>]*>([1-2][0-9]{3})<\/td>', html)
            if match:
                return int(match.group(1))
        except Exception as e:
            logger.error(f"Failed to fetch Elo for {team_name}: {e}")
        return 1500  # Default Elo if not found

 
    async def analyze_fixture(self, fixtures: Dict) -> Dict:

        for fixture in fixtures:
            try:
                home = fixture['Home']
                away = fixture['Away']
                logger.info(f"\nAnalyzing {home} vs {away}")

            # --- Use H2H_URL directly ---
                try:
                    h2h_url = fixture.get("H2H_URL", "")
                    if h2h_url:
                        h2h = self.get_h2h_stats_from_url(h2h_url)
                    else:
                        raise ValueError("Missing H2H_URL")
                except Exception as e:
                    logger.error(f"H2H fetch failed for {home} vs {away}: {e}")
                    h2h = {"W-D-L": (0, 0, 0), "Avg goals last 3": 0, "Avg goals last 4": 0}

                  # --- Elo Ratings ---
                try:
                    home_elo = await self.get_team_elo_rating(home)
                except Exception as e:
                    logger.error(f"Home Elo fetch failed for {home}: {e}")
                    home_elo = 1500

                try:
                    away_elo = await self.get_team_elo_rating(away)
                except Exception as e:
                    logger.error(f"Away Elo fetch failed for {away}: {e}")
                    away_elo = 1500

            # --- Build result ---
                result = {
                    "Date": fixture['Date'],
                    "Time": fixture['Time'],
                    "Fixture": f"{home} vs {away}",
                    "Venue": fixture['Venue'],
                    "Head-to-Head": h2h.get("W-D-L", (0, 0, 0)),
                    "Avg Goals (Last 3 H2H)": h2h.get("Avg goals last 3", 0),
                    "Avg Goals (Last 4 H2H)": h2h.get("Avg goals last 4", 0),
                    "HomeStats": h2h.get(home, {}),
                    "AwayStats": h2h.get(away, {}),
                }

                logger.info(f"Analysis complete for {home} vs {away}")
                print(f"\n[DEBUG] analyze_fixture result for {home} vs {away}:")
                for k, v in result.items():
                    print(f"{k}: {v}")
                print("[END DEBUG]\n")
                return result

            except Exception as e:
                logger.error(f"Failed to analyze {fixture['Home']} vs {fixture['Away']}: {str(e)}")
                return {}

    def extract_team_stats_from_h2h(self, soup: BeautifulSoup) -> Dict:
        """Extract home and away team stats from fctables H2H page soup (using divs, not tables)"""
        team_blocks = soup.find_all("div", class_="col-sm-6 col-md-12")
        stats = {}

        for idx, block in enumerate(team_blocks[:2]):  # Only first two blocks
            # Team name
            name_tag = block.find("span", itemprop="name")
            team_name = name_tag.get_text(strip=True) if name_tag else f"Team{idx+1}"

            # Form (W/D/L)
            form_labels = block.select("div.form-box span div.label")
            form = [label.get_text(strip=True) for label in form_labels]

            # Last 6 matches stats
            last6_stats = {}
            last6_ul = block.find("div", class_="team_stats_forms")
            if last6_ul:
                for li in last6_ul.find_all("li"):
                    stat_name = li.find("p").get_text(strip=True)
                    stat_value = li.find("div").get_text(strip=True)
                    last6_stats[stat_name] = stat_value

            # Overall stats
            overall_stats = {}
            overall_ul = block.find("div", class_="team_stats_item")
            if overall_ul:
                for li in overall_ul.find_all("li"):
                    stat_name = li.find("p").get_text(strip=True)
                    stat_value = li.find("div").get_text(strip=True)
                    overall_stats[stat_name] = stat_value

            stats[team_name] = {
                "Form": form,
                "Last6Stats": last6_stats,
                "OverallStats": overall_stats
            }

        return stats
    
    def extract_team_comparison_stats(self, soup: BeautifulSoup) -> Dict:
        """Extracts team comparison stats from the H2H page's team_stats_vs table."""
        stats = {}
        table = soup.find("table", id="team_stats_vs")
        if not table:
            logger.warning("No team comparison table found (id=team_stats_vs)")
            return stats

        # Get team names from table headers
        headers = table.find_all("th")
        if len(headers) < 3:
            logger.warning("Not enough headers in team_stats_vs table")
            return stats
        home_team = headers[0].get_text(strip=True)
        away_team = headers[2].get_text(strip=True)

        stats[home_team] = {}
        stats[away_team] = {}

        # Iterate over rows and extract stat name and values
        for row in table.find("tbody").find_all("tr"):
            cols = row.find_all("td")
            if len(cols) == 3:
                home_val = cols[0].get_text(strip=True)
                stat_name = cols[1].get_text(strip=True)
                away_val = cols[2].get_text(strip=True)
                stats[home_team][stat_name] = home_val
                stats[away_team][stat_name] = away_val

        return stats
        

async def main():
    agent = FootballStatsAgent()
    # Clear cache before starting
    agent.clear_cache()
    results = await agent.analyze_fixture(await agent.get_fixtures())
    df = pd.DataFrame(results)
    print("\n📊 Football Statistics Analysis")
    print("=" * 80)
    print(df.to_string(index=False))

if __name__ == "__main__":
    asyncio.run(main())