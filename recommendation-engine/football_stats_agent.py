from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.by import By
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
        self.options = Options()
        self.options.add_argument("--headless")
        self.options.add_argument("--disable-gpu")
        self.options.add_argument("--no-sandbox")
        self.options.add_argument("--disable-dev-shm-usage")
        self.options.add_argument("--window-size=1920,1080")
        self.options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")
        
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
        """Scrape fixtures from fbref.com using Playwright or load from cache"""
        # Try to load from cache first
        cached_fixtures, cache_valid = self.load_cached_fixtures()
        if cache_valid:
            return cached_fixtures

        logger.info("Fetching fresh fixtures from fbref.com...")
        try:
            url = "https://fbref.com/en/comps/719/schedule/FIFA-Club-World-Cup-Scores-and-Fixtures"
            
            async with async_playwright() as p:
                browser = await p.chromium.launch(headless=True)
                page = await browser.new_page()
                
                await page.set_viewport_size({"width": 1920, "height": 1080})
                await page.goto(url)
                
                content = await page.content()
                await browser.close()
                
                soup = BeautifulSoup(content, 'html.parser')
                table = soup.find('table', {'id': 'sched_all'})
                
                if not table:
                    raise ValueError("Couldn't find fixtures table on fbref.com")
                
                rows = table.find_all('tr')
                fixtures = []
                
                for row in rows:
                    cols = row.find_all(['th', 'td'])
                    cols_text = [col.text.strip() for col in cols]

                    if len(cols_text) < 11 or cols_text[7] == '':  # Skip header rows or incomplete ones
                        continue

                    # Get team IDs from links
                    home_link = cols[5].find('a')
                    away_link = cols[7].find('a')
                    
                    home_id = extract_team_id(home_link['href']) if home_link else None
                    away_id = extract_team_id(away_link['href']) if away_link else None

                    # Clean team names
                    home_team = clean_team_name(cols_text[5])
                    away_team = clean_team_name(cols_text[7])

                    fixture = {
                        'Date': cols_text[3],
                        'Time': cols_text[4],
                        'Home': home_team,
                        'Score': cols_text[6],
                        'Away': away_team,
                        'HomeID': home_id,
                        'AwayID': away_id,
                        'Venue': cols_text[8] if len(cols_text) > 10 else '',
                    }
                    fixtures.append(fixture)
                    logger.info(f"Found fixture: {fixture}")

                if not fixtures:
                    logger.warning("No fixtures found on fbref.com.")
                    return []
                    
                # Save to cache
                self.save_fixtures_to_cache(fixtures)
                return fixtures
                
        except Exception as e:
            logger.error(f"Failed to fetch fixtures: {str(e)}")
            return []

   # async def get_head_to_head_stats(self, home_id: str, away_id: str) -> Dict:
   #     """Get head-to-head statistics between two teams from fbref"""
   #     logger.info(f"Fetching H2H stats for teams {home_id} vs {away_id}...")
        #try:
            #url = f"https://fbref.com/en/stathead/matchup/teams/{home_id}/{away_id}"
            
            async with async_playwright() as p:
                browser = await p.chromium.launch(headless=True)
                page = await browser.new_page()
                
                await page.goto(url)
                await page.wait_for_selector('table#matchup', timeout=30000)
                await page.wait_for_timeout(2000)
                
                content = await page.content()
                await browser.close()
                
                soup = BeautifulSoup(content, 'html.parser')
                table = soup.find('table', {'id': 'games_history_all'})
                
                if not table:
                    raise ValueError("Couldn't find H2H table on fbref.com")
                
                rows = table.find_all('tr')[1:5]  # Get last 4 matches
                scores = []
                
                for row in rows:
                    cols = row.find_all('td')
                    if len(cols) >= 8:
                        score = cols[7].text.strip()
                        if '-' in score:
                            home_score, away_score = map(int, score.split('-'))
                            scores.append((home_score, away_score))

                # W/D/L logic
                W = D = L = 0
                for h, a in scores:
                    if h > a:
                        W += 1
                    elif h < a:
                        L += 1
                    else:
                        D += 1

                avg_last_3 = sum(h + a for h, a in scores[:3]) / min(3, len(scores)) if scores else 0
                avg_last_4 = sum(h + a for h, a in scores) / len(scores) if scores else 0

                return {
                    "W-D-L": (W, D, L),
                    "Avg goals last 3": round(avg_last_3, 2),
                    "Avg goals last 4": round(avg_last_4, 2),
                }
                
        #except Exception as e:
        #    logger.error(f"Failed to get H2H stats: {str(e)}")
        #    return {"W-D-L": (0, 0, 0), "Avg goals last 3": 0, "Avg goals last 4": 0}

    #async def get_team_form(self, team_id: str, is_home: bool) -> Dict:
        """Get team's form and stats from fbref"""
        logger.info(f"Fetching form for team {team_id}...")
        try:
            # Use the team's stats page URL pattern
            url = f"https://fbref.com/en/squads/{team_id}/2024-2025/all_comps/stats/"
            
            async with async_playwright() as p:
                browser = await p.chromium.launch(headless=True)
                page = await browser.new_page()
                
                await page.goto(url)
                await page.wait_for_selector('table#stats_squads_standard_for', timeout=30000)
                await page.wait_for_timeout(2000)
                
                content = await page.content()
                await browser.close()
                
                soup = BeautifulSoup(content, 'html.parser')
                
                # Get home/away record
                record_type = "Home" if is_home else "Away"
                record_table = soup.find('table', {'id': f'stats_squads_standard_for'})
                
                if not record_table:
                    raise ValueError(f"Couldn't find {record_type} record table on fbref.com")
                
                rows = record_table.find_all('tr')
                for row in rows:
                    if record_type in row.text:
                        cols = row.find_all('td')
                        if len(cols) >= 8:
                            return {
                                "Record": f"{cols[0].text}-{cols[1].text}-{cols[2].text}",
                                "Goals": int(cols[3].text),
                                "GoalsPerGame": float(cols[4].text),
                                "xG": float(cols[5].text) if cols[5].text else 0,
                                "xGA": float(cols[6].text) if cols[6].text else 0,
                            }
                
                return {
                    "Record": "0-0-0",
                    "Goals": 0,
                    "GoalsPerGame": 0,
                    "xG": 0,
                    "xGA": 0,
                }
                
        except Exception as e:
            logger.error(f"Failed to get team form: {str(e)}")
            return {
                "Record": "0-0-0",
                "Goals": 0,
                "GoalsPerGame": 0,
                "xG": 0,
                "xGA": 0,
            }

    async def analyze_fixtures(self) -> List[Dict]:
        fixtures = await self.get_fixtures()
    #    results = []
     #   tasks = []
     #   for fixture in fixtures:
     #       if fixture['HomeID'] and fixture['AwayID']:
     #           tasks.append(self.analyze_fixture(fixture))
     #   analyzed = await asyncio.gather(*tasks, return_exceptions=True)
     #   for r in analyzed:
     #       if isinstance(r, dict):
     #           results.append(r)
     #   return results
        return fixtures

    async def analyze_fixture(self, fixture: Dict) -> Dict:
        try:
            home = fixture['Home']
            away = fixture['Away']
            logger.info(f"\nAnalyzing {home} vs {away} on {fixture['Date']}")

            # Defensive H2H fetch
            try:
                h2h = await self.get_head_to_head_stats(fixture['HomeID'], fixture['AwayID'])
            except Exception as e:
                logger.error(f"H2H fetch failed for {home} vs {away}: {e}")
                h2h = {"W-D-L": (0, 0, 0), "Avg goals last 3": 0, "Avg goals last 4": 0}

            # Defensive home form fetch
            try:
                home_form = await self.get_team_form(fixture['HomeID'], True)
            except Exception as e:
                logger.error(f"Home form fetch failed for {home}: {e}")
                home_form = {"Record": "0-0-0", "Goals": 0, "GoalsPerGame": 0, "xG": 0, "xGA": 0}

            # Defensive away form fetch
            try:
                away_form = await self.get_team_form(fixture['AwayID'], False)
            except Exception as e:
                logger.error(f"Away form fetch failed for {away}: {e}")
                away_form = {"Record": "0-0-0", "Goals": 0, "GoalsPerGame": 0, "xG": 0, "xGA": 0}

            result = {
                "Date": fixture['Date'],
                "Time": fixture['Time'],
                "Fixture": f"{home} vs {away}",
                "Venue": fixture['Venue'],
                "Head-to-Head": h2h["W-D-L"],
                "Avg Goals (Last 3 H2H)": h2h["Avg goals last 3"],
                "Avg Goals (Last 4 H2H)": h2h["Avg goals last 4"],
                f"{home} Record": home_form["Record"],
                f"{home} Goals": home_form["Goals"],
                f"{home} Goals/Game": home_form["GoalsPerGame"],
                f"{home} xG": home_form["xG"],
                f"{away} Record": away_form["Record"],
                f"{away} Goals": away_form["Goals"],
                f"{away} Goals/Game": away_form["GoalsPerGame"],
                f"{away} xG": away_form["xG"],
            }

            logger.info(f"Analysis complete for {home} vs {away}")
            return result
        except Exception as e:
            logger.error(f"Failed to analyze {fixture['Home']} vs {fixture['Away']}: {str(e)}")
            return {}

async def main():
    agent = FootballStatsAgent()
    # Clear cache before starting
    #agent.clear_cache()
    results = await agent.analyze_fixtures()
    df = pd.DataFrame(results)
    print("\n📊 Football Statistics Analysis")
    print("=" * 80)
    print(df.to_string(index=False))

if __name__ == "__main__":
    asyncio.run(main()) 