from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from bs4 import BeautifulSoup
import time
import pandas as pd
from typing import List, Dict, Tuple
import logging
import re

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FootballStatsAgent:
    def __init__(self):
        self.options = Options()
        self.options.add_argument("--headless")  # Run in headless mode
        self.options.add_argument("--disable-gpu")
        self.options.add_argument("--no-sandbox")
        self.options.add_argument("--disable-dev-shm-usage")
        self.options.add_argument("--window-size=1920,1080")
        self.options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")
        
    def get_driver(self):
        """Create and return a new Chrome driver instance"""
        service = Service(ChromeDriverManager().install())
        return webdriver.Chrome(service=service, options=self.options)

    def get_fixtures(self) -> List[Tuple[str, str]]:
        """Scrape today's fixtures from Livescore"""
        logger.info("Fetching fixtures...")
        driver = self.get_driver()
        try:
            driver.get("https://www.livescore.com")
            time.sleep(5)  # Wait for JS to load
            
            # Wait for matches to load
            WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.CLASS_NAME, "match-row"))
            )
            
            soup = BeautifulSoup(driver.page_source, "html.parser")
            fixtures = []
            
            # Find all match containers
            matches = soup.find_all("div", class_="match-row")
            
            for match in matches:
                try:
                    teams = match.find_all("div", class_="team-name")
                    if len(teams) >= 2:
                        home = teams[0].text.strip()
                        away = teams[1].text.strip()
                        fixtures.append((home, away))
                        logger.info(f"Found fixture: {home} vs {away}")
                except Exception as e:
                    logger.error(f"Error parsing match: {str(e)}")
                    continue
            
            return fixtures
        finally:
            driver.quit()

    def get_head_to_head_stats(self, team1: str, team2: str) -> Dict:
        """Get head-to-head statistics between two teams"""
        logger.info(f"Fetching H2H stats for {team1} vs {team2}...")
        driver = self.get_driver()
        try:
            # Use Livescore for H2H
            search_url = f"https://www.livescore.com/en/head2head/{team1}-vs-{team2}"
            driver.get(search_url)
            time.sleep(5)

            soup = BeautifulSoup(driver.page_source, "html.parser")
            matches = soup.find_all("div", class_="match-row")[:4]
            
            results = []
            total_goals = []
            
            for match in matches:
                try:
                    score = match.find("div", class_="score").text.strip()
                    # Handle different score formats (e.g., "2-1", "2 : 1")
                    score = re.sub(r'[^\d-]', '', score)
                    home_goals, away_goals = map(int, score.split('-'))
                    total_goals.append(home_goals + away_goals)
                    
                    if home_goals == away_goals:
                        results.append("D")
                    elif home_goals > away_goals:
                        results.append("W")
                    else:
                        results.append("L")
                except Exception as e:
                    logger.error(f"Error parsing match score: {str(e)}")
                    continue

            if not total_goals:
                logger.warning(f"No H2H data found for {team1} vs {team2}")
                return {
                    "W-D-L": (0, 0, 0),
                    "Avg goals last 3": 0,
                    "Avg goals last 4": 0
                }

            last3_avg = round(sum(total_goals[:3]) / min(3, len(total_goals)), 2)
            last4_avg = round(sum(total_goals[:4]) / min(4, len(total_goals)), 2)
            
            w = results.count("W")
            d = results.count("D")
            l = results.count("L")

            return {
                "W-D-L": (w, d, l),
                "Avg goals last 3": last3_avg,
                "Avg goals last 4": last4_avg
            }
        finally:
            driver.quit()

    def get_team_form(self, team_name: str) -> List[str]:
        """Get team's form in last 4 matches"""
        logger.info(f"Fetching form for {team_name}...")
        driver = self.get_driver()
        try:
            # Use Livescore for team form
            search_url = f"https://www.livescore.com/en/team/{team_name}/results"
            driver.get(search_url)
            time.sleep(5)

            soup = BeautifulSoup(driver.page_source, "html.parser")
            matches = soup.find_all("div", class_="match-row")[:4]
            form = []

            for match in matches:
                try:
                    teams = match.find_all("div", class_="team-name")
                    score = match.find("div", class_="score").text.strip()
                    score = re.sub(r'[^\d-]', '', score)
                    
                    home_team = teams[0].text.strip()
                    away_team = teams[1].text.strip()
                    home_goals, away_goals = map(int, score.split('-'))
                    
                    if team_name.lower() in home_team.lower():
                        if home_goals > away_goals:
                            form.append("W")
                        elif home_goals < away_goals:
                            form.append("L")
                        else:
                            form.append("D")
                    elif team_name.lower() in away_team.lower():
                        if away_goals > home_goals:
                            form.append("W")
                        elif away_goals < home_goals:
                            form.append("L")
                        else:
                            form.append("D")
                except Exception as e:
                    logger.error(f"Error parsing match: {str(e)}")
                    continue

            return form[:4]
        finally:
            driver.quit()

    def analyze_fixtures(self) -> List[Dict]:
        """Analyze all fixtures and return detailed statistics"""
        fixtures = self.get_fixtures()
        if not fixtures:
            logger.error("No fixtures found!")
            return []
            
        results = []
        
        for home, away in fixtures:
            try:
                logger.info(f"\nAnalyzing {home} vs {away}")
                
                h2h = self.get_head_to_head_stats(home, away)
                form_home = self.get_team_form(home)
                form_away = self.get_team_form(away)
                
                result = {
                    "Fixture": f"{home} vs {away}",
                    "Head-to-Head": h2h["W-D-L"],
                    "Avg Goals (Last 3 H2H)": h2h["Avg goals last 3"],
                    "Avg Goals (Last 4 H2H)": h2h["Avg goals last 4"],
                    f"{home} Form": form_home,
                    f"{away} Form": form_away
                }
                
                results.append(result)
                logger.info(f"Analysis complete for {home} vs {away}")
                
            except Exception as e:
                logger.error(f"Failed to analyze {home} vs {away}: {str(e)}")
                continue
                
        return results

def main():
    agent = FootballStatsAgent()
    results = agent.analyze_fixtures()
    
    if not results:
        print("\n❌ No results found. Please check the logs for errors.")
        return
        
    # Convert results to DataFrame for better display
    df = pd.DataFrame(results)
    print("\n📊 Football Statistics Analysis")
    print("=" * 80)
    print(df.to_string(index=False))

if __name__ == "__main__":
    main() 