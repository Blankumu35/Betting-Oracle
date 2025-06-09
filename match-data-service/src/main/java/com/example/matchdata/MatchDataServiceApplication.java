package com.example.matchdata;

import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.client.RestTemplate;
import org.springframework.http.*;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.cache.annotation.EnableCaching;
import org.springframework.cache.annotation.Cacheable;
import org.springframework.cache.annotation.CacheEvict;
import org.springframework.scheduling.annotation.EnableScheduling;
import org.springframework.scheduling.annotation.Scheduled;
import java.util.*;
import java.time.LocalDateTime;
import java.time.LocalDate;
import java.time.format.DateTimeFormatter;
import com.fasterxml.jackson.annotation.JsonIgnoreProperties;
import com.fasterxml.jackson.annotation.JsonProperty;

@SpringBootApplication
@EnableCaching
@EnableScheduling
public class MatchDataServiceApplication {
    public static void main(String[] args) {
        SpringApplication.run(MatchDataServiceApplication.class, args);
    }
}

@RestController
@RequestMapping("/api/matches")
class MatchDataController {
    private final RestTemplate restTemplate = new RestTemplate();

    @Value("${football.api.key}")
    private String apiKey;

    @GetMapping
    @Cacheable(value = "matchData", key = "'today'", unless = "#result == null")
    public List<Match> getMatchData() {
        try {
            HttpHeaders headers = new HttpHeaders();
            headers.set("X-Auth-Token", apiKey);
            HttpEntity<String> entity = new HttpEntity<>(headers);

            // Get matches for today only
            String today = LocalDate.now().format(DateTimeFormatter.ISO_DATE);
            
            ResponseEntity<FixturesResponse> response = restTemplate.exchange(
                "http://api.football-data.org/v4/matches?dateFrom=" + today + "&dateTo=" + today,
                HttpMethod.GET,
                entity,
                FixturesResponse.class
            );

            if (response.getBody() == null || response.getBody().matches == null) {
                throw new RuntimeException("No matches data available from API");
            }

            return response.getBody().matches;
        } catch (Exception e) {
            System.err.println("Error fetching match data: " + e.getMessage());
            throw new RuntimeException("Failed to fetch match data: " + e.getMessage());
        }
    }

    @GetMapping("/head2head/{teamId}")
    @Cacheable(value = "head2head", key = "#teamId", unless = "#result == null")
    public List<Match> getHeadToHead(@PathVariable String teamId) {
        try {
            HttpHeaders headers = new HttpHeaders();
            headers.set("X-Auth-Token", apiKey);
            HttpEntity<String> entity = new HttpEntity<>(headers);

            ResponseEntity<FixturesResponse> response = restTemplate.exchange(
                "http://api.football-data.org/v4/teams/" + teamId + "/matches?limit=3",
                HttpMethod.GET,
                entity,
                FixturesResponse.class
            );

            if (response.getBody() == null || response.getBody().matches == null) {
                throw new RuntimeException("No head-to-head data available from API");
            }

            return response.getBody().matches;
        } catch (Exception e) {
            System.err.println("Error fetching head-to-head data: " + e.getMessage());
            throw new RuntimeException("Failed to fetch head-to-head data: " + e.getMessage());
        }
    }

    @GetMapping("/recommendation-data")
    @Cacheable(value = "recommendationData", key = "'today'", unless = "#result == null")
    public RecommendationData getRecommendationData() {
        try {
            List<Match> upcomingMatches = getMatchData();
            Map<String, List<Match>> headToHeadData = new HashMap<>();

            // Get head-to-head data for each team in upcoming matches
            for (Match match : upcomingMatches) {
                if (!headToHeadData.containsKey(match.homeTeam.id)) {
                    headToHeadData.put(match.homeTeam.id, getHeadToHead(match.homeTeam.id));
                }
                if (!headToHeadData.containsKey(match.awayTeam.id)) {
                    headToHeadData.put(match.awayTeam.id, getHeadToHead(match.awayTeam.id));
                }
            }

            return new RecommendationData(upcomingMatches, headToHeadData);
        } catch (Exception e) {
            System.err.println("Error preparing recommendation data: " + e.getMessage());
            throw new RuntimeException("Failed to prepare recommendation data: " + e.getMessage());
        }
    }

    @GetMapping("/competitions")
    @Cacheable(value = "competitions", key = "'all'", unless = "#result == null")
    public List<Competition> getCompetitions() {
        try {
            HttpHeaders headers = new HttpHeaders();
            headers.set("X-Auth-Token", apiKey);
            HttpEntity<String> entity = new HttpEntity<>(headers);

            ResponseEntity<CompetitionsResponse> response = restTemplate.exchange(
                "http://api.football-data.org/v4/competitions",
                HttpMethod.GET,
                entity,
                CompetitionsResponse.class
            );

            if (response.getBody() == null || response.getBody().competitions == null) {
                throw new RuntimeException("No competitions data available from API");
            }

            return response.getBody().competitions;
        } catch (Exception e) {
            System.err.println("Error fetching competitions: " + e.getMessage());
            throw new RuntimeException("Failed to fetch competitions: " + e.getMessage());
        }
    }

    // Schedule cache refresh at midnight every day
    @Scheduled(cron = "0 0 0 * * ?")
    @CacheEvict(value = {"matchData", "recommendationData"}, allEntries = true)
    public void clearDailyCache() {
        System.out.println("Daily cache cleared at: " + LocalDateTime.now());
    }

    // Keep head-to-head and competitions data cached longer
    @Scheduled(fixedRateString = "${cache.refresh.rate.minutes:60}000")
    @CacheEvict(value = {"head2head", "competitions"}, allEntries = true)
    public void clearOtherCache() {
        System.out.println("Other cache cleared at: " + LocalDateTime.now());
    }
}

@JsonIgnoreProperties(ignoreUnknown = true)
class FixturesResponse {
    public List<Match> matches;
    public int count;
    public Filters filters;
    public Competition competition;
    public Season season;
}

@JsonIgnoreProperties(ignoreUnknown = true)
class CompetitionsResponse {
    public int count;
    public List<Competition> competitions;
}

@JsonIgnoreProperties(ignoreUnknown = true)
class Match {
    public String id;
    public String utcDate;
    public String status;
    public int minute;
    public String stage;
    public String group;
    public Team homeTeam;
    public Team awayTeam;
    public Score score;
    public Competition competition;
    public Season season;
    public Odds odds;
    public List<Referee> referees;
}

@JsonIgnoreProperties(ignoreUnknown = true)
class Team {
    public String id;
    public String name;
    public String shortName;
    public String tla;
    public String crest;
}

@JsonIgnoreProperties(ignoreUnknown = true)
class Score {
    public ScoreDetail fullTime;
    public ScoreDetail halfTime;
    public ScoreDetail extraTime;
    public ScoreDetail penalty;
}

@JsonIgnoreProperties(ignoreUnknown = true)
class ScoreDetail {
    public Integer home;
    public Integer away;
}

@JsonIgnoreProperties(ignoreUnknown = true)
class Competition {
    public String id;
    public String name;
    public String code;
    public String type;
    public String emblem;
    public Area area;
}

@JsonIgnoreProperties(ignoreUnknown = true)
class Season {
    public String id;
    public String startDate;
    public String endDate;
    public String currentMatchday;
    public String winner;
}

@JsonIgnoreProperties(ignoreUnknown = true)
class Area {
    public String id;
    public String name;
    public String code;
    public String flag;
}

@JsonIgnoreProperties(ignoreUnknown = true)
class Filters {
    public String dateFrom;
    public String dateTo;
    public String permission;
}

@JsonIgnoreProperties(ignoreUnknown = true)
class Odds {
    public String msg;
}

@JsonIgnoreProperties(ignoreUnknown = true)
class Referee {
    public String id;
    public String name;
    public String type;
    public String nationality;
}

@JsonIgnoreProperties(ignoreUnknown = true)
class RecommendationData {
    public List<Match> upcomingMatches;
    public Map<String, List<Match>> headToHeadData;

    public RecommendationData(List<Match> upcomingMatches, Map<String, List<Match>> headToHeadData) {
        this.upcomingMatches = upcomingMatches;
        this.headToHeadData = headToHeadData;
    }
}
