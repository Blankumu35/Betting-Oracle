package com.example.betting;

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
import java.time.format.DateTimeFormatter;
import com.fasterxml.jackson.annotation.JsonIgnoreProperties;
import com.fasterxml.jackson.annotation.JsonProperty;

@SpringBootApplication
@EnableCaching
@EnableScheduling
public class BettingServiceApplication {
    public static void main(String[] args) {
        SpringApplication.run(BettingServiceApplication.class, args);
    }
}

@RestController
@RequestMapping("/suggestions")
class BettingController {
    private final RestTemplate restTemplate = new RestTemplate();
    private final Random random = new Random();

    @Value("${football.api.key}")
    private String apiKey;

    @Value("${cache.ttl.minutes:60}")
    private int cacheTtlMinutes;

    @GetMapping("/{userId}")
    @Cacheable(value = "bettingSuggestions", key = "'all'", unless = "#result == null")
    public List<BettingSuggestion> getSuggestions(@PathVariable String userId) {
        try {
            // Fetch fixtures from football-data.org
            HttpHeaders headers = new HttpHeaders();
            headers.set("X-Auth-Token", apiKey);
            HttpEntity<String> entity = new HttpEntity<>(headers);

            ResponseEntity<FixturesResponse> response = restTemplate.exchange(
                "http://api.football-data.org/v4/matches",
                HttpMethod.GET,
                entity,
                FixturesResponse.class
            );

            if (response.getBody() == null || response.getBody().matches == null) {
                throw new RuntimeException("No matches data available from API");
            }

            return generateSuggestionsFromFixtures(response.getBody().matches);
        } catch (Exception e) {
            System.err.println("Error fetching fixtures: " + e.getMessage());
            throw new RuntimeException("Failed to fetch betting suggestions: " + e.getMessage());
        }
    }

    @Scheduled(fixedRateString = "${cache.refresh.rate.minutes:60}000")
    @CacheEvict(value = "bettingSuggestions", allEntries = true)
    public void clearCache() {
        System.out.println("Cache cleared at: " + LocalDateTime.now());
    }

    private List<BettingSuggestion> generateSuggestionsFromFixtures(List<Match> matches) {
        List<BettingSuggestion> suggestions = new ArrayList<>();
        String[] predictions = {"Home Win", "Away Win", "Draw"};
        String[] reasons = {
            "Strong home form",
            "Key players returning from injury",
            "Historical advantage",
            "Recent performance trend",
            "Weather conditions favorable"
        };

        for (Match match : matches) {
            if (match.status == "SCHEDULED") {
                BettingSuggestion suggestion = new BettingSuggestion(
                    UUID.randomUUID().toString(),
                    match.id,
                    match.homeTeam.name,
                    match.awayTeam.name,
                    predictions[random.nextInt(predictions.length)],
                    random.nextDouble() * 100,
                    reasons[random.nextInt(reasons.length)],
                    random.nextDouble() * 3 + 1,
                    match.utcDate
                );
                suggestions.add(suggestion);
            }
        }

        if (suggestions.isEmpty()) {
            throw new RuntimeException("No upcoming matches available");
        }

        return suggestions;
    }
}

@JsonIgnoreProperties(ignoreUnknown = true)
class FixturesResponse {
    public List<Match> matches;
}

@JsonIgnoreProperties(ignoreUnknown = true)
class Match {
    public String id;
    public String utcDate;
    public String status;
    public Team homeTeam;
    public Team awayTeam;
}

@JsonIgnoreProperties(ignoreUnknown = true)
class Team {
    public String name;
}

class BettingSuggestion {
    private String id;
    private String matchId;
    private String homeTeam;
    private String awayTeam;
    private String prediction;
    private double confidence;
    private String reasoning;
    private double odds;
    private String timestamp;

    public BettingSuggestion(String id, String matchId, String homeTeam, String awayTeam,
                           String prediction, double confidence, String reasoning,
                           double odds, String timestamp) {
        this.id = id;
        this.matchId = matchId;
        this.homeTeam = homeTeam;
        this.awayTeam = awayTeam;
        this.prediction = prediction;
        this.confidence = confidence;
        this.reasoning = reasoning;
        this.odds = odds;
        this.timestamp = timestamp;
    }

    // Getters
    public String getId() { return id; }
    public String getMatchId() { return matchId; }
    public String getHomeTeam() { return homeTeam; }
    public String getAwayTeam() { return awayTeam; }
    public String getPrediction() { return prediction; }
    public double getConfidence() { return confidence; }
    public String getReasoning() { return reasoning; }
    public double getOdds() { return odds; }
    public String getTimestamp() { return timestamp; }
} 