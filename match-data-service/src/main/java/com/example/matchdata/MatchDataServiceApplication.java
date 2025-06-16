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
import org.springframework.cache.CacheManager;
import org.springframework.beans.factory.annotation.Autowired;
import java.util.*;
import java.time.LocalDateTime;
import java.time.LocalDate;
import java.time.format.DateTimeFormatter;
import com.fasterxml.jackson.annotation.JsonIgnoreProperties;
import com.fasterxml.jackson.annotation.JsonProperty;
import java.util.stream.Collectors;

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

    @Value("${rapidapi.key}")
    private String rapidApiKey;

    @Value("${rapidapi.host}")
    private String rapidApiHost;

    @Autowired
    private CacheManager cacheManager;

    @GetMapping
    public ResponseEntity<?> getMatchData() {
        try {
            // Get today's date in YYYYMMDD format
            String today = LocalDate.now().format(DateTimeFormatter.ofPattern("yyyyMMdd"));
            System.out.println("Checking matches for date: " + today);

            // Try to get from cache first
            List<Match> cachedMatches = getMatchesFromCache(today);
            if (cachedMatches != null) {
                System.out.println("Retrieved matches from cache for date: " + today);
                return ResponseEntity.ok(cachedMatches);
            }

            // If not in cache, fetch from API
            System.out.println("Cache miss - fetching matches from API for date: " + today);
            List<Match> matches = fetchMatchesFromAPI(today);
            return ResponseEntity.ok(matches);
        } catch (Exception e) {
            System.err.println("Error in getMatchData: " + e.getMessage());
            e.printStackTrace(); // Print full stack trace
            return ResponseEntity.status(HttpStatus.INTERNAL_SERVER_ERROR)
                .body("Error fetching match data: " + e.getMessage());
        }
    }

    private List<Match> getMatchesFromCache(String date) {
        try {
            var cache = cacheManager.getCache("matchData");
            if (cache != null) {
                var cachedValue = cache.get(date);
                if (cachedValue != null) {
                    return (List<Match>) cachedValue.get();
                }
            }
        } catch (Exception e) {
            System.err.println("Error retrieving from cache: " + e.getMessage());
        }
        return null;
    }

    private List<Match> fetchMatchesFromAPI(String date) {
        try {
            HttpHeaders headers = new HttpHeaders();
            headers.set("x-rapidapi-key", rapidApiKey);
            headers.set("x-rapidapi-host", rapidApiHost);

            System.out.println("Making API request with headers: " + headers);
            System.out.println("API URL: https://free-api-live-football-data.p.rapidapi.com/football-get-matches-by-date?date=" + date);

            ResponseEntity<FixturesResponse> response = restTemplate.exchange(
                "https://free-api-live-football-data.p.rapidapi.com/football-get-matches-by-date?date=" + date,
                HttpMethod.GET,
                new HttpEntity<>(headers),
                FixturesResponse.class
            );

            System.out.println("API Response Status: " + response.getStatusCode());
            System.out.println("API Response Body: " + response.getBody());

            if (response.getBody() == null) {
                throw new RuntimeException("API returned null response body");
            }

            if (response.getBody().response == null || response.getBody().response.matches == null) {
                throw new RuntimeException("API response contains null matches array");
            }

            // Cache the results
            var cache = cacheManager.getCache("matchData");
            if (cache != null) {
                cache.put(date, response.getBody().response.matches);
                System.out.println("Cached matches for date: " + date);
            }

            return response.getBody().response.matches;
        } catch (Exception e) {
            System.err.println("Error fetching from API: " + e.getMessage());
            e.printStackTrace();
            throw new RuntimeException("Failed to fetch matches from API: " + e.getMessage());
        }
    }

    @GetMapping("/head2head/{teamId}")
    @Cacheable(value = "head2head", key = "#teamId", unless = "#result == null")
    public ResponseEntity<?> getHeadToHead(@PathVariable String teamId) {
        try {
            HttpHeaders headers = new HttpHeaders();
            headers.set("x-rapidapi-key", rapidApiKey);
            headers.set("x-rapidapi-host", rapidApiHost);

            ResponseEntity<FixturesResponse> response = restTemplate.exchange(
                "https://free-api-live-football-data.p.rapidapi.com/football-get-matches-by-date?date=" + LocalDate.now().format(DateTimeFormatter.ofPattern("yyyyMMdd")),
                HttpMethod.GET,
                new HttpEntity<>(headers),
                FixturesResponse.class
            );

            if (response.getBody() == null || response.getBody().response == null || response.getBody().response.matches == null) {
                throw new RuntimeException("No head-to-head data available from API");
            }

            return ResponseEntity.ok(response.getBody().response.matches);
        } catch (Exception e) {
            System.err.println("Error fetching head-to-head data: " + e.getMessage());
            e.printStackTrace();
            return ResponseEntity.status(HttpStatus.INTERNAL_SERVER_ERROR)
                .body("Error fetching head-to-head data: " + e.getMessage());
        }
    }

    @GetMapping("/recommendation-data")
    @Cacheable(value = "recommendationData", key = "'today'", unless = "#result == null")
    public ResponseEntity<?> getRecommendationData() {
        try {
            ResponseEntity<?> matchesResponse = getMatchData();
            if (matchesResponse.getStatusCode() != HttpStatus.OK) {
                return matchesResponse;
            }
            List<Match> upcomingMatches = (List<Match>) matchesResponse.getBody();
            Map<String, List<Match>> headToHeadData = new HashMap<>();

            // Get head-to-head data for each team in upcoming matches
            for (Match match : upcomingMatches) {
                if (!headToHeadData.containsKey(match.home.id)) {
                    ResponseEntity<?> h2hResponse = getHeadToHead(match.home.id);
                    if (h2hResponse.getStatusCode() == HttpStatus.OK) {
                        headToHeadData.put(match.home.id, (List<Match>) h2hResponse.getBody());
                    }
                }
                if (!headToHeadData.containsKey(match.away.id)) {
                    ResponseEntity<?> h2hResponse = getHeadToHead(match.away.id);
                    if (h2hResponse.getStatusCode() == HttpStatus.OK) {
                        headToHeadData.put(match.away.id, (List<Match>) h2hResponse.getBody());
                    }
                }
            }

            // Transform the data to match the expected format
            Map<String, Object> response = new HashMap<>();
            response.put("upcomingMatches", upcomingMatches.stream()
                .map(match -> {
                    Map<String, Object> transformedMatch = new HashMap<>();
                    transformedMatch.put("id", match.id);
                    transformedMatch.put("homeTeam", new HashMap<String, Object>() {{
                        put("id", match.home.id);
                        put("name", match.home.name);
                    }});
                    transformedMatch.put("awayTeam", new HashMap<String, Object>() {{
                        put("id", match.away.id);
                        put("name", match.away.name);
                    }});
                    transformedMatch.put("score", new HashMap<String, Object>() {{
                        put("fullTime", new HashMap<String, Object>() {{
                            put("home", match.home.score);
                            put("away", match.away.score);
                        }});
                    }});
                    return transformedMatch;
                })
                .collect(Collectors.toList()));

            // Transform head-to-head data
            Map<String, List<Map<String, Object>>> transformedHeadToHead = new HashMap<>();
            headToHeadData.forEach((teamId, matches) -> {
                transformedHeadToHead.put(teamId, matches.stream()
                    .map(match -> {
                        Map<String, Object> transformedMatch = new HashMap<>();
                        transformedMatch.put("id", match.id);
                        transformedMatch.put("homeTeam", new HashMap<String, Object>() {{
                            put("id", match.home.id);
                            put("name", match.home.name);
                        }});
                        transformedMatch.put("awayTeam", new HashMap<String, Object>() {{
                            put("id", match.away.id);
                            put("name", match.away.name);
                        }});
                        transformedMatch.put("score", new HashMap<String, Object>() {{
                            put("fullTime", new HashMap<String, Object>() {{
                                put("home", match.home.score);
                                put("away", match.away.score);
                            }});
                        }});
                        return transformedMatch;
                    })
                    .collect(Collectors.toList()));
            });
            response.put("headToHeadData", transformedHeadToHead);

            return ResponseEntity.ok(response);
        } catch (Exception e) {
            System.err.println("Error preparing recommendation data: " + e.getMessage());
            e.printStackTrace();
            return ResponseEntity.status(HttpStatus.INTERNAL_SERVER_ERROR)
                .body("Error preparing recommendation data: " + e.getMessage());
        }
    }

    @GetMapping("/competitions")
    @Cacheable(value = "competitions", key = "'all'", unless = "#result == null")
    public List<Competition> getCompetitions() {
        try {
            HttpHeaders headers = new HttpHeaders();
            headers.set("X-Auth-Token", rapidApiKey);
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

    // Keep head-to-head data cached longer
    @Scheduled(fixedRateString = "${cache.refresh.rate.minutes:60}000")
    @CacheEvict(value = {"head2head"}, allEntries = true)
    public void clearOtherCache() {
        System.out.println("Other cache cleared at: " + LocalDateTime.now());
    }
}

@JsonIgnoreProperties(ignoreUnknown = true)
class FixturesResponse {
    public String status;
    public ResponseData response;
}

@JsonIgnoreProperties(ignoreUnknown = true)
class ResponseData {
    public List<Match> matches;
}

@JsonIgnoreProperties(ignoreUnknown = true)
class CompetitionsResponse {
    public int count;
    public List<Competition> competitions;
}

@JsonIgnoreProperties(ignoreUnknown = true)
class Match {
    public String id;
    public String leagueId;
    public String time;
    public Team home;
    public Team away;
    public String eliminatedTeamId;
    public int statusId;
    public String tournamentStage;
    public MatchStatus status;
    public long timeTS;
}

@JsonIgnoreProperties(ignoreUnknown = true)
class Team {
    public String id;
    public int score;
    public String name;
    public String longName;
    public int redCards;
}

@JsonIgnoreProperties(ignoreUnknown = true)
class MatchStatus {
    public String utcTime;
    public int numberOfHomeRedCards;
    public int numberOfAwayRedCards;
    public Halfs halfs;
    public int periodLength;
    public boolean finished;
    public boolean started;
    public boolean cancelled;
    public boolean awarded;
    public String scoreStr;
    public Reason reason;
}

@JsonIgnoreProperties(ignoreUnknown = true)
class Halfs {
    public String firstHalfStarted;
    public String secondHalfStarted;
}

@JsonIgnoreProperties(ignoreUnknown = true)
class Reason {
    public String short_;
    public String shortKey;
    public String long_;
    public String longKey;
}

@JsonIgnoreProperties(ignoreUnknown = true)
class Competition {
    public String id;
    public String name;
    public String country;
    public String logo;
    public String flag;
}

@JsonIgnoreProperties(ignoreUnknown = true)
class League {
    public String id;
    public String name;
    public String country;
    public String logo;
    public String flag;
    public int season;
    public String round;
}

@JsonIgnoreProperties(ignoreUnknown = true)
class Filters {
    public String date;
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
