package com.example.matchdata;

import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;
import org.springframework.web.bind.annotation.*;
import java.util.*;

@SpringBootApplication
public class MatchDataServiceApplication {
    public static void main(String[] args) {
        SpringApplication.run(MatchDataServiceApplication.class, args);
    }
}

@RestController
@RequestMapping("/matches")
class MatchController {
    private final List<Match> matches = new ArrayList<>();

    public MatchController() {
        // Initialize with sample data
        matches.add(new Match("1", "Manchester United", "Liverpool", 2, 1, 
            "Premier League", "2024-03-15", "Completed", 
            new MatchStats(55, 45, 8, 4, 12, 8)));
        matches.add(new Match("2", "Arsenal", "Chelsea", 0, 0, 
            "Premier League", "2024-03-16", "Scheduled", 
            new MatchStats(0, 0, 0, 0, 0, 0)));
        matches.add(new Match("3", "Barcelona", "Real Madrid", 3, 2, 
            "La Liga", "2024-03-14", "Completed", 
            new MatchStats(60, 40, 10, 5, 15, 10)));
    }

    @GetMapping
    public List<Match> getAllMatches() {
        return matches;
    }

    @GetMapping("/{id}")
    public Match getMatchById(@PathVariable String id) {
        return matches.stream()
                .filter(match -> match.getId().equals(id))
                .findFirst()
                .orElseThrow(() -> new RuntimeException("Match not found"));
    }

    @GetMapping("/league/{league}")
    public List<Match> getMatchesByLeague(@PathVariable String league) {
        return matches.stream()
                .filter(match -> match.getLeague().equals(league))
                .toList();
    }

    @GetMapping("/status/{status}")
    public List<Match> getMatchesByStatus(@PathVariable String status) {
        return matches.stream()
                .filter(match -> match.getStatus().equals(status))
                .toList();
    }
}

class Match {
    private String id;
    private String teamA;
    private String teamB;
    private int scoreA;
    private int scoreB;
    private String league;
    private String date;
    private String status;
    private MatchStats stats;

    public Match(String id, String teamA, String teamB, int scoreA, int scoreB, 
                String league, String date, String status, MatchStats stats) {
        this.id = id;
        this.teamA = teamA;
        this.teamB = teamB;
        this.scoreA = scoreA;
        this.scoreB = scoreB;
        this.league = league;
        this.date = date;
        this.status = status;
        this.stats = stats;
    }

    // Getters and setters
    public String getId() { return id; }
    public String getTeamA() { return teamA; }
    public String getTeamB() { return teamB; }
    public int getScoreA() { return scoreA; }
    public int getScoreB() { return scoreB; }
    public String getLeague() { return league; }
    public String getDate() { return date; }
    public String getStatus() { return status; }
    public MatchStats getStats() { return stats; }
}

class MatchStats {
    private int possessionA;
    private int possessionB;
    private int shotsA;
    private int shotsB;
    private int cornersA;
    private int cornersB;

    public MatchStats(int possessionA, int possessionB, int shotsA, int shotsB, 
                     int cornersA, int cornersB) {
        this.possessionA = possessionA;
        this.possessionB = possessionB;
        this.shotsA = shotsA;
        this.shotsB = shotsB;
        this.cornersA = cornersA;
        this.cornersB = cornersB;
    }

    // Getters and setters
    public int getPossessionA() { return possessionA; }
    public int getPossessionB() { return possessionB; }
    public int getShotsA() { return shotsA; }
    public int getShotsB() { return shotsB; }
    public int getCornersA() { return cornersA; }
    public int getCornersB() { return cornersB; }
}
