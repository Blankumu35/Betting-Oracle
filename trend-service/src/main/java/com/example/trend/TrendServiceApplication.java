package com.example.trend;

import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;
import org.springframework.web.bind.annotation.*;
import java.util.*;

@SpringBootApplication
public class TrendServiceApplication {
    public static void main(String[] args) {
        SpringApplication.run(TrendServiceApplication.class, args);
    }
}

@RestController
@RequestMapping("/trends")
class TrendController {
    
    @GetMapping("/team/{teamName}")
    public TeamTrends getTeamTrends(@PathVariable String teamName) {
        // Sample data - in a real application, this would be calculated from historical data
        return new TeamTrends(
            teamName,
            new FormTrend(5, 3, 1, 1, 10, 5),
            new PerformanceTrend(65, 2.1, 1.2, 4.5),
            new HeadToHeadTrend(3, 1, 1)
        );
    }

    @GetMapping("/league/{leagueName}")
    public LeagueTrends getLeagueTrends(@PathVariable String leagueName) {
        // Sample data - in a real application, this would be calculated from historical data
        return new LeagueTrends(
            leagueName,
            new ScoringTrend(2.8, 1.2, 3.5),
            new HomeAdvantageTrend(45, 30, 25),
            new OverUnderTrend(60, 40)
        );
    }
}

class TeamTrends {
    private String teamName;
    private FormTrend formTrend;
    private PerformanceTrend performanceTrend;
    private HeadToHeadTrend headToHeadTrend;

    public TeamTrends(String teamName, FormTrend formTrend, 
                     PerformanceTrend performanceTrend, HeadToHeadTrend headToHeadTrend) {
        this.teamName = teamName;
        this.formTrend = formTrend;
        this.performanceTrend = performanceTrend;
        this.headToHeadTrend = headToHeadTrend;
    }

    // Getters
    public String getTeamName() { return teamName; }
    public FormTrend getFormTrend() { return formTrend; }
    public PerformanceTrend getPerformanceTrend() { return performanceTrend; }
    public HeadToHeadTrend getHeadToHeadTrend() { return headToHeadTrend; }
}

class FormTrend {
    private int matchesPlayed;
    private int wins;
    private int draws;
    private int losses;
    private int goalsScored;
    private int goalsConceded;

    public FormTrend(int matchesPlayed, int wins, int draws, int losses, 
                    int goalsScored, int goalsConceded) {
        this.matchesPlayed = matchesPlayed;
        this.wins = wins;
        this.draws = draws;
        this.losses = losses;
        this.goalsScored = goalsScored;
        this.goalsConceded = goalsConceded;
    }

    // Getters
    public int getMatchesPlayed() { return matchesPlayed; }
    public int getWins() { return wins; }
    public int getDraws() { return draws; }
    public int getLosses() { return losses; }
    public int getGoalsScored() { return goalsScored; }
    public int getGoalsConceded() { return goalsConceded; }
}

class PerformanceTrend {
    private double averagePossession;
    private double averageGoalsScored;
    private double averageGoalsConceded;
    private double averageCorners;

    public PerformanceTrend(double averagePossession, double averageGoalsScored,
                          double averageGoalsConceded, double averageCorners) {
        this.averagePossession = averagePossession;
        this.averageGoalsScored = averageGoalsScored;
        this.averageGoalsConceded = averageGoalsConceded;
        this.averageCorners = averageCorners;
    }

    // Getters
    public double getAveragePossession() { return averagePossession; }
    public double getAverageGoalsScored() { return averageGoalsScored; }
    public double getAverageGoalsConceded() { return averageGoalsConceded; }
    public double getAverageCorners() { return averageCorners; }
}

class HeadToHeadTrend {
    private int wins;
    private int draws;
    private int losses;

    public HeadToHeadTrend(int wins, int draws, int losses) {
        this.wins = wins;
        this.draws = draws;
        this.losses = losses;
    }

    // Getters
    public int getWins() { return wins; }
    public int getDraws() { return draws; }
    public int getLosses() { return losses; }
}

class LeagueTrends {
    private String leagueName;
    private ScoringTrend scoringTrend;
    private HomeAdvantageTrend homeAdvantageTrend;
    private OverUnderTrend overUnderTrend;

    public LeagueTrends(String leagueName, ScoringTrend scoringTrend,
                       HomeAdvantageTrend homeAdvantageTrend, OverUnderTrend overUnderTrend) {
        this.leagueName = leagueName;
        this.scoringTrend = scoringTrend;
        this.homeAdvantageTrend = homeAdvantageTrend;
        this.overUnderTrend = overUnderTrend;
    }

    // Getters
    public String getLeagueName() { return leagueName; }
    public ScoringTrend getScoringTrend() { return scoringTrend; }
    public HomeAdvantageTrend getHomeAdvantageTrend() { return homeAdvantageTrend; }
    public OverUnderTrend getOverUnderTrend() { return overUnderTrend; }
}

class ScoringTrend {
    private double averageGoalsPerGame;
    private double homeGoalsPerGame;
    private double awayGoalsPerGame;

    public ScoringTrend(double averageGoalsPerGame, double homeGoalsPerGame, 
                       double awayGoalsPerGame) {
        this.averageGoalsPerGame = averageGoalsPerGame;
        this.homeGoalsPerGame = homeGoalsPerGame;
        this.awayGoalsPerGame = awayGoalsPerGame;
    }

    // Getters
    public double getAverageGoalsPerGame() { return averageGoalsPerGame; }
    public double getHomeGoalsPerGame() { return homeGoalsPerGame; }
    public double getAwayGoalsPerGame() { return awayGoalsPerGame; }
}

class HomeAdvantageTrend {
    private int homeWinsPercentage;
    private int drawsPercentage;
    private int awayWinsPercentage;

    public HomeAdvantageTrend(int homeWinsPercentage, int drawsPercentage, 
                             int awayWinsPercentage) {
        this.homeWinsPercentage = homeWinsPercentage;
        this.drawsPercentage = drawsPercentage;
        this.awayWinsPercentage = awayWinsPercentage;
    }

    // Getters
    public int getHomeWinsPercentage() { return homeWinsPercentage; }
    public int getDrawsPercentage() { return drawsPercentage; }
    public int getAwayWinsPercentage() { return awayWinsPercentage; }
}

class OverUnderTrend {
    private int overPercentage;
    private int underPercentage;

    public OverUnderTrend(int overPercentage, int underPercentage) {
        this.overPercentage = overPercentage;
        this.underPercentage = underPercentage;
    }

    // Getters
    public int getOverPercentage() { return overPercentage; }
    public int getUnderPercentage() { return underPercentage; }
}