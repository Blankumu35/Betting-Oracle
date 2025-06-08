package com.example.userpref;

import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;
import org.springframework.web.bind.annotation.*;

import java.util.concurrent.ConcurrentHashMap;
import java.util.Map;

@SpringBootApplication
public class UserPrefServiceApplication {
    public static void main(String[] args) {
        SpringApplication.run(UserPrefServiceApplication.class, args);
    }
}

@RestController
@RequestMapping("/preferences")
class UserPrefController {
    private final Map<String, String> preferences = new ConcurrentHashMap<>();

    @GetMapping
    public Map<String, String> getPreferences() {
        return preferences;
    }

    @PostMapping
    public String setPreference(@RequestParam String user, @RequestParam String pref) {
        preferences.put(user, pref);
        return "Preference set for user: " + user;
    }
}