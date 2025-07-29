package com.example.user;

import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;
import org.springframework.web.bind.annotation.*;
import java.util.*;
import java.util.concurrent.ConcurrentHashMap;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

@SpringBootApplication
public class UserServiceApplication {
    public static void main(String[] args) {
        SpringApplication.run(UserServiceApplication.class, args);
    }
}

@RestController
@RequestMapping("/api/users")
class UserController {
    private static final Logger logger = LoggerFactory.getLogger(UserController.class);
    
    // In-memory storage for users
    private final Map<String, User> users = new ConcurrentHashMap<>();
    private final Map<String, String> userPasswords = new ConcurrentHashMap<>();

    @PostMapping("/signup")
    public User signup(@RequestBody SignupRequest request) {
        logger.info("Attempting signup for email: {}", request.getEmail());
        
        if (users.containsKey(request.getEmail())) {
            logger.warn("Signup failed - Email already in use: {}", request.getEmail());
            throw new RuntimeException("Email already in use");
        }

        String userId = UUID.randomUUID().toString();
        User user = new User(userId, request.getEmail());
        users.put(request.getEmail(), user);
        userPasswords.put(request.getEmail(), request.getPassword()); // In a real app, this would be hashed
        
        logger.info("Successfully created user: {}", request.getEmail());
        return user;
    }

    @PostMapping("/login")
    public User login(@RequestBody LoginRequest request) {
        logger.info("Attempting login for email: {}", request.getEmail());
        
        User user = users.get(request.getEmail());
        if (user == null) {
            logger.warn("Login failed - User not found: {}", request.getEmail());
            throw new RuntimeException("Invalid credentials");
        }
        
        String storedPassword = userPasswords.get(request.getEmail());
        if (!storedPassword.equals(request.getPassword())) {
            logger.warn("Login failed - Invalid password for user: {}", request.getEmail());
            throw new RuntimeException("Invalid credentials");
        }
        
        logger.info("Successfully logged in user: {}", request.getEmail());
        return user;
    }

    @GetMapping
    public List<User> getAllUsers() {
        logger.info("Retrieving all users. Total users: {}", users.size());
        return new ArrayList<>(users.values());
    }

    @GetMapping("/{email}")
    public User getUser(@PathVariable String email) {
        logger.info("Retrieving user: {}", email);
        User user = users.get(email);
        if (user == null) {
            logger.warn("User not found: {}", email);
            throw new RuntimeException("User not found");
        }
        return user;
    }
}

class User {
    private String id;
    private String email;

    public User(String id, String email) {
        this.id = id;
        this.email = email;
    }

    // Getters and setters
    public String getId() { return id; }
    public void setId(String id) { this.id = id; }
    public String getEmail() { return email; }
    public void setEmail(String email) { this.email = email; }
}

class SignupRequest {
    private String email;
    private String password;

    // Getters and setters
    public String getEmail() { return email; }
    public void setEmail(String email) { this.email = email; }
    public String getPassword() { return password; }
    public void setPassword(String password) { this.password = password; }
}

class LoginRequest {
    private String email;
    private String password;

    // Getters and setters
    public String getEmail() { return email; }
    public void setEmail(String email) { this.email = email; }
    public String getPassword() { return password; }
    public void setPassword(String password) { this.password = password; }
}