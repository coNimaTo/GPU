// Config.h
#pragma once
#include <fstream>
#include <sstream>
#include <string>
#include <stdexcept>
#include <unordered_map>

struct Config {
    // Grid
    int   N           = 65;
    int   algorithm   = 2;

    // Simulation
    int   N_STEPS     = 5000;
    int   FREQ_SAVE   = 10;
    float DT          = 0.025f;
    float GRAVITY     = 9.81f;
    float DX          = 1.0f;

    // Erosion
    float KC          = 1.0f;
    float KS          = 0.01f;
    float KD          = 0.01f;
    float KE          = 0.2f;

    // Rain
    float RAIN_AMOUNT = 0.1f;
    float RAIN_DROPS  = 50.0f;
    int   RAIN_STOP   = 1000;

    // Sources
    float WATER_LEVEL = 0.005f;

    static Config load(const char* path) {
        std::ifstream f(path);
        if (!f) throw std::runtime_error("Could not open config file");

        // Parse key=value pairs, ignore # comments and blank lines
        std::unordered_map<std::string, std::string> kv;
        std::string line;
        while (std::getline(f, line)) {
            auto comment = line.find('#');
            if (comment != std::string::npos) line = line.substr(0, comment);
            auto eq = line.find('=');
            if (eq == std::string::npos) continue;
            std::string key = trim(line.substr(0, eq));
            std::string val = trim(line.substr(eq + 1));
            if (!key.empty() && !val.empty()) kv[key] = val;
        }

        Config c;
        auto get_i = [&](const char* k, int&   v){ if (kv.count(k)) v = std::stoi(kv[k]); };
        auto get_f = [&](const char* k, float& v){ if (kv.count(k)) v = std::stof(kv[k]); };

        get_i("N",           c.N);
        get_i("algorithm",   c.algorithm);
        get_i("N_STEPS",     c.N_STEPS);
        get_i("FREQ_SAVE",   c.FREQ_SAVE);
        get_f("DT",          c.DT);
        get_f("GRAVITY",     c.GRAVITY);
        get_f("DX",          c.DX);
        get_f("KC",          c.KC);
        get_f("KS",          c.KS);
        get_f("KD",          c.KD);
        get_f("KE",          c.KE);
        get_f("RAIN_AMOUNT", c.RAIN_AMOUNT);
        get_f("RAIN_DROPS",  c.RAIN_DROPS);
        get_i("RAIN_STOP",   c.RAIN_STOP);
        get_f("WATER_LEVEL", c.WATER_LEVEL);

        return c;
    }

    void print() const {
        printf("── Config ──────────────────────────\n");
        printf("  N=%d  algorithm=%d\n",     N, algorithm);
        printf("  N_STEPS=%d  FREQ_SAVE=%d\n", N_STEPS, FREQ_SAVE);
        printf("  DT=%.4f  G=%.2f  DX=%.2f\n", DT, GRAVITY, DX);
        printf("  KC=%.4f  KS=%.4f  KD=%.4f  KE=%.4f\n", KC, KS, KD, KE);
        printf("  RAIN_AMOUNT=%.4f  RAIN_DROPS=%.1f  RAIN_STOP=%d\n",
               RAIN_AMOUNT, RAIN_DROPS, RAIN_STOP);
        printf("  WATER_LEVEL=%.4f\n", WATER_LEVEL);
        printf("────────────────────────────────────\n");
    }

private:
    static std::string trim(const std::string& s) {
        size_t a = s.find_first_not_of(" \t\r\n");
        size_t b = s.find_last_not_of(" \t\r\n");
        return (a == std::string::npos) ? "" : s.substr(a, b - a + 1);
    }
};