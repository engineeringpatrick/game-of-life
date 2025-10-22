#pragma once
#include <vector>
#include <thread>
#include <barrier>
#include <atomic>
#include <random>
#include <cstdint>
#include <algorithm>

struct SimConfig {
    int width  = 1024;
    int height = 768;
    int species = 6;                  
    int workers = std::max(1u, std::thread::hardware_concurrency());
    unsigned seed = 12345;            
    float init_density = 0.08f; // parsity at the start
};

class Sim {
public:
    explicit Sim(const SimConfig& cfg);

    // called by worker threads: process [y0, y1) rows for all species
    void step_rows(int y0, int y1);

    // main thread joins the barrier once per frame; completion swaps buffers
    void join_barrier();

    // convert current state -> RGBA8 pixel buffer (species draw priority: higher ID wins)
    void blit_rgba(std::vector<unsigned char>& out_rgba);

    void stop();                      
    bool running() const { return running_.load(); }

    int W() const { return W_; }
    int H() const { return H_; }
    int S() const { return S_; }

    std::barrier<>* barrier = nullptr; // set by main

    void swap_buffers(); // called by barrier completion
    
private:
    int W_, H_, S_, T_ [[maybe_unused]];                // width, height, species, worker threads
    std::atomic<bool> running_{true};

    // double buffers (read from curr, write to next)
    std::vector<std::vector<uint8_t>> layers_curr_; // 0 or 1
    std::vector<std::vector<uint8_t>> layers_next_;

    inline int idx(int x, int y) const { return y*W_ + x; }
    inline uint8_t get(const std::vector<uint8_t>& layer, int x, int y) const {
        return (unsigned)x < (unsigned)W_ && (unsigned)y < (unsigned)H_ ? layer[idx(x,y)] : 0;
    }

    inline uint8_t neighbor_count(const std::vector<uint8_t>& layer, int x, int y) const {
        int c = 0;
        // 8 neighbors
        c += get(layer, x-1,y-1); c += get(layer, x,y-1); c += get(layer, x+1,y-1);
        c += get(layer, x-1,y  );                          c += get(layer, x+1,y  );
        c += get(layer, x-1,y+1); c += get(layer, x,y+1); c += get(layer, x+1,y+1);
        return (uint8_t)c;
    }
};
