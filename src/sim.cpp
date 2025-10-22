#include "sim.hpp"

Sim::Sim(const SimConfig& cfg)
: W_(cfg.width), H_(cfg.height), S_(cfg.species), T_(cfg.workers)
{
    layers_curr_.resize(S_, std::vector<uint8_t>(W_*H_, 0));
    layers_next_.resize(S_, std::vector<uint8_t>(W_*H_, 0));

    std::mt19937 rng(cfg.seed);
    std::uniform_real_distribution<float> uni(0.0f, 1.0f);

    for (int s = 0; s < S_; ++s) {
        auto& L = layers_curr_[s];
        for (int i = 0; i < W_*H_; ++i) {
            L[i] = (uni(rng) < cfg.init_density) ? 1u : 0u;
        }
    }
}

void Sim::swap_buffers() {
    for (int s = 0; s < S_; ++s) {
        std::swap(layers_curr_[s], layers_next_[s]);
    }
}

void Sim::step_rows(int y0, int y1) {
    while (running()) {
        for (int s = 0; s < S_; ++s) {
            const auto& cur = layers_curr_[s];
            auto& nxt = layers_next_[s];
            for (int y = y0; y < y1; ++y) {
                int base = y * W_;
                for (int x = 0; x < W_; ++x) {
                    uint8_t alive = cur[base + x];
                    uint8_t n = neighbor_count(cur, x, y);
                    uint8_t out = 0;
                    if (alive) {
                        // survives on 2 or 3
                        out = (n == 2 || n == 3) ? 1u : 0u;
                    } else {
                        // birth on exactly 3
                        out = (n == 3) ? 1u : 0u;
                    }
                    nxt[base + x] = out;
                }
            }
        }
        // main thread's completion fun swaps buffers
        barrier->arrive_and_wait();
    }
}

void Sim::join_barrier() {
    barrier->arrive_and_wait();
}

void Sim::blit_rgba(std::vector<unsigned char>& out_rgba) {
    out_rgba.assign(W_*H_*4, 0);

    // simple color look up table -> species 1..S map to distinct colors
    auto color_of = [&](int s)->std::array<unsigned char,4>{
        // deterministic palette
        static const unsigned char P[][3] = {
            {237,  28,  36},{255,127, 39},{255,242,  0},{34,177, 76},
            { 63,  72, 204},{163, 73,164},{ 0, 162,232},{136,  0,21},
            {112,146,190},{181,230, 29}
        };
        const auto& c = P[s % (sizeof(P)/sizeof(P[0]))];
        return {c[0], c[1], c[2], 255};
    };

    // highest species id wins (draw priority)
    for (int y = 0; y < H_; ++y) {
        for (int x = 0; x < W_; ++x) {
            int i = idx(x,y);
            int winner = -1;
            for (int s = 0; s < S_; ++s) {
                // todo: choose one at random
                if (layers_curr_[s][i]) winner = s; // later (higher id) overwrites
            }
            if (winner >= 0) {
                auto rgba = color_of(winner);
                int p = i*4;
                out_rgba[p+0] = rgba[0];
                out_rgba[p+1] = rgba[1];
                out_rgba[p+2] = rgba[2];
                out_rgba[p+3] = 255;
            }
        }
    }
}

void Sim::stop() { running_.store(false); }
