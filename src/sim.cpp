#include "sim.hpp"
#include <random>
#include <tbb/parallel_for.h>
#include <tbb/blocked_range.h>

Sim::Sim(const SimConfig& cfg)
: W_(cfg.width), H_(cfg.height), S_(cfg.species)
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

std::mt19937 Sim::gen_{ std::random_device{}() };

void Sim::swap_buffers() {
    for (int s = 0; s < S_; ++s) {
        std::swap(layers_curr_[s], layers_next_[s]);
    }
}

void Sim::update_frame_tbb() {
    for (int s = 0; s < S_; ++s) {
        const auto& cur = layers_curr_[s];
        auto& nxt = layers_next_[s];

        tbb::parallel_for(
            tbb::blocked_range<int>(0, H_), // auto partitioning
            [&](const tbb::blocked_range<int>& r){
                for (int y = r.begin(); y < r.end(); ++y) {
                    int base = y * W_;
                    for (int x = 0; x < W_; ++x) {
                        uint8_t a = cur[base + x];
                        uint8_t n = neighbor_count(cur, x, y);
                        nxt[base + x] = a ? (n == 2 || n == 3) : (n == 3);
                    }
                }
            }
        );
    }
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

    // todo: could parallelize this with tbb easily!
    for (int y = 0; y < H_; ++y) {
        for (int x = 0; x < W_; ++x) {
            int i = idx(x,y);
            // choose winner at random!
            std::vector<int> winners;
            for (int s = 0; s < S_; ++s) {
                if (layers_curr_[s][i]) winners.push_back(s);
            }
            if (!winners.empty()) {
                int winner = winners[std::uniform_int_distribution<std::size_t>(0, winners.size() - 1)(gen_)];
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