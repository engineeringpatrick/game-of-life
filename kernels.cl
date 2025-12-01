// kernels.cl

// index helpers
inline int idx2d(const int x, const int y, const int W) {
    return y * W + x;
}
inline int idx3d(const int s, const int x, const int y, const int W, const int H) {
    return s * (W * H) + idx2d(x, y, W);
}

inline int wrap_or_clamp(int v, int maxv, int wrap) {
    return wrap ? ((v + maxv) % maxv) : clamp(v, 0, maxv - 1);
}

// simple 32-bit hash
inline uint hash_u32(uint x) {
    x ^= x >> 16;
    x *= 0x7feb352dU;
    x ^= x >> 15;
    x *= 0x846ca68bU;
    x ^= x >> 16;
    return x;
}

// conway step on S layers (each layer independent)
__kernel void life_update(
    __global const uchar* curr,   // [S * W * H]
    __global uchar*       next,   // [S * W * H]
    int W, int H, int S,
    int wrap                     
){
    const int x = get_global_id(0);
    const int y = get_global_id(1);
    if (x >= W || y >= H) return;

    // for each species/layer do a life step
    for (int s = 0; s < S; ++s) {
        int n = 0;
        // count 8 neighbors on same layer
        for (int dy = -1; dy <= 1; ++dy) {
            for (int dx = -1; dx <= 1; ++dx) {
                if (dx == 0 && dy == 0) continue;
                int xx = wrap_or_clamp(x + dx, W, wrap);
                int yy = wrap_or_clamp(y + dy, H, wrap);
                n += (int)curr[idx3d(s, xx, yy, W, H)];
            }
        }

        const int base = idx3d(s, x, y, W, H);
        const uchar alive = curr[base];
        const uchar next_alive = (alive ? (n == 2 || n == 3) : (n == 3)) ? (uchar)1 : (uchar)0;
        next[base] = next_alive;
    }
}

// simple tint per species -> cycle r/g/b and choose a random alive species as winner
__kernel void blit_rgba(
    __global const uchar* curr,    // [S * W * H]
    write_only image2d_t  outImg,  // gl-shared image
    int W, int H, int S,
    uint seed                      // per-frame random seed
){
    const int x = get_global_id(0);
    const int y = get_global_id(1);
    if (x >= W || y >= H) return;

    // count all alive scells
    int alive_count = 0;
    for (int s = 0; s < S; ++s) {
        const uchar alive = curr[idx3d(s, x, y, W, H)];
        if (alive) {
            ++alive_count;
        }
    }

    float4 color;

    if (alive_count == 0) {
        // no one alive, set as black
        color = (float4)(0.0f, 0.0f, 0.0f, 1.0f);
    } else {
        // deterministic pseudo-random winner among the alive ones
        uint base_hash = (uint)x;
        base_hash = base_hash * 73856093u ^ (uint)y * 19349663u;
        base_hash ^= seed;
        uint rnd = hash_u32(base_hash);
        uint win_index = rnd % (uint)alive_count;

        int winner_s = -1;
        int seen = 0;
        for (int s = 0; s < S; ++s) {
            const uchar alive = curr[idx3d(s, x, y, W, H)];
            if (!alive) continue;
            if (seen == (int)win_index) {
                winner_s = s;
                break;
            }
            ++seen;
        }

        // map winner species to RGB (cycle the three channels)
        float r = 0.0f, g = 0.0f, b = 0.0f;
        if (winner_s >= 0) {
            const int c = winner_s % 3;
            if (c == 0)      r = 1.0f;
            else if (c == 1) g = 1.0f;
            else             b = 1.0f;
        }

        color = (float4)(r, g, b, 1.0f);
    }

    const int2 coord = (int2)(x, y);
    write_imagef(outImg, coord, color);
}
