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

// simple tint per species: cycle R/G/B and accumulate up to white
__kernel void blit_rgba(
    __global const uchar* curr,    // [S * W * H]
    write_only image2d_t  outImg,  // gl-shared image
    int W, int H, int S
){
    const int x = get_global_id(0);
    const int y = get_global_id(1);
    if (x >= W || y >= H) return;

    float r = 0.0f, g = 0.0f, b = 0.0f;
    
    // TODO: SELECT A WINNER AT RANDOM!

    for (int s = 0; s < S; ++s) {
        const uchar alive = curr[idx3d(s, x, y, W, H)];
        if (!alive) continue;

        const int c = s % 3;
        if (c == 0) r += 1.0f;
        else if (c == 1) g += 1.0f;
        else b += 1.0f;
    }

    float brightness = 1.0f; // boost factor
    float4 color = (float4)(clamp(r * brightness, 0.0f, 1.0f),
                            clamp(g * brightness, 0.0f, 1.0f),
                            clamp(b * brightness, 0.0f, 1.0f),
                            1.0f);

    const int2 coord = (int2)(x, y);
    write_imagef(outImg, coord, color);
}
