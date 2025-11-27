inline int idx2d(const int x, const int y, const int W) {
    return y * W + x;
}

inline int idx3d(const int s, const int x, const int y, const int W, const int H) {
    return s * (W * H) + idx2d(x, y, W);
}

inline int wrap_or_clamp(int v, int maxv, int wrap) {
    // should wrap if we dont want game to go dead soon
    return wrap ? ((v + maxv) % maxv) : clamp(v, 0, maxv - 1);
}

// multi device life update -> updates only species in [sBegin, sEnd)
__kernel void life_update_range(
    __global const uchar* curr, // [S * W * H]
    __global uchar* next, // [S * W * H]
    int W, int H, int S,
    int wrap,
    int sBegin,
    int sEnd
){
    const int x = get_global_id(0);
    const int y = get_global_id(1);
    if (x >= W || y >= H) return;

    // clamp to valid range just in case
    if (sBegin < 0)   sBegin = 0;
    if (sEnd   > S)   sEnd   = S;

    for (int s = sBegin; s < sEnd; ++s) {
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
        const uchar next_alive =
            (alive ? (n == 2 || n == 3) : (n == 3)) ? (uchar)1 : (uchar)0;
        next[base] = next_alive;
    }
}

// per species blit -> cycle rgb channels
__kernel void blit_rgba(
    __global const uchar* curr, // [S * W * H]
    write_only image2d_t outImg, // gl shared image -> this is clTex
    int W, int H, int S
){
    const int x = get_global_id(0);
    const int y = get_global_id(1);
    if (x >= W || y >= H) return;

    float r = 0.0f, g = 0.0f, b = 0.0f;

    for (int s = 0; s < S; ++s) {
        const uchar alive = curr[idx3d(s, x, y, W, H)];
        if (!alive) continue;

        const int c = s % 3;
        if (c == 0) r += 1.0f;
        else if (c == 1) g += 1.0f;
        else  b += 1.0f;
    }

    float4 color = (float4)(
        clamp(r, 0.0f, 1.0f),
        clamp(g, 0.0f, 1.0f),
        clamp(b, 0.0f, 1.0f),
        1.0f
    );

    const int2 coord = (int2)(x, y);
    write_imagef(outImg, coord, color);
}
