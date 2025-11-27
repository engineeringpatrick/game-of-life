// index helpers
inline int idx2d(const int x, const int y, const int W) {
    return y * W + x;
}
inline int idx3d(const int s, const int x, const int y, const int W, const int H) {
    return s * (W * H) + idx2d(x, y, W);
}

// simple 32bit hash (for random winner)
inline uint hash_u32(uint x) {
    x ^= x >> 16;
    x *= 0x7feb352dU;
    x ^= x >> 15;
    x *= 0x846ca68bU;
    x ^= x >> 16;
    return x;
}

// [ick a random alive species per pixel and color it
__kernel void blit_rgba(
    __global const uchar* curr,    // [S * W * H]
    write_only image2d_t outImg,  // gl shared image
    int W, int H, int S,
    uint seed                      // per-frame seed
){
    const int x = get_global_id(0);
    const int y = get_global_id(1);
    if (x >= W || y >= H) return;

    // count alive species at this pixel
    int alive_count = 0;
    for (int s = 0; s < S; ++s) {
        const uchar alive = curr[idx3d(s, x, y, W, H)];
        if (alive) ++alive_count;
    }

    float4 color;

    if (alive_count == 0) {
        color = (float4)(0.0f, 0.0f, 0.0f, 1.0f);
    } else {
        // deterministic random index in [0, alive_count)
        uint base_hash = (uint)x;
        base_hash = base_hash * 73856093u ^ (uint)y * 19349663u;
        base_hash ^= seed;
        uint rnd = hash_u32(base_hash);
        uint win_index = rnd % (uint)alive_count;

        // map this index to the actual species id
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

        float r = 0.0f, g = 0.0f, b = 0.0f;
        if (winner_s >= 0) {
            const int c = winner_s % 3;
            if (c == 0) r = 1.0f;
            else if (c == 1) g = 1.0f;
            else b = 1.0f;
        }
        color = (float4)(r, g, b, 1.0f);
    }

    const int2 coord = (int2)(x, y);
    write_imagef(outImg, coord, color);
}
