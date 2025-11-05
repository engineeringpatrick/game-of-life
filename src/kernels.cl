// OpenCL 1.2

// Wrap helper (toroidal) or clamp. Toggle via a #define or kernel arg.
inline int wrapi(int a, int m) { int r = a % m; return r < 0 ? r + m : r; }

__kernel void life_update(
    __global const uchar* curr,   // S * W * H
    __global uchar*       next,   // S * W * H
    int W, int H, int S,
    int wrap)                     // 0=clamp, 1=wrap
{
    int x = get_global_id(0);
    int y = get_global_id(1);
    if (x >= W || y >= H) return;

    int idx2D = y * W + x;

    // For each species, apply Life rule independently
    for (int s = 0; s < S; ++s) {
        int base = s * (W * H);
        int alive = curr[base + idx2D];
        int n = 0;

        // neighbor loop (unrolled for clarity)
        int xm1 = wrap ? wrapi(x-1,W) : (x>0 ? x-1 : -1);
        int xp1 = wrap ? wrapi(x+1,W) : (x<W-1 ? x+1 : -1);
        int ym1 = wrap ? wrapi(y-1,H) : (y>0 ? y-1 : -1);
        int yp1 = wrap ? wrapi(y+1,H) : (y<H-1 ? y+1 : -1);

        // helper to get with clamp
        #define GET(xx,yy) (((xx)<0 || (yy)<0 || (xx)>=W || (yy)>=H) ? 0 : curr[base + (yy)*W + (xx)])

        if (wrap) {
            n += curr[base + ym1*W + xm1];
            n += curr[base + ym1*W + x];
            n += curr[base + ym1*W + xp1];
            n += curr[base + y   *W + xm1];
            n += curr[base + y   *W + xp1];
            n += curr[base + yp1*W + xm1];
            n += curr[base + yp1*W + x];
            n += curr[base + yp1*W + xp1];
        } else {
            n += GET(xm1, ym1); n += GET(x, ym1); n += GET(xp1, ym1);
            n += GET(xm1, y  );                 n += GET(xp1, y  );
            n += GET(xm1, yp1); n += GET(x, yp1); n += GET(xp1, yp1);
        }
        #undef GET

        uchar out = alive ? (n==2 || n==3) : (n==3);
        next[base + idx2D] = out;
    }
}

// Visualization LUT (8 colors). Expand if you have more species.
constant uchar3 LUT[8] = {
    (uchar3)(237, 28, 36), (uchar3)(255,127,39), (uchar3)(255,242,0),
    (uchar3)(34,177,76),   (uchar3)(63,72,204),  (uchar3)(163,73,164),
    (uchar3)(0,162,232),   (uchar3)(136,0,21)
};

// Writes directly into a GL-shared RGBA8 texture.
__kernel void blit_rgba(
    __global const uchar* curr,   // S * W * H
    write_only image2d_t  outImg, // GL-shared texture (RGBA8)
    int W, int H, int S)
{
    int x = get_global_id(0);
    int y = get_global_id(1);
    if (x >= W || y >= H) return;

    int idx2D = y * W + x;
    int winner = -1;

    // highest species id wins for display
    for (int s = 0; s < S; ++s) {
        int base = s * (W * H);
        if (curr[base + idx2D]) winner = s;
    }

    uchar4 rgba = (uchar4)(0,0,0,255);
    if (winner >= 0) {
        uchar3 c = LUT[winner & 7];
        rgba = (uchar4)(c.x, c.y, c.z, 255);
    }

    int2 coord = (int2)(x, y);
    float4 cf = (float4)((float)rgba.x/255.0f, (float)rgba.y/255.0f, (float)rgba.z/255.0f, 1.0f);
    write_imagef(outImg, coord, cf);
}
