#ifndef GL_SILENCE_DEPRECATION
    #define GL_SILENCE_DEPRECATION
#endif
#define GLFW_INCLUDE_NONE
#include <glad/glad.h>
#include <GLFW/glfw3.h>

#include <OpenGL/OpenGL.h>
#include <OpenCL/opencl.h>

#include <cassert>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>
#include <thread>
#include <vector>

// fps tracking stuff
double cpu_accum = 0.0;
double gpu_accum = 0.0;
double frame_accum = 0.0;
int cpu_frames = 0;
int gpu_frames = 0;
int frame_frames = 0;
auto last_fps_time = std::chrono::steady_clock::now();

using SteadyClock = std::chrono::steady_clock;

// glsl shaders
static const char* VS_SRC = R"(
#version 330 core
layout (location=0) in vec2 inPos;
layout (location=1) in vec2 inUV;
out vec2 uv;
void main(){
  uv = inUV;
  gl_Position = vec4(inPos, 0.0, 1.0);
}
)";

static const char* FS_SRC = R"(
#version 330 core
in vec2 uv;
out vec4 fragColor;
uniform sampler2D uTex;
void main(){
  fragColor = texture(uTex, uv);
}
)";

static GLuint compile_shader(GLenum type, const char* src) {
    GLuint s = glCreateShader(type);
    glShaderSource(s, 1, &src, nullptr);
    glCompileShader(s);
    GLint ok = 0; glGetShaderiv(s, GL_COMPILE_STATUS, &ok);
    if (!ok) {
        char log[4096];
        glGetShaderInfoLog(s, sizeof(log), nullptr, log);
        std::cerr << "[shader] compile error:\n" << log << "\n";
    }
    return s;
}

static GLuint make_program() {
    GLuint vs = compile_shader(GL_VERTEX_SHADER, VS_SRC);
    GLuint fs = compile_shader(GL_FRAGMENT_SHADER, FS_SRC);
    GLuint p = glCreateProgram();
    glAttachShader(p, vs); glAttachShader(p, fs);
    glLinkProgram(p);
    GLint ok = 0; glGetProgramiv(p, GL_LINK_STATUS, &ok);
    if (!ok) {
        char log[4096];
        glGetProgramInfoLog(p, sizeof(log), nullptr, log);
        std::cerr << "[shader] link error:\n" << log << "\n";
    }
    glDeleteShader(vs);
    glDeleteShader(fs);
    return p;
}

static std::string load_file(const char* path) {
    std::ifstream ifs(path, std::ios::binary);
    if (!ifs) return {};
    return std::string((std::istreambuf_iterator<char>(ifs)),
                       std::istreambuf_iterator<char>());
}

static void CheckCLError(cl_int err, const char* where) {
    if (err != CL_SUCCESS) {
        std::cerr << "[OpenCL ERROR] " << where
                  << " failed with code " << err << std::endl;
        std::exit(1);
    }
}

// cpu side helpers, same indexing as kernels.cl and wrapping logic
inline int idx2d_cpp(const int x, const int y, const int W) {
    return y * W + x;
}
inline int idx3d_cpp(const int s, const int x, const int y, const int W, const int H) {
    return s * (W * H) + idx2d_cpp(x, y, W);
}
inline int wrap_or_clamp_cpp(int v, int maxv, int wrap) {
    return wrap ? ((v + maxv) % maxv) : (v < 0 ? 0 : (v >= maxv ? maxv - 1 : v));
}

// multi species update on cpu
static void life_step_cpu(
    const std::vector<unsigned char>& curr,
    std::vector<unsigned char>& next,
    int W, int H, int S,
    int wrap
) {
    for (int y = 0; y < H; ++y) {
        for (int x = 0; x < W; ++x) {
            for (int s = 0; s < S; ++s) {
                int n = 0;
                for (int dy = -1; dy <= 1; ++dy) {
                    for (int dx = -1; dx <= 1; ++dx) {
                        if (dx == 0 && dy == 0) continue;
                        int xx = wrap_or_clamp_cpp(x + dx, W, wrap);
                        int yy = wrap_or_clamp_cpp(y + dy, H, wrap);
                        n += (int)curr[idx3d_cpp(s, xx, yy, W, H)];
                    }
                }
                const int base = idx3d_cpp(s, x, y, W, H);
                const unsigned char alive = curr[base];
                const unsigned char next_alive =
                    (alive ? (n == 2 || n == 3) : (n == 3)) ? (unsigned char)1 : (unsigned char)0;
                next[base] = next_alive;
            }
        }
    }
}

int main() {
    // config
    const int W = 1024, H = 768, S = 6; // width, height, species
    const int wrap = 0;

    // glfw init
    if (!glfwInit()) {
        std::cerr << "glfwInit failed\n";
        return 1;
    }
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR,3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR,3);
    glfwWindowHint(GLFW_OPENGL_PROFILE,GLFW_OPENGL_CORE_PROFILE);
    GLFWwindow* win = glfwCreateWindow(W,H,"GoL OpenCL-OpenGL (CPU+GPU pipeline)",nullptr,nullptr);
    if (!win) {
        std::cerr << "glfwCreateWindow failed\n";
        return 1;
    }
    glfwMakeContextCurrent(win);
    glfwSwapInterval(1);
    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) {
        std::cerr << "gladLoadGLLoader failed\n";
        return 1;
    }

    // rgba8 gl tex
    GLuint tex = 0;
    glGenTextures(1,&tex);
    glBindTexture(GL_TEXTURE_2D, tex);
    glTexImage2D(GL_TEXTURE_2D,0,GL_RGBA8,W,H,0,GL_RGBA,GL_UNSIGNED_BYTE,nullptr);
    glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MIN_FILTER,GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MAG_FILTER,GL_NEAREST);

    // opencl setup
    cl_int err = 0;

    cl_uint numPlat = 0;
    CheckCLError(clGetPlatformIDs(0,nullptr,&numPlat), "clGetPlatformIDs(count)");
    if (numPlat == 0) {
        std::cerr << "No OpenCL platforms found\n";
        return 1;
    }
    std::vector<cl_platform_id> plats(numPlat);
    CheckCLError(clGetPlatformIDs(numPlat, plats.data(), nullptr), "clGetPlatformIDs(list)");
    cl_platform_id plat = plats[0];

    // gpu device
    cl_uint numGpu = 0;
    err = clGetDeviceIDs(plat, CL_DEVICE_TYPE_GPU, 0, nullptr, &numGpu);
    CheckCLError(err, "clGetDeviceIDs(GPU count)");
    if (numGpu == 0) {
        std::cerr << "No GPU OpenCL device found\n";
        return 1;
    }
    std::vector<cl_device_id> devs(numGpu);
    CheckCLError(clGetDeviceIDs(plat, CL_DEVICE_TYPE_GPU, numGpu, devs.data(), nullptr), "clGetDeviceIDs(GPU list)");
    cl_device_id dev = devs[0];

    // gl sharing context
    CGLContextObj cgl_ctx = CGLGetCurrentContext();
    CGLShareGroupObj share = CGLGetShareGroup(cgl_ctx);
    cl_context_properties cps[] = { CL_CONTEXT_PROPERTY_USE_CGL_SHAREGROUP_APPLE, (cl_context_properties)share, 0 };
    cl_context ctx = clCreateContext(cps, 1, &dev, nullptr, nullptr, &err);
    CheckCLError(err, "clCreateContext");

    cl_command_queue q = clCreateCommandQueue(ctx, dev, 0, &err);
    CheckCLError(err, "clCreateCommandQueue");

    // load kernel source
    std::string src = load_file("kernels.cl");
    if (src.empty()) {
        std::cerr << "kernels.cl file not found or empty!\n";
        return 1;
    }
    const char* csrc = src.c_str();
    size_t srclen = src.size();

    cl_program prog = clCreateProgramWithSource(ctx, 1, &csrc, &srclen, &err);
    CheckCLError(err, "clCreateProgramWithSource");
    err = clBuildProgram(prog, 1, &dev, "", nullptr, nullptr);
    if (err != CL_SUCCESS) {
        size_t logsz = 0;
        clGetProgramBuildInfo(prog, dev, CL_PROGRAM_BUILD_LOG, 0, nullptr, &logsz);
        std::string log(logsz, '\0');
        clGetProgramBuildInfo(prog, dev, CL_PROGRAM_BUILD_LOG, logsz, log.data(), nullptr);
        std::cerr << "[GPU build log]\n" << log << "\n";
        return 1;
    }

    cl_kernel kBlit = clCreateKernel(prog, "blit_rgba", &err);
    CheckCLError(err, "clCreateKernel(blit_rgba)");

    // buffers and host grids 
    size_t layerSize = (size_t)W * (size_t)H;
    size_t gridSize = (size_t)S * layerSize;

    // gpu buffer (current state for display)
    cl_mem dCurr = clCreateBuffer(ctx, CL_MEM_READ_WRITE, gridSize, nullptr, &err);
    CheckCLError(err, "clCreateBuffer(dCurr)");

    // host side double buffer for simulation
    std::vector<unsigned char> hBufA(gridSize, 0);
    std::vector<unsigned char> hBufB(gridSize, 0);
    std::vector<unsigned char>* hCurr = &hBufA;
    std::vector<unsigned char>* hNext = &hBufB;

    // sparse random init
    for (size_t i = 0; i < gridSize; ++i)
        (*hCurr)[i] = (rand() % 100 < 12) ? 1u : 0u;

    // upload initial state to gpu
    CheckCLError(
        clEnqueueWriteBuffer(q, dCurr, CL_TRUE, 0, gridSize, hCurr->data(), 0, nullptr, nullptr),
        "clEnqueueWriteBuffer(dCurr init)"
    );

    // gl shared image
    cl_mem clTex = clCreateFromGLTexture(ctx, CL_MEM_WRITE_ONLY, GL_TEXTURE_2D, 0, tex, &err);
    CheckCLError(err, "clCreateFromGLTexture");

    // static kernel args (except seed and dnext which may be reset)
    CheckCLError(clSetKernelArg(kBlit, 1, sizeof(cl_mem), &clTex), "clSetKernelArg(kBlit,tex)");
    CheckCLError(clSetKernelArg(kBlit, 2, sizeof(int), &W), "clSetKernelArg(kBlit,W)");
    CheckCLError(clSetKernelArg(kBlit, 3, sizeof(int), &H), "clSetKernelArg(kBlit,H)");
    CheckCLError(clSetKernelArg(kBlit, 4, sizeof(int), &S), "clSetKernelArg(kBlit,S)");

    // Shader program
    GLuint glProg = make_program();
    glUseProgram(glProg);
    GLint uTexLoc = glGetUniformLocation(glProg, "uTex");
    glUniform1i(uTexLoc, 0);

    // fullscreen quad
    float quad[] = {
        // pos     // uv
        -1.f,-1.f, 0.f,0.f,
         1.f,-1.f, 1.f,0.f,
         1.f, 1.f, 1.f,1.f,
        -1.f, 1.f, 0.f,1.f
    };
    unsigned idx[] = {0,1,2, 2,3,0};

    GLuint vao=0, vbo=0, ebo=0;
    glGenVertexArrays(1, &vao); glBindVertexArray(vao);
    glGenBuffers(1, &vbo); glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof(quad), quad, GL_STATIC_DRAW);
    glGenBuffers(1,&ebo); glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(idx), idx, GL_STATIC_DRAW);

    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0,2,GL_FLOAT,GL_FALSE,4*sizeof(float),(void*)0);
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1,2,GL_FLOAT,GL_FALSE,4*sizeof(float),(void*)(2*sizeof(float)));
    glBindVertexArray(0);

    using clock_t = std::chrono::steady_clock;
    const auto frame_dt = std::chrono::milliseconds(33);
    size_t gsz[2] = { (size_t)W, (size_t)H };
    cl_uint frameSeed = 0;

    while (!glfwWindowShouldClose(win)) {
        auto t0 = clock_t::now();

        // cpu compute next state from hCurr -> hNext
        auto cpu_start = SteadyClock::now();
        life_step_cpu(*hCurr, *hNext, W, H, S, wrap);
        auto cpu_end = SteadyClock::now();
        double cpu_ms = std::chrono::duration<double, std::milli>(cpu_end - cpu_start).count();
        cpu_accum += cpu_ms;
        cpu_frames++;
        
        auto gpu_start = SteadyClock::now();
        // gpu blit current dCurr to gl texture (showing hCurr)
        glFinish(); // ensure gl is done with tex
        CheckCLError(clEnqueueAcquireGLObjects(q, 1, &clTex, 0, nullptr, nullptr), "clEnqueueAcquireGLObjects");

        cl_uint seed = frameSeed++;
        CheckCLError(clSetKernelArg(kBlit, 0, sizeof(cl_mem), &dCurr),
                     "clSetKernelArg(kBlit,dCurr)");
        CheckCLError(clSetKernelArg(kBlit, 5, sizeof(cl_uint), &seed),
                     "clSetKernelArg(kBlit,seed)");

        cl_int errEnq = clEnqueueNDRangeKernel(q, kBlit, 2, nullptr, gsz, nullptr, 0, nullptr, nullptr);
        CheckCLError(errEnq, "clEnqueueNDRangeKernel(kBlit)");

        CheckCLError(clEnqueueReleaseGLObjects(q, 1, &clTex, 0, nullptr, nullptr), "clEnqueueReleaseGLObjects");
        CheckCLError(clFinish(q), "clFinish(q)");

        // upload next state to gpu (for next frame)
        CheckCLError(
            clEnqueueWriteBuffer(q, dCurr, CL_TRUE, 0, gridSize, hNext->data(), 0, nullptr, nullptr),
            "clEnqueueWriteBuffer(dCurr next)"
        );

        // swap host buffers (pipeline is current vs next)
        std::swap(hCurr, hNext);

        // draw quad with the texture
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, tex);

        int fbw, fbh;
        glfwGetFramebufferSize(win, &fbw, &fbh);
        glViewport(0, 0, fbw, fbh);
        glClearColor(0.f,0.f,0.f,1.f);
        glClear(GL_COLOR_BUFFER_BIT);

        glUseProgram(glProg);
        glBindVertexArray(vao);
        glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);

        auto gpu_end = SteadyClock::now();

        double gpu_ms = std::chrono::duration<double, std::milli>(gpu_end - gpu_start).count();
        gpu_accum += gpu_ms;
        gpu_frames++;

        glfwSwapBuffers(win);
        glfwPollEvents();


        auto now = SteadyClock::now();
        double frame_ms = std::chrono::duration<double, std::milli>(now - t0).count();
        frame_accum += frame_ms;
        frame_frames++;

        // print fps every second or so
        if (std::chrono::duration<double>(now - last_fps_time).count() >= 1.0) {
            double cpu_fps   = 1000.0 / (cpu_accum / cpu_frames);
            double gpu_fps   = 1000.0 / (gpu_accum / gpu_frames);
            double frame_fps = 1000.0 / (frame_accum / frame_frames);

            std::cout << "CPU FPS: " << cpu_fps
                    << " | GPU FPS: " << gpu_fps
                    << " | Frame FPS: " << frame_fps
                    << std::endl;

            cpu_accum = gpu_accum = frame_accum = 0.0;
            cpu_frames = gpu_frames = frame_frames = 0;
            last_fps_time = now;
        }
        std::this_thread::sleep_until(t0 + frame_dt); // around 30fps
    }

    return 0;
}
