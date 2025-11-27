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
#include <fstream>
#include <iostream>
#include <thread>
#include <vector>

using SteadyClock = std::chrono::steady_clock;

static const char* VS_SRC = R"(
#version 330 core
layout (location=0) in vec2 inPos;
layout (location=1) in vec2 inUV;
out vec2 uv;
void main() {
  uv = inUV;
  gl_Position = vec4(inPos, 0.0, 1.0);
}
)";

static const char* FS_SRC = R"(
#version 330 core
in vec2 uv;
out vec4 fragColor;
uniform sampler2D uTex;
void main() {
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
    // uhh using this function for debugging coz opencl sometimes stops working for random reasons (a single crucial instruction missing)
    if (err != CL_SUCCESS) {
        std::cerr << "[OpenCL ERROR] " << where
                  << " failed with code " << err << std::endl;
        std::exit(1);
    }
}

int main() {
    // config
    const int W = 1024, H = 768, S = 6;
    const int wrap = 0;

    // glfw init
    if (!glfwInit()) {
        std::cerr << "glfwInit failed\n";
        return 1;
    }
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR,3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR,3);
    glfwWindowHint(GLFW_OPENGL_PROFILE,GLFW_OPENGL_CORE_PROFILE);
    GLFWwindow* win = glfwCreateWindow(W,H,"GoL OpenCL CPU+GPU (shared buffer)",nullptr,nullptr);
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

    // create GL texture
    GLuint tex = 0;
    glGenTextures(1,&tex);
    glBindTexture(GL_TEXTURE_2D, tex);
    glTexImage2D(GL_TEXTURE_2D,0,GL_RGBA8,W,H,0,GL_RGBA,GL_UNSIGNED_BYTE,nullptr);
    glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MIN_FILTER,GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MAG_FILTER,GL_NEAREST);

    cl_int err = 0;

    // platforms
    cl_uint numPlat = 0;
    CheckCLError(clGetPlatformIDs(0,nullptr,&numPlat), "clGetPlatformIDs(count)");
    if (numPlat == 0) {
        std::cerr << "No OpenCL platforms found\n";
        return 1;
    }
    std::vector<cl_platform_id> plats(numPlat);
    CheckCLError(clGetPlatformIDs(numPlat, plats.data(), nullptr), "clGetPlatformIDs(list)");
    cl_platform_id plat = plats[0];

    // cpu device
    cl_uint numCpu = 0;
    err = clGetDeviceIDs(plat, CL_DEVICE_TYPE_CPU, 0, nullptr, &numCpu);
    if (err != CL_SUCCESS || numCpu == 0) {
        std::cerr << "No CPU OpenCL device found on this platform (apple silicon)\n";
        return 1;
    }

    std::vector<cl_device_id> cpuDevs(numCpu);
    CheckCLError(clGetDeviceIDs(plat, CL_DEVICE_TYPE_CPU, numCpu, cpuDevs.data(), nullptr),
                 "clGetDeviceIDs(CPU)");
    cl_device_id cpuDev = cpuDevs[0];

    // gpu device
    cl_uint numGpu = 0;
    err = clGetDeviceIDs(plat, CL_DEVICE_TYPE_GPU, 0, nullptr, &numGpu);
    CheckCLError(err, "clGetDeviceIDs(GPU count)");
    if (numGpu == 0) {
        std::cerr << "No GPU OpenCL device found\n";
        return 1;
    }
    std::vector<cl_device_id> gpuDevs(numGpu);
    CheckCLError(clGetDeviceIDs(plat, CL_DEVICE_TYPE_GPU, numGpu, gpuDevs.data(), nullptr),
                 "clGetDeviceIDs(GPU)");
    cl_device_id gpuDev = gpuDevs[0];

    // combined device list (cpu+gpu) in one context
    cl_device_id devices[2] = { cpuDev, gpuDev };

    // gl sharing context
    CGLContextObj cgl_ctx = CGLGetCurrentContext();
    CGLShareGroupObj share = CGLGetShareGroup(cgl_ctx);
    cl_context_properties cps[] = {
        CL_CONTEXT_PROPERTY_USE_CGL_SHAREGROUP_APPLE, (cl_context_properties)share,
        0
    };

    cl_context ctx = clCreateContext(cps, 2, devices, nullptr, nullptr, &err);
    CheckCLError(err, "clCreateContext");

    // command queues for CPU and GPU
    cl_command_queue qCPU = clCreateCommandQueue(ctx, cpuDev, 0, &err);
    CheckCLError(err, "clCreateCommandQueue(cpu)");

    cl_command_queue qGPU = clCreateCommandQueue(ctx, gpuDev, 0, &err);
    CheckCLError(err, "clCreateCommandQueue(gpu)");

    // program
    std::string src = load_file("kernels.cl");
    if (src.empty()) {
        std::cerr << "kernels.cl file not found or empty!\n";
        return 1;
    }
    const char* csrc = src.c_str();
    size_t srclen = src.size();

    cl_program prog = clCreateProgramWithSource(ctx, 1, &csrc, &srclen, &err);
    CheckCLError(err, "clCreateProgramWithSource");
    err = clBuildProgram(prog, 2, devices, "", nullptr, nullptr);
    if (err != CL_SUCCESS) {
        size_t logsz = 0;
        clGetProgramBuildInfo(prog, gpuDev, CL_PROGRAM_BUILD_LOG, 0, nullptr, &logsz);
        std::string log(logsz, '\0');
        clGetProgramBuildInfo(prog, gpuDev, CL_PROGRAM_BUILD_LOG, logsz, log.data(), nullptr);
        std::cerr << "[Build log GPU]\n" << log << "\n";
        clGetProgramBuildInfo(prog, cpuDev, CL_PROGRAM_BUILD_LOG, 0, nullptr, &logsz);
        log.assign(logsz, '\0');
        clGetProgramBuildInfo(prog, cpuDev, CL_PROGRAM_BUILD_LOG, logsz, log.data(), nullptr);
        std::cerr << "[Build log CPU]\n" << log << "\n";
        return 1;
    }

    cl_kernel kUpdateRange = clCreateKernel(prog, "life_update_range", &err);
    CheckCLError(err, "clCreateKernel(life_update_range)");
    cl_kernel kBlit = clCreateKernel(prog, "blit_rgba", &err);
    CheckCLError(err, "clCreateKernel(blit_rgba)");

    // shared grid buffers
    size_t layerSize = (size_t)W * (size_t)H;
    size_t gridSize  = (size_t)S * layerSize;
    cl_mem dCurr = clCreateBuffer(ctx, CL_MEM_READ_WRITE, gridSize, nullptr, &err);
    CheckCLError(err, "clCreateBuffer(dCurr)");
    cl_mem dNext = clCreateBuffer(ctx, CL_MEM_READ_WRITE, gridSize, nullptr, &err);
    CheckCLError(err, "clCreateBuffer(dNext)");

    // host init
    std::vector<unsigned char> init(gridSize, 0);
    for (size_t i = 0; i < gridSize; ++i)
        init[i] = (rand() % 100 < 12) ? 1u : 0u;

    CheckCLError(
        clEnqueueWriteBuffer(qGPU, dCurr, CL_TRUE, 0, gridSize, init.data(), 0, nullptr, nullptr),
        "clEnqueueWriteBuffer(dCurr init)"
    );

    // GL-shared texture as OpenCL image
    cl_mem clTex = clCreateFromGLTexture(ctx, CL_MEM_WRITE_ONLY, GL_TEXTURE_2D, 0, tex, &err);
    CheckCLError(err, "clCreateFromGLTexture");

    // static args for blit
    CheckCLError(clSetKernelArg(kBlit, 0, sizeof(cl_mem), &dCurr), "kBlit arg0");
    CheckCLError(clSetKernelArg(kBlit, 1, sizeof(cl_mem), &clTex), "kBlit arg1");
    CheckCLError(clSetKernelArg(kBlit, 2, sizeof(int), &W), "kBlit arg2");
    CheckCLError(clSetKernelArg(kBlit, 3, sizeof(int), &H), "kBlit arg3");
    CheckCLError(clSetKernelArg(kBlit, 4, sizeof(int), &S), "kBlit arg4");

    // GL shader
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

    GLuint vao=0,vbo=0,ebo=0;
    glGenVertexArrays(1,&vao); glBindVertexArray(vao);
    glGenBuffers(1,&vbo); glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof(quad), quad, GL_STATIC_DRAW);
    glGenBuffers(1,&ebo); glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(idx), idx, GL_STATIC_DRAW);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0,2,GL_FLOAT,GL_FALSE,4*sizeof(float),(void*)0);
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1,2,GL_FLOAT,GL_FALSE,4*sizeof(float),(void*)(2*sizeof(float)));
    glBindVertexArray(0);

    // cpu does first half and gpu does second half
    const int cpuSpecies = S / 2;
    const int gpuSpecies = S - cpuSpecies;

    using clock_t = std::chrono::steady_clock;
    const auto frame_dt = std::chrono::milliseconds(33);
    size_t gsz[2] = { (size_t)W, (size_t)H };

    while (!glfwWindowShouldClose(win)) {
        auto t0 = clock_t::now();

        // cpu device update species [0, cpuSpecies)
        CheckCLError(clSetKernelArg(kUpdateRange, 0, sizeof(cl_mem), &dCurr), "kUpdateRange arg0");
        CheckCLError(clSetKernelArg(kUpdateRange, 1, sizeof(cl_mem), &dNext), "kUpdateRange arg1");
        CheckCLError(clSetKernelArg(kUpdateRange, 2, sizeof(int), &W), "kUpdateRange arg2");
        CheckCLError(clSetKernelArg(kUpdateRange, 3, sizeof(int), &H), "kUpdateRange arg3");
        CheckCLError(clSetKernelArg(kUpdateRange, 4, sizeof(int), &S), "kUpdateRange arg4");
        CheckCLError(clSetKernelArg(kUpdateRange, 5, sizeof(int), &wrap), "kUpdateRange arg5");

        int sBeginCPU = 0;
        int sEndCPU = cpuSpecies;
        CheckCLError(clSetKernelArg(kUpdateRange, 6, sizeof(int), &sBeginCPU), "kUpdateRange arg6");
        CheckCLError(clSetKernelArg(kUpdateRange, 7, sizeof(int), &sEndCPU), "kUpdateRange arg7");

        CheckCLError(
            clEnqueueNDRangeKernel(qCPU, kUpdateRange, 2, nullptr, gsz, nullptr,
                                   0, nullptr, nullptr),
            "clEnqueueNDRangeKernel(kUpdateRange CPU)"
        );

        // gpu device update species [cpuSpecies, S)
        sBeginCPU = cpuSpecies;
        sEndCPU   = S;
        CheckCLError(clSetKernelArg(kUpdateRange, 6, sizeof(int), &sBeginCPU), "kUpdateRange arg6 GPU");
        CheckCLError(clSetKernelArg(kUpdateRange, 7, sizeof(int), &sEndCPU), "kUpdateRange arg7 GPU");

        CheckCLError(clEnqueueNDRangeKernel(qGPU, kUpdateRange, 2, nullptr, gsz, nullptr, 0, nullptr, nullptr), "clEnqueueNDRangeKernel(kUpdateRange GPU)");

        // wait for both devices to finish writing dNext
        CheckCLError(clFinish(qCPU), "clFinish(qCPU)");
        CheckCLError(clFinish(qGPU), "clFinish(qGPU)");

        // swap current / next
        std::swap(dCurr, dNext);

        // gpu blit into gl texture
        glFinish();
        CheckCLError(clEnqueueAcquireGLObjects(qGPU, 1, &clTex, 0, nullptr, nullptr),
                     "clEnqueueAcquireGLObjects");

        CheckCLError(clSetKernelArg(kBlit, 0, sizeof(cl_mem), &dCurr), "kBlit arg0 curr");
        CheckCLError(
            clEnqueueNDRangeKernel(qGPU, kBlit, 2, nullptr, gsz, nullptr, 0, nullptr, nullptr),
            "clEnqueueNDRangeKernel(kBlit)"
        );

        CheckCLError(clEnqueueReleaseGLObjects(qGPU, 1, &clTex, 0, nullptr, nullptr),
                     "clEnqueueReleaseGLObjects");
        CheckCLError(clFinish(qGPU), "clFinish(qGPU after blit)");

        // final step is drawing quad
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
        glfwSwapBuffers(win);
        glfwPollEvents();

        std::this_thread::sleep_until(t0 + frame_dt);
    }

    return 0;
}
