// main.cpp — clean final version

#ifndef GL_SILENCE_DEPRECATION
    #define GL_SILENCE_DEPRECATION
#endif
#define GLFW_INCLUDE_NONE
#include <glad/glad.h>
#include <fstream>
#include <cassert>
#include <GLFW/glfw3.h>

#include <OpenGL/OpenGL.h>  
#include <OpenCL/opencl.h> 

#include <barrier>
#include <chrono>
#include <cstdint>
#include <iostream>
#include <thread>
#include <vector>

using SteadyClock = std::chrono::steady_clock;

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
    return std::string((std::istreambuf_iterator<char>(ifs)),
                       std::istreambuf_iterator<char>());
}

auto CheckCLError = [](cl_int err, const char* where) {
    if (err != CL_SUCCESS) {
        std::cerr << "[OpenCL ERROR] " << where
                  << " failed with code " << err << std::endl;
        std::exit(1); // stop immediately so we can see where
    }
};


int main() {
    // config
    const int W = 1024, H = 768, S = 6; // width, height, species
    const int wrap = 0;

    // GLFW/GL init 
    glfwInit();
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR,3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR,3);
    glfwWindowHint(GLFW_OPENGL_PROFILE,GLFW_OPENGL_CORE_PROFILE);
    GLFWwindow* win = glfwCreateWindow(W,H,"GoL OpenCL-OpenGL",nullptr,nullptr);
    glfwMakeContextCurrent(win);
    glfwSwapInterval(1);
    gladLoadGLLoader((GLADloadproc)glfwGetProcAddress);

    // create gl texture (rgba8)
    GLuint tex=0; glGenTextures(1,&tex);
    glBindTexture(GL_TEXTURE_2D, tex);
    glTexImage2D(GL_TEXTURE_2D,0,GL_RGBA8,W,H,0,GL_RGBA,GL_UNSIGNED_BYTE,nullptr);
    glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MIN_FILTER,GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MAG_FILTER,GL_NEAREST);

    cl_int err=0;

    // pick platform/device
    cl_uint numPlat=0; clGetPlatformIDs(0,nullptr,&numPlat);
    std::vector<cl_platform_id> plats(numPlat);
    clGetPlatformIDs(numPlat, plats.data(), nullptr);

    cl_platform_id plat = plats[0]; // pick the first (Apple)
    cl_uint numDev=0; clGetDeviceIDs(plat, CL_DEVICE_TYPE_GPU, 0, nullptr, &numDev);
    std::vector<cl_device_id> devs(numDev);
    clGetDeviceIDs(plat, CL_DEVICE_TYPE_GPU, numDev, devs.data(), nullptr);
    cl_device_id dev = devs[0];

    CGLContextObj cgl_ctx = CGLGetCurrentContext();
    CGLShareGroupObj share = CGLGetShareGroup(cgl_ctx);
    cl_context_properties cps[] = {
        CL_CONTEXT_PROPERTY_USE_CGL_SHAREGROUP_APPLE, (cl_context_properties)share,
        0
    };
    cl_context ctx = clCreateContext(cps, 1, &dev, nullptr, nullptr, &err);
    assert(err==CL_SUCCESS);

    cl_command_queue q = clCreateCommandQueue(ctx, dev, 0, &err);
    assert(err==CL_SUCCESS);

    // program and kernels
    std::string src = load_file("kernels.cl");

    if (src.empty()) {
        std::cerr << "kernels.cl file not found or empty!\n";
        return 1;
    }

    const char* csrc = src.c_str();
    size_t srclen = src.size();
    cl_program clProg = clCreateProgramWithSource(ctx, 1, &csrc, &srclen, &err);
    assert(err==CL_SUCCESS);
    err = clBuildProgram(clProg, 1, &dev, "", nullptr, nullptr);
    if (err != CL_SUCCESS) {
        size_t logsz=0; clGetProgramBuildInfo(clProg, dev, CL_PROGRAM_BUILD_LOG, 0, nullptr, &logsz);
        std::string log(logsz, '\0');
        clGetProgramBuildInfo(clProg, dev, CL_PROGRAM_BUILD_LOG, logsz, log.data(), nullptr);
        std::cerr << log << "\n"; return 1;
    }
    cl_kernel kUpdate = clCreateKernel(clProg, "life_update", &err);
    cl_kernel kBlit   = clCreateKernel(clProg, "blit_rgba",   &err);

    CheckCLError(err, "clCreateKernel(life_update)");
    CheckCLError(err, "clCreateKernel(blit_rgba)");

    // device buffers (curr/next)
    size_t layerSize = W * H;
    size_t gridSize  = S * layerSize;
    cl_mem dCurr = clCreateBuffer(ctx, CL_MEM_READ_WRITE, gridSize, nullptr, &err);
    cl_mem dNext = clCreateBuffer(ctx, CL_MEM_READ_WRITE, gridSize, nullptr, &err);

    // initialize: random (on host then upload) or others
    std::vector<unsigned char> init(gridSize, 0);
    // sparse init:
    for (size_t i=0;i<gridSize;++i) init[i] = (rand()%100 < 12) ? 1u : 0u;
    clEnqueueWriteBuffer(q, dCurr, CL_TRUE, 0, gridSize, init.data(), 0, nullptr, nullptr);

    // make cl_mem from GL texture (no cpu copies)
    cl_mem clTex = clCreateFromGLTexture(ctx, CL_MEM_WRITE_ONLY, GL_TEXTURE_2D, 0, tex, &err);
    assert(err==CL_SUCCESS);

    // set static args
    clSetKernelArg(kUpdate, 0, sizeof(cl_mem), &dCurr);
    clSetKernelArg(kUpdate, 1, sizeof(cl_mem), &dNext);
    clSetKernelArg(kUpdate, 2, sizeof(int), &W);
    clSetKernelArg(kUpdate, 3, sizeof(int), &H);
    clSetKernelArg(kUpdate, 4, sizeof(int), &S);
    clSetKernelArg(kUpdate, 5, sizeof(int), &wrap);

    clSetKernelArg(kBlit, 0, sizeof(cl_mem), &dCurr);
    clSetKernelArg(kBlit, 1, sizeof(cl_mem), &clTex);
    clSetKernelArg(kBlit, 2, sizeof(int), &W);
    clSetKernelArg(kBlit, 3, sizeof(int), &H);
    clSetKernelArg(kBlit, 4, sizeof(int), &S);

    // make shader program
    GLuint glProg = make_program();
    glUseProgram(glProg);
    GLint uTexLoc = glGetUniformLocation(glProg, "uTex");
    glUniform1i(uTexLoc, 0); // sampler uses texture unit 0

    // fullscreen quad (two triangles)
    float quad[] = {
    // pos       // uv
    -1.f,-1.f,   0.f,0.f,
    1.f,-1.f,   1.f,0.f,
    1.f, 1.f,   1.f,1.f,
    -1.f, 1.f,   0.f,1.f
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

    // FRAME LOOP
    using clock_t = std::chrono::steady_clock;
    const auto frame_dt = std::chrono::milliseconds(33);

    size_t gsz[2] = { (size_t)W, (size_t)H };
    cl_uint frameSeed = 0;

    while (!glfwWindowShouldClose(win)) {
        auto t0 = clock_t::now();

        // 1) compute next from curr
        clSetKernelArg(kUpdate, 0, sizeof(cl_mem), &dCurr);
        clSetKernelArg(kUpdate, 1, sizeof(cl_mem), &dNext);
        err = clEnqueueNDRangeKernel(q, kUpdate, 2, nullptr, gsz, nullptr, 0, nullptr, nullptr);
        CheckCLError(err, "clEnqueueNDRangeKernel (kupdate)");

        // 2) swap device buffers (no copy)
        std::swap(dCurr, dNext);

        // 3) fill GL texture on GPU (acquire -> blit -> release)
        glFinish();  // ensure gl is done with the texture before cl uses it
        err = clEnqueueAcquireGLObjects(q, 1, &clTex, 0, nullptr, nullptr);
        CheckCLError(err, "clEnqueueAcquireGLObjects");

        // update seed each frame
        cl_uint seed = frameSeed++;
        clSetKernelArg(kBlit, 0, sizeof(cl_mem), &dCurr); // current is the display source
        clSetKernelArg(kBlit, 5, sizeof(cl_uint), &seed);

        err = clEnqueueNDRangeKernel(q, kBlit, 2, nullptr, gsz, nullptr, 0, nullptr, nullptr);
        CheckCLError(err, "clEnqueueNDRangeKernel (kblit)");

        err = clEnqueueReleaseGLObjects(q, 1, &clTex, 0, nullptr, nullptr);
        CheckCLError(err, "clEnqueueReleaseGLObjects");
        err = clFinish(q); // ensure texture writes done before GL draws
        CheckCLError(err, "clFinish");

        // 4) draw quad with the texture
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, tex);
        
        // bind and draw
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
        
        // auto now = clock_t::now();
        // double dt = std::chrono::duration<double>(now-t0).count();
        // double fps = 1.0/dt;
        // std::cout << fps << std::endl;
        std::this_thread::sleep_until(t0 + frame_dt); // ~30 FPS
    }

    // destroy everything and gl cleanup
    return 0;
}