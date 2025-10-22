// main.cpp — clean final version

#define GL_SILENCE_DEPRECATION
#define GLFW_INCLUDE_NONE
#include <glad/glad.h>
#include <GLFW/glfw3.h>

#include <barrier>
#include <chrono>
#include <cstdint>
#include <iostream>
#include <thread>
#include <vector>

#include "sim.hpp"

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

int main(int argc, char** argv) {
    // ---- config (optional cli species, workers) ----
    SimConfig cfg;
    if (argc > 1) cfg.species = std::clamp(std::atoi(argv[1]), 1, 10);
    if (argc > 2) cfg.workers = std::max(1, std::atoi(argv[2]));
    std::cout << "Species=" << cfg.species << "  Workers=" << cfg.workers << "\n";

    // ---- glfw init / window ----
    if (!glfwInit()) { std::cerr << "glfwInit failed\n"; return 1; }
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GLFW_TRUE);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    GLFWwindow* win = glfwCreateWindow(cfg.width, cfg.height, "Multi-Species GoL (30 FPS)", nullptr, nullptr);
    if (!win) { std::cerr << "GLFW window creation failed\n"; glfwTerminate(); return 1; }

    glfwMakeContextCurrent(win);
    glfwSwapInterval(1); // vsync

    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) {
        std::cerr << "glad initialization failed\n";
        return 1;
    }

    // ---- gl pipeline textured fullscreen quad ----
    GLuint prog = make_program();
    glUseProgram(prog);
    GLint uTexLoc = glGetUniformLocation(prog, "uTex");
    glUniform1i(uTexLoc, 0);

    float quad[] = {
        // pos      // uv
        -1.f,-1.f,  0.f,0.f,
         1.f,-1.f,  1.f,0.f,
         1.f, 1.f,  1.f,1.f,
        -1.f, 1.f,  0.f,1.f
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

    // texture for simulation pixels (RGBA8)
    GLuint tex = 0; glGenTextures(1,&tex);
    glBindTexture(GL_TEXTURE_2D, tex);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, cfg.width, cfg.height, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glBindTexture(GL_TEXTURE_2D, 0);

    // ---- simulation + threads ----
    Sim sim(cfg);

    // barrier with workers + main
    std::barrier<> barrier(cfg.workers + 1);
    sim.barrier = &barrier;

    // launch worker threads with row partitions
    std::vector<std::thread> workers;
    int rowsPer = sim.H() / cfg.workers, extra = sim.H() % cfg.workers, y = 0;
    for (int t = 0; t < cfg.workers; ++t) {
        int take = rowsPer + (t < extra ? 1 : 0);
        int y0 = y, y1 = y + take; y = y1;
        workers.emplace_back([&sim, y0, y1]() {
            sim.step_rows(y0, y1);  // inside: barrier->arrive_and_wait();
        });
    }

    // ---- frame loop @ 30 fps ----
    const auto frame_dt = std::chrono::milliseconds(33);
    std::vector<unsigned char> rgba;
    glClearColor(0.f, 0.f, 0.f, 1.f);

    while (!glfwWindowShouldClose(win)) {
        auto t0 = SteadyClock::now();

        // synchronize with workers they just finished writing 'next'
        barrier.arrive_and_wait();

        // make 'next' -> 'curr' for this frame
        sim.swap_buffers();

        // build rgba from current state and upload
        sim.blit_rgba(rgba);
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, tex);
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, sim.W(), sim.H(), GL_RGBA, GL_UNSIGNED_BYTE, rgba.data());

        // draw
        int fbw, fbh;
        glfwGetFramebufferSize(win, &fbw, &fbh);
        glViewport(0, 0, fbw, fbh);
        glClear(GL_COLOR_BUFFER_BIT);
        glUseProgram(prog);
        glBindVertexArray(vao);
        glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);

        glfwSwapBuffers(win);
        glfwPollEvents();

        // pace to 30 fps
        std::this_thread::sleep_until(t0 + frame_dt);
    }

    // ---- shutdown ----
    sim.stop();
    // one last sync to release workers stuck at the barrier
    barrier.arrive_and_wait();
    for (auto& th : workers) th.join();

    glDeleteTextures(1, &tex);
    glDeleteBuffers(1, &ebo);
    glDeleteBuffers(1, &vbo);
    glDeleteVertexArrays(1, &vao);
    glDeleteProgram(prog);

    glfwDestroyWindow(win);
    glfwTerminate();
    return 0;
}