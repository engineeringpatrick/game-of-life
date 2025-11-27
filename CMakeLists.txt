cmake_minimum_required(VERSION 3.15)
project(MultiSpeciesGameOfLife LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
add_compile_options(-Wall -Wextra -Wpedantic -O3 -march=native)

find_package(OpenGL REQUIRED)
find_package(glfw3 CONFIG REQUIRED)

# ---- GLAD ----
add_library(glad lib/glad/src/glad.c)
target_include_directories(glad PUBLIC lib/glad/include)

# ---- the app ----
add_executable(game src/main.cpp)
target_include_directories(game PRIVATE lib/glad/include src)

# ---- macOS frameworks ----
# GLFW/OpenGL require these on macOS
find_library(COCOA_LIBRARY Cocoa)
find_library(IOKIT_LIBRARY IOKit)
find_library(COREVIDEO_LIBRARY CoreVideo)
find_library(OPENCL_LIBRARY OpenCL)

target_link_libraries(game PRIVATE
    glad
    glfw
    OpenGL::GL
    ${COCOA_LIBRARY}
    ${IOKIT_LIBRARY}
    ${COREVIDEO_LIBRARY}
    ${OPENCL_LIBRARY}
)

target_compile_definitions(game PRIVATE
    GL_SILENCE_DEPRECATION
    MAC_OS_X_VERSION_MIN_REQUIRED=101500
)

configure_file(${CMAKE_SOURCE_DIR}/src/kernels.cl ${CMAKE_SOURCE_DIR}/kernels.cl COPYONLY)
configure_file(${CMAKE_SOURCE_DIR}/src/kernels.cl ${CMAKE_BINARY_DIR}/kernels.cl COPYONLY)