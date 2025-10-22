cmake_minimum_required(VERSION 3.15)

# Enable both C and C++
project(MultiSpeciesGameOfLife LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Warnings/opts
if (APPLE)
  add_compile_options(-Wall -Wextra -Wpedantic -O3 -march=native)
else()
  add_compile_options(-Wall -Wextra -Wpedantic -O3)
endif()

find_package(OpenGL REQUIRED)
find_package(glfw3 CONFIG REQUIRED)   # via Homebrew
find_package(TBB CONFIG REQUIRED)

# ---- Vendored GLAD (no find_package(glad)) ----
# Expecting files at:
# lib/glad/include/glad/glad.h
# lib/glad/include/KHR/khrplatform.h
# lib/glad/src/glad.c
add_library(glad lib/glad/src/glad.c)
target_include_directories(glad PUBLIC lib/glad/include)

# ---- Your app ----
add_executable(game src/main.cpp src/sim.cpp)
target_include_directories(game PRIVATE lib/glad/include src)
target_link_libraries(game PRIVATE TBB::tbb glfw OpenGL::GL glad)


if(APPLE)
  target_compile_definitions(game PRIVATE MAC_OS_X_VERSION_MIN_REQUIRED=101500)
endif()
