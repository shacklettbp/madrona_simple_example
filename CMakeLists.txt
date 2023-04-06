cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
cmake_policy(VERSION 3.18)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH}
    "${CMAKE_CURRENT_SOURCE_DIR}/external/madrona/cmake")

project(SimpleExample LANGUAGES C CXX)

include(setup)
set(MADRONA_REQUIRE_PYTHON ON)
set(MADRONA_REQUIRE_CUDA ON)
include(dependencies)

add_subdirectory(external)
add_subdirectory(src)
