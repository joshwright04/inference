cmake_minimum_required(VERSION 3.24)
project(inference)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include(FetchContent)
FetchContent_Declare(
        googletest
        URL https://github.com/google/googletest/archive/03597a01ee50ed33e9dfd640b249b4be3799d395.zip
)
# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

add_library(inference_core
        src/tensor_shape.cpp
        src/tensor.cpp
        backends/cpu/cpu_ops.cpp)
target_include_directories(inference_core PUBLIC include)

add_executable(inference main.cpp)
target_link_libraries(inference PRIVATE inference_core)

enable_testing()



add_executable(tensor_tests tests/tensor_shape_tests.cpp
        tests/tensor_tests.cpp
        tests/cpu_ops_tests.cpp)
target_link_libraries(tensor_tests PRIVATE inference_core GTest::gtest_main)

include(GoogleTest)

gtest_discover_tests(tensor_tests)
