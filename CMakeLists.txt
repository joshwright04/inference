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

add_executable(inference main.cpp)

enable_testing()

add_executable(simple_test simple_test.cpp)
target_link_libraries(simple_test PRIVATE GTest::gtest_main)

include(GoogleTest)
gtest_discover_tests(simple_test)
