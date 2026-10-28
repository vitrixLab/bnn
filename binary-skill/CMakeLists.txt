cmake_minimum_required(VERSION 3.20)
project(binary_skill CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Portable compiler warning flags
if(MSVC)
    add_compile_options(/W4 /WX /permissive-)
else()
    add_compile_options(-Wall -Wextra -Wpedantic -Wno-missing-field-initializers)
endif()

add_executable(binary_skill_cli main.cpp binary_skill.hpp)
target_include_directories(binary_skill_cli PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

add_executable(benchmark_runner benchmark.cpp binary_skill.hpp)
target_include_directories(benchmark_runner PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

add_executable(benchmark_evidence_validator benchmark_evidence_validator.cpp binary_skill.hpp)
target_include_directories(benchmark_evidence_validator PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})


# Record the configured source revision for benchmark evidence. CI configures after checkout.
find_package(Git QUIET)
if(GIT_FOUND)
    execute_process(COMMAND ${GIT_EXECUTABLE} rev-parse HEAD
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
        OUTPUT_VARIABLE BNN_SOURCE_COMMIT
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET)
    execute_process(COMMAND ${GIT_EXECUTABLE} status --porcelain
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
        OUTPUT_VARIABLE BNN_GIT_STATUS
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET)
    if(BNN_GIT_STATUS STREQUAL "")
        set(BNN_SOURCE_DIRTY "false")
    else()
        set(BNN_SOURCE_DIRTY "true")
    endif()
else()
    set(BNN_SOURCE_COMMIT "unknown")
    set(BNN_SOURCE_DIRTY "unknown")
endif()
