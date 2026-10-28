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


# Embed the checkout revision used to configure this build. This avoids
# attributing a binary to a different revision based on its runtime working directory.
find_package(Git QUIET)
set(BNN_BUILD_GIT_COMMIT "unknown")
set(BNN_BUILD_SOURCE_DIRTY "unknown")
set(BNN_BENCH_CONTEXT "local" CACHE STRING "Benchmark context: local | ci-smoke | controlled-release")
set_property(CACHE BNN_BENCH_CONTEXT PROPERTY STRINGS local ci-smoke controlled-release)
if(NOT BNN_BENCH_CONTEXT MATCHES "^(local|ci-smoke|controlled-release)$")
    message(FATAL_ERROR "BNN_BENCH_CONTEXT must be local, ci-smoke, or controlled-release")
endif()
set(BNN_BUILD_BENCH_CONTEXT "${BNN_BENCH_CONTEXT}")
if(GIT_FOUND)
    execute_process(
        COMMAND "${GIT_EXECUTABLE}" rev-parse --verify HEAD
        WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
        OUTPUT_VARIABLE _bnn_git_sha
        OUTPUT_STRIP_TRAILING_WHITESPACE
        RESULT_VARIABLE _bnn_git_result
        ERROR_QUIET
    )
    string(LENGTH "${_bnn_git_sha}" _bnn_git_sha_length)
    if(_bnn_git_result EQUAL 0 AND
       _bnn_git_sha_length EQUAL 40 AND
       _bnn_git_sha MATCHES "^[0-9a-fA-F]+$")
        set(BNN_BUILD_GIT_COMMIT "${_bnn_git_sha}")
    endif()

    # Snapshot dirty state at configure time; no shell quoting or runtime popen.
    execute_process(
        COMMAND "${GIT_EXECUTABLE}" status --porcelain -- . ":(exclude)benchmark_results.json"
        WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
        OUTPUT_VARIABLE _bnn_git_status
        OUTPUT_STRIP_TRAILING_WHITESPACE
        RESULT_VARIABLE _bnn_git_status_result
        ERROR_QUIET)
    if(_bnn_git_status_result EQUAL 0)
        if(_bnn_git_status STREQUAL "")
            set(BNN_BUILD_SOURCE_DIRTY "false")
        else()
            set(BNN_BUILD_SOURCE_DIRTY "true")
        endif()
    endif()

    # Reconfigure if HEAD/ref/index or source inputs change. Resolve paths from
    # the repository root because CMAKE_CURRENT_SOURCE_DIR is binary-skill/.
    execute_process(COMMAND "${GIT_EXECUTABLE}" rev-parse --show-toplevel
        WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
        OUTPUT_VARIABLE _bnn_git_root OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET)
    foreach(_bnn_git_item IN ITEMS HEAD index)
        execute_process(COMMAND "${GIT_EXECUTABLE}" rev-parse --git-path "${_bnn_git_item}"
            WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
            OUTPUT_VARIABLE _bnn_git_path OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET)
        if(_bnn_git_path)
            get_filename_component(_bnn_git_path "${_bnn_git_path}" ABSOLUTE BASE_DIR "${_bnn_git_root}")
            if(EXISTS "${_bnn_git_path}")
                set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${_bnn_git_path}")
            endif()
        endif()
    endforeach()
    execute_process(COMMAND "${GIT_EXECUTABLE}" symbolic-ref -q HEAD
        WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
        OUTPUT_VARIABLE _bnn_git_ref OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET)
    if(_bnn_git_ref)
        execute_process(COMMAND "${GIT_EXECUTABLE}" rev-parse --git-path "${_bnn_git_ref}"
            WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
            OUTPUT_VARIABLE _bnn_git_ref_path OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET)
        if(_bnn_git_ref_path)
            get_filename_component(_bnn_git_ref_path "${_bnn_git_ref_path}" ABSOLUTE BASE_DIR "${_bnn_git_root}")
            if(EXISTS "${_bnn_git_ref_path}")
                set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${_bnn_git_ref_path}")
            endif()
        endif()
    endif()
    file(GLOB_RECURSE _bnn_source_inputs CONFIGURE_DEPENDS
        "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/*.hpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/*.h"
        "${CMAKE_CURRENT_SOURCE_DIR}/*.in")
    set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS ${_bnn_source_inputs})
endif()
file(TO_CMAKE_PATH "${CMAKE_CURRENT_SOURCE_DIR}" BNN_BUILD_SOURCE_DIR)
configure_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/build_provenance.hpp.in"
    "${CMAKE_CURRENT_BINARY_DIR}/build_provenance.hpp"
    @ONLY ESCAPE_QUOTES
)

add_executable(binary_skill_cli main.cpp binary_skill.hpp)
target_include_directories(binary_skill_cli PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

add_executable(benchmark_runner benchmark.cpp binary_skill.hpp)
target_include_directories(benchmark_runner PRIVATE ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_BINARY_DIR})

add_executable(benchmark_evidence_validator benchmark_evidence_validator.cpp binary_skill.hpp)
target_include_directories(benchmark_evidence_validator PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
