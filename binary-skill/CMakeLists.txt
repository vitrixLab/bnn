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


include(CTest)
if(BUILD_TESTING)
    add_test(
        NAME cli_verify_rejects_wrong_expected_sha256
        COMMAND binary_skill_cli ${CMAKE_CURRENT_SOURCE_DIR}/secondbrain.bin verify 0000000000000000000000000000000000000000000000000000000000000000
    )
    set_tests_properties(cli_verify_rejects_wrong_expected_sha256 PROPERTIES WILL_FAIL TRUE)
endif()
