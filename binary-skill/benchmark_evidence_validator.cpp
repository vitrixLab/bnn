#include "binary_skill.hpp"

#include <cctype>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <iterator>
#include <limits>
#include <regex>
#include <string>

namespace {

std::string extract_object(const std::string& json, const std::string& key) {
    const std::string needle = "\"" + key + "\"";
    const auto start = json.find(needle);
    if (start == std::string::npos) {
        throw std::runtime_error("missing object: " + key);
    }

    const auto colon = json.find(':', start);
    if (colon == std::string::npos) {
        throw std::runtime_error("invalid object: " + key);
    }

    const auto brace_open = json.find('{', colon);
    if (brace_open == std::string::npos) {
        throw std::runtime_error("missing object body: " + key);
    }

    unsigned depth = 0;
    bool in_string = false;
    bool escaped = false;
    for (std::size_t i = brace_open; i < json.size(); ++i) {
        const char ch = json[i];
        if (in_string) {
            if (escaped) {
                escaped = false;
            } else if (ch == '\\') {
                escaped = true;
            } else if (ch == '"') {
                in_string = false;
            }
            continue;
        }

        if (ch == '"') {
            in_string = true;
        } else if (ch == '{') {
            ++depth;
        } else if (ch == '}') {
            --depth;
            if (depth == 0) {
                return json.substr(brace_open, i - brace_open + 1);
            }
        }
    }

    throw std::runtime_error("unterminated object: " + key);
}

std::string read_file(const std::string& path) {
    std::ifstream input(path, std::ios::binary);
    if (!input) {
        throw std::runtime_error("cannot open file: " + path);
    }
    return std::string(std::istreambuf_iterator<char>(input),
                       std::istreambuf_iterator<char>());
}

std::string required_string(const std::string& json, const std::string& key) {
    const std::regex field(std::string("\"") + key + "\"[[:space:]]*:[[:space:]]*\"([^\"]*)\"");
    std::smatch match;
    if (!std::regex_search(json, match, field)) {
        throw std::runtime_error("missing or invalid string field: " + key);
    }
    return match[1].str();
}

uint64_t required_uint(const std::string& json, const std::string& key,
                       bool allow_zero = false) {
    const std::regex field(std::string("\"") + key + "\"[[:space:]]*:[[:space:]]*([0-9]+)");
    std::smatch match;
    if (!std::regex_search(json, match, field)) {
        throw std::runtime_error("missing or invalid integer field: " + key);
    }
    const std::string value = match[1].str();
    size_t consumed = 0;
    const unsigned long long parsed = std::stoull(value, &consumed);
    if (consumed != value.size() || (!allow_zero && parsed == 0U)) {
        throw std::runtime_error("integer field outside allowed range: " + key);
    }
    return static_cast<uint64_t>(parsed);
}

bool has_key(const std::string& object, const std::string& key) {
    const std::regex field(std::string("\"") + key + "\"[[:space:]]*:");
    return std::regex_search(object, field);
}

void validate_results(const std::string& json) {
    const auto results_key = json.find("\"results\"");
    if (results_key == std::string::npos) {
        throw std::runtime_error("missing results array");
    }
    const auto array_start = json.find('[', results_key);
    if (array_start == std::string::npos) {
        throw std::runtime_error("results is not an array");
    }

    size_t position = array_start + 1U;
    size_t result_count = 0U;
    while (position < json.size()) {
        while (position < json.size() &&
               (std::isspace(static_cast<unsigned char>(json[position])) ||
                json[position] == ',')) {
            ++position;
        }
        if (position >= json.size()) break;
        if (json[position] == ']') break;
        if (json[position] != '{') {
            throw std::runtime_error("invalid results array entry");
        }

        const size_t object_start = position;
        bool in_string = false;
        bool escaped = false;
        unsigned depth = 0U;
        for (; position < json.size(); ++position) {
            const char ch = json[position];
            if (in_string) {
                if (escaped) escaped = false;
                else if (ch == '\\') escaped = true;
                else if (ch == '"') in_string = false;
                continue;
            }
            if (ch == '"') in_string = true;
            else if (ch == '{') ++depth;
            else if (ch == '}' && --depth == 0U) {
                ++position;
                break;
            }
        }
        if (depth != 0U || in_string) {
            throw std::runtime_error("unterminated result object");
        }

        const std::string object = json.substr(object_start, position - object_start);
        (void)required_string(object, "operation");
        (void)required_uint(object, "samples");
        const uint64_t min_ns = required_uint(object, "min_ns", true);
        const uint64_t median_ns = required_uint(object, "median_ns", true);
        const uint64_t p95_ns = required_uint(object, "p95_ns", true);
        const uint64_t max_ns = required_uint(object, "max_ns", true);
        if (!(min_ns <= median_ns && median_ns <= p95_ns && p95_ns <= max_ns)) {
            throw std::runtime_error("timing quantiles are out of order");
        }
        ++result_count;
    }

    if (result_count == 0U) {
        throw std::runtime_error("results array must not be empty");
    }
}

void validate_report(const std::string& report, const std::string& binary_path) {
    const std::string provenance = extract_object(report, "provenance");

    if (required_uint(report, "schema_version") != 1U) {
        throw std::runtime_error("unsupported schema_version");
    }

    (void)required_string(provenance, "os");
    (void)required_string(provenance, "compiler");
    const std::string build_type = required_string(provenance, "build_type");
    (void)required_uint(provenance, "iterations_per_op");
    (void)required_uint(report, "skill_count");

    const uint64_t expected_size = required_uint(report, "file_size_bytes", true);
    const std::string expected_hash = required_string(report, "file_sha256");
    if (!std::regex_match(expected_hash, std::regex("[0-9a-f]{64}"))) {
        throw std::runtime_error("file_sha256 must be 64 lowercase hexadecimal characters");
    }

    const std::string binary = read_file(binary_path);
    if (binary.size() != expected_size) {
        throw std::runtime_error("binary file size does not match benchmark report");
    }
    const std::string actual_hash = bnn::Sha256::hash_hex(binary);
    if (actual_hash != expected_hash) {
        throw std::runtime_error("binary SHA-256 does not match benchmark report");
    }

    validate_results(report);

    const std::string commit = has_key(provenance, "commit")
        ? required_string(provenance, "commit") : std::string();
    const std::string dirty = has_key(provenance, "source_dirty")
        ? required_string(provenance, "source_dirty") : "unknown";
    const std::string context = has_key(provenance, "bench_context")
        ? required_string(provenance, "bench_context") : "unknown";

    if (build_type == "Release" &&
        std::regex_match(commit, std::regex("[0-9a-f]{40}")) &&
        dirty == "false" && context == "controlled-release") {
        std::cout << "Controlled-release provenance contract PASS\n";
    } else {
        std::cout << "NOTICE: validated as historical evidence; not certified as controlled-release provenance.\n";
    }
}

}  // namespace

int main(int argc, char* argv[]) {
    if (argc != 3) {
        std::cerr << "Usage: benchmark_evidence_validator <report.json> <binary-file>\n";
        return 2;
    }

    try {
        const std::string report = read_file(argv[1]);
        validate_report(report, argv[2]);
        std::cout << "PASS: benchmark report contract and binary integrity\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
}

