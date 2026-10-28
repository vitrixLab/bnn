#include <iostream>
#include <string>
#include <vector>
#include <chrono>
#include <algorithm>
#include <fstream>
#include <iomanip>
#include <cctype>
#include "binary_skill.hpp"

struct StatResult {
    std::string operation;
    size_t samples;
    uint64_t min_ns;
    uint64_t max_ns;
    uint64_t median_ns;
    uint64_t p95_ns;
    size_t matches;
};

template <typename Func>
StatResult run_benchmark(const std::string& op_name, size_t iterations, Func&& fn) {
    std::vector<uint64_t> times;
    times.reserve(iterations);
    size_t matches = 0;

    for (size_t i = 0; i < iterations; ++i) {
        auto t0 = std::chrono::high_resolution_clock::now();
        size_t m = fn();
        auto t1 = std::chrono::high_resolution_clock::now();
        times.push_back(std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count());
        matches = m;
    }

    std::sort(times.begin(), times.end());

    StatResult res;
    res.operation = op_name;
    res.samples = iterations;
    res.min_ns = times.front();
    res.max_ns = times.back();
    res.median_ns = times[iterations / 2];
    res.p95_ns = times[static_cast<size_t>(iterations * 0.95)];
    res.matches = matches;
    return res;
}

// Independent correctness oracle: plain case-insensitive substring matching.
// This intentionally does not call BinarySkillReader::search_keyword().
size_t oracle_keyword_count(const bnn::BinarySkillReader& reader, const std::string& query) {
    if (query.empty()) return 0U;
    std::string needle = query;
    std::transform(needle.begin(), needle.end(), needle.begin(),
                   [](unsigned char ch) { return static_cast<char>(std::tolower(ch)); });
    size_t matches = 0U;
    for (size_t i = 0U; i < reader.count(); ++i) {
        const std::string fields[] = {
            std::string(reader.name(i)),
            std::string(reader.description(i)),
            std::string(reader.body(i))
        };
        bool found = false;
        for (std::string text : fields) {
            std::transform(text.begin(), text.end(), text.begin(),
                           [](unsigned char ch) { return static_cast<char>(std::tolower(ch)); });
            if (text.find(needle) != std::string::npos) {
                found = true;
                break;
            }
        }
        if (found) ++matches;
    }
    return matches;
}

void require_oracle_match(const bnn::BinarySkillReader& reader,
                          const std::string& query) {
    const size_t expected = oracle_keyword_count(reader, query);
    const size_t actual = reader.search_keyword(query).size();
    if (actual != expected) {
        throw std::runtime_error("Benchmark correctness failure: search_keyword('" + query +
                                 "') returned " + std::to_string(actual) +
                                 " results; independent oracle expected " +
                                 std::to_string(expected));
    }
}

int main(int argc, char* argv[]) {
    std::string bin_path = "secondbrain.bin";
    if (argc >= 2) bin_path = argv[1];

    std::cout << "Starting BNN Binary-Skill Benchmark Suite (50 iterations per op)...\n";

    bnn::BinarySkillReader reader;
    if (!reader.open(bin_path, true)) {
        bin_path = "binary-skill/" + bin_path;
        if (!reader.open(bin_path, true)) {
            std::cerr << "Error: Could not open " << bin_path << "\n";
            return 1;
        }
    }

    const size_t ITERS = 50; // 50 repetitions for robust median/p95
    // Correctness is checked outside timed regions against an independent oracle.
    require_oracle_match(reader, "nvidia");
    require_oracle_match(reader, "whisper");
    require_oracle_match(reader, "xyznonexistent999");
    std::vector<StatResult> results;

    // 1. Process startup & memory map (Finding 4: explicitly fail if open fails)
    results.push_back(run_benchmark("open_mmap", ITERS, [&]() -> size_t {
        bnn::BinarySkillReader r;
        if (!r.open(bin_path, true) || !r.is_valid()) {
            throw std::runtime_error("Benchmark failure: open_mmap failed to map binary");
        }
        return r.count();
    }));

    // 2. Direct O(1) single-record indexed access (constant time pointer dereference)
    results.push_back(run_benchmark("single_record_access_O1", ITERS, [&]() -> size_t {
        const auto& rec = reader.record(25);
        if (rec.category_id == 0) {
            throw std::runtime_error("Benchmark failure: invalid category_id on record 25");
        }
        return rec.category_id;
    }));

    // 3. Sequential indexed traversal across all 52 records
    results.push_back(run_benchmark("indexed_record_access_all_52", ITERS, [&]() -> size_t {
        size_t dummy = 0;
        for (size_t i = 0; i < reader.count(); ++i) {
            dummy += reader.name(i).size() + reader.flags(i);
        }
        if (dummy == 0) {
            throw std::runtime_error("Benchmark failure: empty records in indexed traversal");
        }
        return dummy;
    }));

    // 4. Exact name lookup (O(N) linear search)
    results.push_back(run_benchmark("exact_name_lookup", ITERS, [&]() -> size_t {
        int idx = reader.find_by_name("omniroute");
        if (idx < 0 || reader.name(idx) != "omniroute") {
            throw std::runtime_error("Benchmark failure: exact_name_lookup failed to find omniroute");
        }
        return 1;
    }));

    // 5. Category lookup (O(N) search)
    results.push_back(run_benchmark("category_lookup_nvidia", ITERS, [&]() -> size_t {
        auto matches = reader.find_by_category(bnn::CAT_NVIDIA_STACK);
        if (matches.empty()) {
            throw std::runtime_error("Benchmark failure: category_lookup_nvidia found 0 matches");
        }
        return matches.size();
    }));

    // 6. Keyword search - common query ("nvidia")
    results.push_back(run_benchmark("search_common_nvidia", ITERS, [&]() -> size_t {
        auto matches = reader.search_keyword("nvidia");
        if (matches.empty()) {
            throw std::runtime_error("Benchmark failure: search_common_nvidia found 0 matches");
        }
        return matches.size();
    }));

    // 7. Keyword search - rare query ("whisper")
    results.push_back(run_benchmark("search_rare_whisper", ITERS, [&]() -> size_t {
        auto matches = reader.search_keyword("whisper");
        return matches.size();
    }));

    // 8. Keyword search - no match query
    results.push_back(run_benchmark("search_nomatch", ITERS, [&]() -> size_t {
        auto matches = reader.search_keyword("xyznonexistent999");
        if (!matches.empty()) {
            throw std::runtime_error("Benchmark failure: search_nomatch unexpectedly found matches");
        }
        return 0;
    }));

    // 9. Whole-file SHA-256 calculation
    results.push_back(run_benchmark("sha256_whole_file", 20, [&]() -> size_t {
        std::string h = reader.compute_file_sha256_hex();
        if (h.size() != 64) {
            throw std::runtime_error("Benchmark failure: sha256_whole_file produced invalid hash length");
        }
        return h.size();
    }));

    // Output human-readable table
    std::cout << "\n========================================================================================\n";
    std::cout << std::left << std::setw(30) << "Operation"
              << std::setw(10) << "Samples"
              << std::setw(14) << "Min (us)"
              << std::setw(14) << "Median (us)"
              << std::setw(14) << "P95 (us)"
              << std::setw(14) << "Max (us)"
              << "Matches\n";
    std::cout << "========================================================================================\n";

    for (const auto& r : results) {
        std::cout << std::left << std::setw(30) << r.operation
                  << std::setw(10) << r.samples
                  << std::setw(14) << (r.min_ns / 1000.0)
                  << std::setw(14) << (r.median_ns / 1000.0)
                  << std::setw(14) << (r.p95_ns / 1000.0)
                  << std::setw(14) << (r.max_ns / 1000.0)
                  << r.matches << "\n";
    }
    std::cout << "========================================================================================\n";

    // Write machine-readable JSON results schema (Section 3.4 compliance + provenance)
    std::ofstream json_out("benchmark_results.json");
    if (json_out.is_open()) {
        std::string safe_bin_path = bin_path;
        std::replace(safe_bin_path.begin(), safe_bin_path.end(), '\\', '/');

        json_out << "{\n"
                 << "  \"schema_version\": 1,\n"
                 << "  \"provenance\": {\n"
                 << "    \"commit\": \"b251258\",\n"
#ifdef _WIN32
                 << "    \"os\": \"Windows x64\",\n"
#elif defined(__APPLE__)
                 << "    \"os\": \"macOS\",\n"
#else
                 << "    \"os\": \"Linux\",\n"
#endif
#if defined(_MSC_VER)
                 << "    \"compiler\": \"MSVC " << _MSC_VER << "\",\n"
#elif defined(__clang__)
                 << "    \"compiler\": \"Clang " << __clang_major__ << "." << __clang_minor__ << "\",\n"
#elif defined(__GNUC__)
                 << "    \"compiler\": \"GCC " << __GNUC__ << "." << __GNUC_MINOR__ << "\",\n"
#else
                 << "    \"compiler\": \"Unknown\",\n"
#endif
#ifdef NDEBUG
                 << "    \"build_type\": \"Release\",\n"
#else
                 << "    \"build_type\": \"Debug\",\n"
#endif
                 << "    \"iterations_per_op\": " << ITERS << "\n"
                 << "  },\n"
                 << "  \"binary_file\": \"" << safe_bin_path << "\",\n"
                 << "  \"file_size_bytes\": " << reader.file_size() << ",\n"
                 << "  \"skill_count\": " << reader.count() << ",\n"
                 << "  \"file_sha256\": \"" << reader.compute_file_sha256_hex() << "\",\n"
                 << "  \"is_memory_mapped\": " << (reader.is_memory_mapped() ? "true" : "false") << ",\n"
                 << "  \"results\": [\n";

        for (size_t i = 0; i < results.size(); ++i) {
            const auto& r = results[i];
            json_out << "    {\n"
                     << "      \"operation\": \"" << r.operation << "\",\n"
                     << "      \"samples\": " << r.samples << ",\n"
                     << "      \"min_ns\": " << r.min_ns << ",\n"
                     << "      \"median_ns\": " << r.median_ns << ",\n"
                     << "      \"p95_ns\": " << r.p95_ns << ",\n"
                     << "      \"max_ns\": " << r.max_ns << ",\n"
                     << "      \"median_us\": " << (r.median_ns / 1000.0) << ",\n"
                     << "      \"matches\": " << r.matches << "\n"
                     << "    }" << (i + 1 < results.size() ? "," : "") << "\n";
        }

        json_out << "  ]\n}\n";
        std::cout << "Wrote machine-readable JSON metrics to benchmark_results.json\n";
    }

    return 0;
}
