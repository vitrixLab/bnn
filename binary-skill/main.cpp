#include <iostream>
#include <string>
#include <vector>
#include <array>
#include <chrono>
#include "binary_skill.hpp"

void print_usage(const char* prog) {
    std::cout << "BNN Binary-Skill CLI & Runtime (C++20 Zero-FPU Knowledge Engine)\n"
              << "Usage: " << prog << " [binary_path] [command] [args...]\n\n"
              << "Commands:\n"
              << "  info             Display binary header and memory-mapping metadata\n"
              << "  list             List all distilled skills in tabular layout\n"
              << "  category <name>  List skills by category (e.g. NVIDIA_STACK)\n"
              << "  search <query>   Perform keyword lookup (sub-millisecond linear scan)\n"
              << "  show <name>      Inspect skill definition, canonical SHA-256, and body\n"
              << "  verify           Cryptographic and structural layout verification\n"
              << "  test-compile     Run pure C++ compiler self-test roundtrip\n";
}

int main(int argc, char* argv[]) {
    std::string bin_path = "secondbrain.bin";
    std::string command = "info";
    std::string arg = "";

    if (argc >= 2) {
        std::string first = argv[1];
        if (first == "-h" || first == "--help") {
            print_usage(argv[0]);
            return 0;
        }
        if (first == "info" || first == "list" || first == "verify" || first == "test-compile" ||
            first == "search" || first == "show" || first == "category") {
            command = first;
            if (argc >= 3) arg = argv[2];
        } else {
            bin_path = first;
            if (argc >= 3) command = argv[2];
            if (argc >= 4) arg = argv[3];
        }
    }

    if (command == "test-compile") {
        std::cout << "Running pure C++ binary skill compiler self-test...\n";

        // Gate 1: NIST FIPS 180-4 known-answer test (3 vectors)
        if (!bnn::Sha256::self_test()) {
            std::cerr << "FAIL: SHA-256 NIST known-answer self-test FAILED\n";
            return 1;
        }
        std::cout << "  [SHA-256] NIST FIPS 180-4 known-answer vectors: PASS\n";

        // Gate 2: compiler round-trip
        bnn::BinarySkillCompiler compiler;

        // Compute correct payload SHA-256 for each test skill so verify_record_digests() passes
        auto make_sha = [](const char* desc, const char* body) {
            uint8_t sha[32];
            bnn::Sha256 h;
            h.update(desc, std::strlen(desc));
            h.update(body, std::strlen(body));
            h.finalize(sha);
            std::array<uint8_t, 32> arr;
            std::memcpy(arr.data(), sha, 32);
            return arr;
        };

        const char* d0 = "Quantized INT4/BIN1 inference acceleration.";
        const char* b0 = "# BNN Quantization\nRuns XNOR/popcount kernels without FPU floating-point conversion.";
        auto sha0 = make_sha(d0, b0);
        compiler.add_skill("bnn-quant", bnn::CAT_NVIDIA_STACK, 0x02, d0, b0, sha0.data());

        const char* d1 = "Zero-copy memory mapped state cache.";
        const char* b1 = "# BNN Memory Cache\nDirect buffer reading bypassing tokenization layers.";
        auto sha1 = make_sha(d1, b1);
        compiler.add_skill("bnn-memory", bnn::CAT_CORE_ENGINEERING, 0x04, d1, b1, sha1.data());

        std::string test_bin = "test_roundtrip.bin";
        if (!compiler.compile_to_file(test_bin)) {
            std::cerr << "FAIL: compilation failed\n";
            return 1;
        }

        bnn::BinarySkillReader test_reader;
        if (!test_reader.open(test_bin, true)) {
            std::cerr << "FAIL: reader could not open compiled test binary\n";
            std::remove(test_bin.c_str());
            return 1;
        }

        std::string err;
        if (!test_reader.verify_all(&err)) {
            std::cerr << "FAIL: layout verification failed: " << err << "\n";
            std::remove(test_bin.c_str());
            return 1;
        }

        // Gate 3: per-record digest verification (closes stored-vs-computed gap)
        if (!test_reader.verify_record_digests(&err)) {
            std::cerr << "FAIL: per-record SHA-256 digest verification failed: " << err << "\n";
            std::remove(test_bin.c_str());
            return 1;
        }

        if (test_reader.count() != 2 || test_reader.name(0) != "bnn-quant") {
            std::cerr << "FAIL: verification mismatch\n";
            std::remove(test_bin.c_str());
            return 1;
        }

        std::cout << "  [Compiler] round-trip compile + layout verify: PASS\n";
        std::cout << "  [Digests]  per-record SHA-256 stored == computed: PASS\n";

        // Gate 4: Adversarial & failure-path verification
        // 4a. Tampered payload detection (flip bits in string pool)
        {
            std::string tampered_bin = "test_tampered.bin";
            std::ifstream src(test_bin, std::ios::binary);
            std::vector<char> buf((std::istreambuf_iterator<char>(src)), std::istreambuf_iterator<char>());
            src.close();
            if (buf.size() <= 100U) {
                std::cerr << "FAIL: Test fixture is too small for payload tampering\n";
                std::remove(test_bin.c_str());
                return 1;
            }

            // Flip byte in payload area
            buf.back() ^= static_cast<char>(0xFF);
            std::ofstream dst(tampered_bin, std::ios::binary);
            dst.write(buf.data(), static_cast<std::streamsize>(buf.size()));
            dst.close();

            bnn::BinarySkillReader tamp_reader;
            // The tampered fixture MUST open successfully (valid structure)
            if (!tamp_reader.open(tampered_bin, true)) {
                std::cerr << "FAIL: Tampered fixture failed to open (must open successfully to prove digest detection)!\n";
                std::remove(tampered_bin.c_str());
                std::remove(test_bin.c_str());
                return 1;
            }

            // Record-digest verification MUST fail specifically because payload was altered
            std::string t_err;
            if (tamp_reader.verify_record_digests(&t_err)) {
                std::cerr << "FAIL: Tampered payload was NOT detected by verify_record_digests!\n";
                tamp_reader.close();
                std::remove(tampered_bin.c_str());
                std::remove(test_bin.c_str());
                return 1;
            }
            if (t_err.find("stored SHA-256 does not match computed payload digest") == std::string::npos) {
                std::cerr << "FAIL: Tamper test failed for an unexpected reason: " << t_err << "\\n";
                tamp_reader.close();
                std::remove(tampered_bin.c_str());
                std::remove(test_bin.c_str());
                return 1;
            }
            tamp_reader.close();
            std::remove(tampered_bin.c_str());
            std::cout << "  [Adversarial] Tampered payload detection: PASS (fixture opened successfully, digest mismatch detected: " << t_err << ")\n";
        }

        // 4b. Corrupted magic bytes rejected safely
        {
            std::string corrupt_bin = "test_corrupt_magic.bin";
            std::ofstream dst(corrupt_bin, std::ios::binary);
            char bad_hdr[20] = {'N', 'O', 'P', 'E', 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
            dst.write(bad_hdr, 20);
            dst.close();

            bnn::BinarySkillReader bad_reader;
            const bool opened = bad_reader.open(corrupt_bin, true);
            if (opened || bad_reader.is_valid()) {
                std::cerr << "FAIL: Corrupted magic was accepted or left reader in valid state!\n";
                std::remove(corrupt_bin.c_str());
                std::remove(test_bin.c_str());
                return 1;
            }
            std::remove(corrupt_bin.c_str());
            std::cout << "  [Adversarial] Corrupted magic rejected: PASS (open() returned false, reader safe)\n";
        }

        // 4c. Truncated header (< 20 bytes) rejected safely
        {
            std::string trunc_bin = "test_truncated.bin";
            std::ofstream dst(trunc_bin, std::ios::binary);
            dst.write("SHORT", 5);
            dst.close();

            bnn::BinarySkillReader trunc_reader;
            const bool opened = trunc_reader.open(trunc_bin, true);
            if (opened || trunc_reader.is_valid()) {
                std::cerr << "FAIL: Truncated file was accepted or left reader in valid state!\n";
                std::remove(trunc_bin.c_str());
                std::remove(test_bin.c_str());
                return 1;
            }
            std::remove(trunc_bin.c_str());
            std::cout << "  [Adversarial] Truncated file rejected: PASS (open() returned false, reader safe)\n";
        }

        // 4d. Unmapped/out-of-bounds dereference throws std::out_of_range
        {
            bnn::BinarySkillReader empty_reader;
            bool threw = false;
            try {
                (void)empty_reader.record(0U);
            } catch (const std::out_of_range&) {
                threw = true;
            }
            if (!threw) {
                std::cerr << "FAIL: record(0) on unmapped reader did not throw std::out_of_range!\n";
                std::remove(test_bin.c_str());
                return 1;
            }
            std::cout << "  [Adversarial] Unmapped dereference throws out_of_range: PASS\n";
        }

        std::cout << "SUCCESS: compiled and verified " << test_reader.count() << " skills in pure C++!\n";
        std::cout << "Mapping mode: " << (test_reader.is_memory_mapped() ? "Native OS MMAP" : "Buffered") << "\n";
        std::remove(test_bin.c_str());
        return 0;
    }

    bnn::BinarySkillReader reader;
    if (!reader.open(bin_path, true)) {
        bin_path = "binary-skill/" + bin_path;
        if (!reader.open(bin_path, true)) {
            std::cerr << "Error: Could not open valid binary skill file at '" << bin_path << "'\n";
            return 1;
        }
    }

    if (command == "info") {
        const auto* hdr = reader.header();
        std::cout << "========================================================\n"
                  << "  BNN Binary Skill Knowledge Base Engine\n"
                  << "========================================================\n"
                  << "Magic Header       : " << std::string(hdr->magic, 4) << "\n"
                  << "Format Version     : " << hdr->version << "\n"
                  << "Skills Count       : " << hdr->skill_count << " records\n"
                  << "String Table Offset: " << hdr->string_table_offset << " bytes\n"
                  << "String Table Size  : " << hdr->string_table_bytes << " bytes ("
                  << (hdr->string_table_bytes / 1024) << " KB)\n"
                  << "Total File Size    : " << reader.file_size() << " bytes\n"
                  << "Whole-File SHA-256 : " << reader.compute_file_sha256_hex() << "\n"
                  << "Access Mode        : " << (reader.is_memory_mapped() ? "Native OS Memory-Mapped (Zero-Heap)" : "Buffered Memory Load") << "\n"
                  << "Record Seek (Index): O(1) Constant-Time\n"
                  << "Name / Text Search : O(N) Linear Scan\n"
                  << "Status             : VERIFIED & ACTIVE\n"
                  << "========================================================\n";
        return 0;
    }

    if (command == "list") {
        std::cout << "-------------------------------------------------------------------------------------------------\n";
        std::cout << std::left << std::setw(6)  << "ID"
                               << std::setw(34) << "SKILL NAME"
                               << std::setw(24) << "CATEGORY"
                               << std::setw(12) << "FLAGS"
                               << "DESCRIPTION PREVIEW\n";
        std::cout << "-------------------------------------------------------------------------------------------------\n";
        for (size_t i = 0; i < reader.count(); ++i) {
            std::string desc(reader.description(i));
            if (desc.size() > 40) {
                desc.resize(37);
                desc += "...";
            }
            std::cout << std::left << std::setw(6)  << (i + 1)
                                   << std::setw(34) << reader.name(i)
                                   << std::setw(24) << reader.category_name(i)
                                   << "0x" << std::hex << std::setw(10) << reader.flags(i) << std::dec
                                   << desc << "\n";
        }
        std::cout << "-------------------------------------------------------------------------------------------------\n";
        std::cout << "Total: " << reader.count() << " skills distilled.\n";
        return 0;
    }

    if (command == "category") {
        if (arg.empty()) {
            std::cout << "Available categories:\n"
                      << "  1: NVIDIA_STACK\n"
                      << "  2: CORE_ENGINEERING\n"
                      << "  3: ANTI_SLOP\n"
                      << "  4: DESIGN_FRONTEND\n"
                      << "  5: DATABASE_BACKEND\n"
                      << "  6: MULTIMODAL_CREATIVE\n";
            return 0;
        }

        uint16_t cat_id = 0;
        if (arg == "NVIDIA_STACK" || arg == "1") cat_id = bnn::CAT_NVIDIA_STACK;
        else if (arg == "CORE_ENGINEERING" || arg == "2") cat_id = bnn::CAT_CORE_ENGINEERING;
        else if (arg == "ANTI_SLOP" || arg == "3") cat_id = bnn::CAT_ANTI_SLOP;
        else if (arg == "DESIGN_FRONTEND" || arg == "4") cat_id = bnn::CAT_DESIGN_FRONTEND;
        else if (arg == "DATABASE_BACKEND" || arg == "5") cat_id = bnn::CAT_DATABASE_BACKEND;
        else if (arg == "MULTIMODAL_CREATIVE" || arg == "6") cat_id = bnn::CAT_MULTIMODAL_CREATIVE;

        auto indices = reader.find_by_category(cat_id);
        std::cout << "Category " << bnn::category_to_string(cat_id) << " (" << indices.size() << " skills):\n";
        for (size_t idx : indices) {
            std::cout << "  - " << std::left << std::setw(32) << reader.name(idx)
                      << " : " << reader.description(idx) << "\n";
        }
        return 0;
    }

    if (command == "search") {
        if (arg.empty()) {
            std::cerr << "Usage: search <query>\n";
            return 1;
        }

        auto start = std::chrono::high_resolution_clock::now();
        auto matches = reader.search_keyword(arg);
        auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::high_resolution_clock::now() - start
        ).count();

        std::cout << "Search query: '" << arg << "' found " << matches.size() << " match(es) in "
                  << elapsed << " us (O(N) text scan):\n";
        for (size_t idx : matches) {
            std::cout << "  [" << (idx + 1) << "] " << reader.name(idx)
                      << " (" << reader.category_name(idx) << ")\n"
                      << "      " << reader.description(idx) << "\n";
        }
        return 0;
    }

    if (command == "show") {
        if (arg.empty()) {
            std::cerr << "Usage: show <skill_name>\n";
            return 1;
        }

        int idx = reader.find_by_name(arg);
        if (idx < 0) {
            std::cerr << "Skill '" << arg << "' not found.\n";
            return 1;
        }

        std::cout << "Skill Name      : " << reader.name(idx) << "\n"
                  << "Category        : " << reader.category_name(idx) << " (ID: " << reader.category(idx) << ")\n"
                  << "Flags           : 0x" << std::hex << reader.flags(idx) << std::dec << "\n"
                  << "Stored SHA-256  : " << reader.stored_sha256_hex(idx) << "\n"
                  << "Payload SHA-256 : " << reader.compute_payload_sha256_hex(idx) << "\n"
                  << "Description     : " << reader.description(idx) << "\n\n"
                  << "--- Body Extract ---\n"
                  << reader.body(idx).substr(0, 1000) << "\n...\n";
        return 0;
    }

    if (command == "verify") {
        std::cout << "========================================================\n"
                  << "  BNN Cryptographic & Structural Integrity Verification\n"
                  << "========================================================\n";

        // Phase 1: Structural layout & bounds validation
        std::string err;
        if (!reader.verify_all(&err)) {
            std::cerr << "FAIL: Structural verification error: " << err << "\n";
            return 1;
        }
        std::cout << "  [1/4] Header & record table bounds: PASS\n";
        std::cout << "  [2/4] String table checked arithmetic: PASS\n";

        // Phase 2: Per-record cryptographic digest verification (Finding 1)
        if (!reader.verify_record_digests(&err)) {
            std::cerr << "FAIL: Per-record SHA-256 digest mismatch: " << err << "\n";
            return 1;
        }
        std::cout << "  [3/4] Per-record SHA-256 payload digests: PASS ("
                  << reader.count() << "/" << reader.count() << " records verified)\n";

        // Phase 3: Whole-file SHA-256 digest calculation & optional expected comparison
        std::string file_sha = reader.compute_file_sha256_hex();
        std::cout << "  [4/4] Whole-file cryptographic digest: " << file_sha << "\n";

        if (!arg.empty()) {
            if (arg != file_sha) {
                std::cerr << "FAIL: Expected SHA-256 '" << arg << "' does not match computed '" << file_sha << "'\n";
                return 1;
            }
            std::cout << "        Expected hash comparison: MATCH (PASS)\n";
        }

        std::cout << "========================================================\n"
                  << "PASS: All " << reader.count()
                  << " records structurally valid and cryptographically verified.\n"
                  << "========================================================\n";
        return 0;
    }

    std::cerr << "Unknown command: " << command << "\n";
    print_usage(argv[0]);
    return 1;
}
