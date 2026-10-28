#include "binary_skill.hpp"

#include <cstdint>
#include <fstream>
#include <iostream>
#include <iterator>
#include <limits>
#include <map>
#include <regex>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

struct JsonValue {
    enum class Type { Null, Boolean, Number, String, Array, Object };
    Type type = Type::Null;
    bool boolean_value = false;
    std::string text;
    std::vector<JsonValue> array_value;
    std::map<std::string, JsonValue> object_value;
};

class JsonParser {
public:
    explicit JsonParser(const std::string& input) : input_(input) {}

    JsonValue parse() {
        skip_whitespace();
        JsonValue root = parse_value(0);
        skip_whitespace();
        if (position_ != input_.size()) {
            fail("trailing characters after the JSON value");
        }
        return root;
    }

private:
    static constexpr std::size_t kMaxDepth = 128;
    const std::string& input_;
    std::size_t position_ = 0;

    [[noreturn]] void fail(const std::string& message) const {
        throw std::runtime_error("invalid JSON at byte " +
                                 std::to_string(position_) + ": " + message);
    }

    void skip_whitespace() {
        while (position_ < input_.size()) {
            const char ch = input_[position_];
            if (ch != ' ' && ch != '\t' && ch != '\n' && ch != '\r') break;
            ++position_;
        }
    }

    bool consume(char expected) {
        if (position_ < input_.size() && input_[position_] == expected) {
            ++position_;
            return true;
        }
        return false;
    }

    void expect(char expected) {
        if (!consume(expected)) {
            fail(std::string("expected '") + expected + "'");
        }
    }

    static void append_utf8(std::string& output, unsigned int cp) {
        if (cp <= 0x7FU) {
            output.push_back(static_cast<char>(cp));
        } else if (cp <= 0x7FFU) {
            output.push_back(static_cast<char>(0xC0U | (cp >> 6U)));
            output.push_back(static_cast<char>(0x80U | (cp & 0x3FU)));
        } else if (cp <= 0xFFFFU) {
            output.push_back(static_cast<char>(0xE0U | (cp >> 12U)));
            output.push_back(static_cast<char>(0x80U | ((cp >> 6U) & 0x3FU)));
            output.push_back(static_cast<char>(0x80U | (cp & 0x3FU)));
        } else {
            output.push_back(static_cast<char>(0xF0U | (cp >> 18U)));
            output.push_back(static_cast<char>(0x80U | ((cp >> 12U) & 0x3FU)));
            output.push_back(static_cast<char>(0x80U | ((cp >> 6U) & 0x3FU)));
            output.push_back(static_cast<char>(0x80U | (cp & 0x3FU)));
        }
    }

    unsigned int parse_hex_quad() {
        if (input_.size() - position_ < 4U) fail("incomplete unicode escape");
        unsigned int value = 0;
        for (unsigned int i = 0; i < 4U; ++i) {
            const char ch = input_[position_++];
            value <<= 4U;
            if (ch >= '0' && ch <= '9') value |= static_cast<unsigned int>(ch - '0');
            else if (ch >= 'a' && ch <= 'f') value |= static_cast<unsigned int>(ch - 'a' + 10);
            else if (ch >= 'A' && ch <= 'F') value |= static_cast<unsigned int>(ch - 'A' + 10);
            else fail("non-hexadecimal digit in unicode escape");
        }
        return value;
    }

    std::string parse_string() {
        expect('"');
        std::string output;
        while (position_ < input_.size()) {
            const unsigned char ch = static_cast<unsigned char>(input_[position_++]);
            if (ch == '"') return output;
            if (ch < 0x20U) fail("unescaped control character in string");
            if (ch != '\\') {
                output.push_back(static_cast<char>(ch));
                continue;
            }
            if (position_ >= input_.size()) fail("incomplete escape sequence");
            const char escape = input_[position_++];
            switch (escape) {
                case '"': output.push_back('"'); break;
                case '\\': output.push_back('\\'); break;
                case '/': output.push_back('/'); break;
                case 'b': output.push_back('\b'); break;
                case 'f': output.push_back('\f'); break;
                case 'n': output.push_back('\n'); break;
                case 'r': output.push_back('\r'); break;
                case 't': output.push_back('\t'); break;
                case 'u': {
                    unsigned int cp = parse_hex_quad();
                    if (cp >= 0xD800U && cp <= 0xDBFFU) {
                        if (input_.size() - position_ < 6U ||
                            input_[position_] != '\\' || input_[position_ + 1U] != 'u') {
                            fail("high surrogate without low surrogate");
                        }
                        position_ += 2U;
                        const unsigned int low = parse_hex_quad();
                        if (low < 0xDC00U || low > 0xDFFFU) {
                            fail("invalid low surrogate");
                        }
                        cp = 0x10000U + ((cp - 0xD800U) << 10U) + (low - 0xDC00U);
                    } else if (cp >= 0xDC00U && cp <= 0xDFFFU) {
                        fail("low surrogate without high surrogate");
                    }
                    append_utf8(output, cp);
                    break;
                }
                default: fail("unknown string escape");
            }
        }
        fail("unterminated string");
    }

    JsonValue parse_number() {
        const std::size_t start = position_;
        consume('-');
        if (position_ >= input_.size()) fail("incomplete number");
        if (consume('0')) {
            if (position_ < input_.size() && input_[position_] >= '0' &&
                input_[position_] <= '9') {
                fail("leading zero in number");
            }
        } else {
            if (input_[position_] < '1' || input_[position_] > '9') {
                fail("invalid number");
            }
            while (position_ < input_.size() &&
                   input_[position_] >= '0' && input_[position_] <= '9') ++position_;
        }
        if (consume('.')) {
            const std::size_t fraction_start = position_;
            while (position_ < input_.size() &&
                   input_[position_] >= '0' && input_[position_] <= '9') ++position_;
            if (position_ == fraction_start) fail("fraction requires a digit");
        }
        if (position_ < input_.size() &&
            (input_[position_] == 'e' || input_[position_] == 'E')) {
            ++position_;
            if (position_ < input_.size() &&
                (input_[position_] == '+' || input_[position_] == '-')) ++position_;
            const std::size_t exponent_start = position_;
            while (position_ < input_.size() &&
                   input_[position_] >= '0' && input_[position_] <= '9') ++position_;
            if (position_ == exponent_start) fail("exponent requires a digit");
        }
        JsonValue value;
        value.type = JsonValue::Type::Number;
        value.text = input_.substr(start, position_ - start);
        return value;
    }

    bool consume_literal(const char* literal) {
        std::size_t i = 0;
        while (literal[i] != '\0') {
            if (position_ + i >= input_.size() || input_[position_ + i] != literal[i]) {
                return false;
            }
            ++i;
        }
        position_ += i;
        return true;
    }

    JsonValue parse_value(std::size_t depth) {
        if (depth > kMaxDepth) fail("maximum nesting depth exceeded");
        skip_whitespace();
        if (position_ >= input_.size()) fail("expected a JSON value");
        const char ch = input_[position_];
        if (ch == '"') {
            JsonValue value;
            value.type = JsonValue::Type::String;
            value.text = parse_string();
            return value;
        }
        if (ch == '{') return parse_object(depth);
        if (ch == '[') return parse_array(depth);
        if (ch == 't') {
            if (!consume_literal("true")) fail("invalid literal");
            JsonValue value;
            value.type = JsonValue::Type::Boolean;
            value.boolean_value = true;
            return value;
        }
        if (ch == 'f') {
            if (!consume_literal("false")) fail("invalid literal");
            JsonValue value;
            value.type = JsonValue::Type::Boolean;
            value.boolean_value = false;
            return value;
        }
        if (ch == 'n') {
            if (!consume_literal("null")) fail("invalid literal");
            return JsonValue{};
        }
        if (ch == '-' || (ch >= '0' && ch <= '9')) return parse_number();
        fail("unexpected token");
    }

    JsonValue parse_object(std::size_t depth) {
        expect('{');
        JsonValue object;
        object.type = JsonValue::Type::Object;
        skip_whitespace();
        if (consume('}')) return object;
        while (true) {
            skip_whitespace();
            if (position_ >= input_.size() || input_[position_] != '"') {
                fail("object member name must be a string");
            }
            std::string key = parse_string();
            skip_whitespace();
            expect(':');
            JsonValue value = parse_value(depth + 1U);
            if (!object.object_value.emplace(key, std::move(value)).second) {
                fail("duplicate object key: " + key);
            }
            skip_whitespace();
            if (consume('}')) return object;
            expect(',');
            skip_whitespace();
            if (position_ < input_.size() && input_[position_] == '}') {
                fail("trailing comma in object");
            }
        }
    }

    JsonValue parse_array(std::size_t depth) {
        expect('[');
        JsonValue array;
        array.type = JsonValue::Type::Array;
        skip_whitespace();
        if (consume(']')) return array;
        while (true) {
            array.array_value.push_back(parse_value(depth + 1U));
            skip_whitespace();
            if (consume(']')) return array;
            expect(',');
            skip_whitespace();
            if (position_ < input_.size() && input_[position_] == ']') {
                fail("trailing comma in array");
            }
        }
    }
};

const JsonValue& required_field(const JsonValue& object, const std::string& key) {
    if (object.type != JsonValue::Type::Object) {
        throw std::runtime_error("expected JSON object while looking for field: " + key);
    }
    const auto found = object.object_value.find(key);
    if (found == object.object_value.end()) {
        throw std::runtime_error("missing required field: " + key);
    }
    return found->second;
}

const JsonValue* optional_field(const JsonValue& object, const std::string& key) {
    if (object.type != JsonValue::Type::Object) {
        throw std::runtime_error("expected JSON object while looking for field: " + key);
    }
    const auto found = object.object_value.find(key);
    return found == object.object_value.end() ? nullptr : &found->second;
}

std::string required_string(const JsonValue& object, const std::string& key) {
    const JsonValue& value = required_field(object, key);
    if (value.type != JsonValue::Type::String) {
        throw std::runtime_error("field must be a string: " + key);
    }
    return value.text;
}

uint64_t required_uint(const JsonValue& object, const std::string& key,
                       bool allow_zero = false) {
    const JsonValue& value = required_field(object, key);
    if (value.type != JsonValue::Type::Number || value.text.empty() ||
        value.text[0] == '-' ||
        value.text.find_first_of(".eE") != std::string::npos) {
        throw std::runtime_error("field must be an unsigned integer: " + key);
    }
    try {
        std::size_t consumed = 0;
        const unsigned long long parsed = std::stoull(value.text, &consumed, 10);
        if (consumed != value.text.size() ||
            parsed > static_cast<unsigned long long>(std::numeric_limits<uint64_t>::max()) ||
            (!allow_zero && parsed == 0ULL)) {
            throw std::runtime_error("integer field outside allowed range: " + key);
        }
        return static_cast<uint64_t>(parsed);
    } catch (const std::invalid_argument&) {
        throw std::runtime_error("invalid unsigned integer field: " + key);
    } catch (const std::out_of_range&) {
        throw std::runtime_error("integer field outside allowed range: " + key);
    }
}

std::string optional_string(const JsonValue& object, const std::string& key,
                            const std::string& fallback) {
    const JsonValue* value = optional_field(object, key);
    if (!value) return fallback;
    if (value->type != JsonValue::Type::String) {
        throw std::runtime_error("field must be a string: " + key);
    }
    return value->text;
}

std::string read_file(const std::string& path) {
    std::ifstream input(path, std::ios::binary);
    if (!input) {
        throw std::runtime_error("cannot open file: " + path);
    }
    return std::string(std::istreambuf_iterator<char>(input),
                       std::istreambuf_iterator<char>());
}

void validate_results(const JsonValue& report) {
    const JsonValue& results = required_field(report, "results");
    if (results.type != JsonValue::Type::Array || results.array_value.empty()) {
        throw std::runtime_error("results must be a non-empty array");
    }
    for (const JsonValue& item : results.array_value) {
        if (item.type != JsonValue::Type::Object) {
            throw std::runtime_error("each results entry must be an object");
        }
        (void)required_string(item, "operation");
        (void)required_uint(item, "samples");
        (void)required_uint(item, "matches", true);
        const uint64_t min_ns = required_uint(item, "min_ns", true);
        const uint64_t median_ns = required_uint(item, "median_ns", true);
        const uint64_t p95_ns = required_uint(item, "p95_ns", true);
        const uint64_t max_ns = required_uint(item, "max_ns", true);
        if (!(min_ns <= median_ns && median_ns <= p95_ns && p95_ns <= max_ns)) {
            throw std::runtime_error("timing quantiles are out of order");
        }
    }
}

void validate_report(const JsonValue& report, const std::string& binary_path) {
    if (report.type != JsonValue::Type::Object) {
        throw std::runtime_error("benchmark report root must be an object");
    }
    if (required_uint(report, "schema_version") != 1U) {
        throw std::runtime_error("unsupported schema_version");
    }

    const JsonValue& provenance = required_field(report, "provenance");
    if (provenance.type != JsonValue::Type::Object) {
        throw std::runtime_error("provenance must be an object");
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

    const std::string commit = optional_string(provenance, "commit", "");
    const std::string dirty = optional_string(provenance, "source_dirty", "unknown");
    const std::string context = optional_string(provenance, "bench_context", "unknown");
    if (build_type == "Release" &&
        std::regex_match(commit, std::regex("[0-9a-f]{40}")) &&
        dirty == "false" && context == "controlled-release") {
        std::cout << "Controlled-release provenance contract PASS\n";
    } else {
        std::cout << "NOTICE: validated as historical evidence; not certified as controlled-release provenance.\n";
    }
}

bool parser_rejects(const std::string& input) {
    try {
        (void)JsonParser(input).parse();
        return false;
    } catch (const std::exception&) {
        return true;
    }
}

int run_json_parser_self_tests() {
    try {
        const JsonValue valid = JsonParser(
            R"({"message":"string with \"provenance\": { and braces }","provenance":{"build_type":"Release","note":"close } and \"quote\""}})"
        ).parse();
        const auto provenance = valid.object_value.find("provenance");
        if (valid.type != JsonValue::Type::Object ||
            provenance == valid.object_value.end() ||
            provenance->second.type != JsonValue::Type::Object ||
            required_string(provenance->second, "build_type") != "Release" ||
            required_string(provenance->second, "note") != "close } and \"quote\"") {
            throw std::runtime_error("valid escaped strings or nested object were parsed incorrectly");
        }

        const JsonValue string_only = JsonParser(
            R"({"message":"\"provenance\": {\"build_type\":\"Release\"}"})"
        ).parse();
        if (string_only.object_value.find("provenance") != string_only.object_value.end()) {
            throw std::runtime_error("key-like text inside a string became an object property");
        }

        if (!parser_rejects(R"({"provenance":{},"provenance":{}})")) {
            throw std::runtime_error("duplicate keys after unicode decoding were accepted");
        }
        if (!parser_rejects(R"({"provenance": })")) {
            throw std::runtime_error("malformed object value was accepted");
        }
        if (!parser_rejects(R"({"provenance":{},})")) {
            throw std::runtime_error("trailing object comma was accepted");
        }
        if (!parser_rejects(R"({"value":"bad\q escape"})")) {
            throw std::runtime_error("invalid string escape was accepted");
        }
        if (!parser_rejects("{\"value\":\"line one\nline two\"}")) {
            throw std::runtime_error("unescaped control character was accepted");
        }
        if (!parser_rejects(R"({"value":"ok"} trailing)")) {
            throw std::runtime_error("trailing non-JSON text was accepted");
        }

        std::cout << "PASS: strict JSON parser self-tests (escaping, property scope, unicode keys, duplicates, malformed input)\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "FAIL: JSON parser self-test: " << error.what() << '\n';
        return 1;
    }
}

}  // namespace

int main(int argc, char* argv[]) {
    if (argc == 2 && std::string(argv[1]) == "--self-test-json-parser") {
        return run_json_parser_self_tests();
    }
    if (argc != 3) {
        std::cerr << "Usage: benchmark_evidence_validator <report.json> <binary-file>\n"
                  << "       benchmark_evidence_validator --self-test-json-parser\n";
        return 2;
    }

    try {
        const JsonValue report = JsonParser(read_file(argv[1])).parse();
        validate_report(report, argv[2]);
        std::cout << "PASS: benchmark report contract and binary integrity\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
}
