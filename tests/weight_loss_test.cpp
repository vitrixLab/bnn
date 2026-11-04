#include "../src/weight_loss.hpp"

#include <cassert>
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>

static float fp32_dot(const float* weights, const float* x, int cols) {
    float result = 0.0f;
    for (int c = 0; c < cols; ++c) result += weights[c] * x[c];
    return result;
}

static bool within_quantization_bound(float actual, float reference) {
    return std::fabs(actual - reference) <= 0.25f + 0.1f * std::fabs(reference);
}

// INT4 weight fixture: {2,-1,0,-2,1,0} -> {+7,-4,0,-7,+4,0}
// Tests double-normalized rounding: max_abs=2, scale=2/7
static void test_int4_weight_fixture() {
    // Weights: 2 rows x 3 cols = {2, -1, 0} and {-2, 1, 0}
    const float w[] = {2.0f, -1.0f, 0.0f, -2.0f, 1.0f, 0.0f};
    const auto q4 = weight_loss::quantize_int4(w, 2, 3);
    assert(q4.packed.size() == 3);
    assert(q4.row_scales.size() == 2);
    // Row 0: max_abs=2, scale=2/7 -> {2,-1,0} -> {7,-4,0}
    assert(weight_loss::unpack_int4(q4, 0) == 7);
    assert(weight_loss::unpack_int4(q4, 1) == -4);
    assert(weight_loss::unpack_int4(q4, 2) == 0);
    // Row 1: max_abs=2, scale=2/7 -> {-2,1,0} -> {-7,4,0}
    assert(weight_loss::unpack_int4(q4, 3) == -7);
    assert(weight_loss::unpack_int4(q4, 4) == 4);
    assert(weight_loss::unpack_int4(q4, 5) == 0);
}

// INT8 activation fixture: weights {0,1}, inputs {1,+0.5} and {1,-0.5}
// Requires finite outputs with y>0.5 and y<-0.5 respectively
static void test_int8_activation_fixture() {
    // Test case: 1 row x 2 cols weights {0, 1}
    const float w[] = {0.0f, 1.0f};
    const auto q8 = weight_loss::quantize_int8(w, 1, 2);
    assert(q8.packed.size() == 2);
    assert(q8.row_scales.size() == 1);
    // max_abs=1, scale=1/127, quantized: 0->0, 1->127
    assert(weight_loss::unpack_int8(q8, 0) == 0);
    assert(weight_loss::unpack_int8(q8, 1) == 127);

    // Test input {1, +0.5} -> expect y > 0.5
    float y_pos[1] = {};
    const float x_pos[] = {1.0f, 0.5f};
    weight_loss::matvec_int8(q8, x_pos, y_pos);
    assert(std::isfinite(y_pos[0]));
    assert(y_pos[0] > 0.5f);

    // Test input {1, -0.5} -> expect y < -0.5
    float y_neg[1] = {};
    const float x_neg[] = {1.0f, -0.5f};
    weight_loss::matvec_int8(q8, x_neg, y_neg);
    assert(std::isfinite(y_neg[0]));
    assert(y_neg[0] < -0.5f);
}

// INT4 positive half-step boundary using exactly representable values.
// max_abs=14.0f, scale=14.0f/7.0f=2.0f (exact in float32).
// weight=7.0f → 7.0f/2.0f=3.5f (exact in float32) → std::round(3.5f)=4.
static void test_int4_positive_half_step() {
    const float w[] = {14.0f, 7.0f, 0.0f};
    const auto q4 = weight_loss::quantize_int4(w, 1, 3);
    assert(q4.row_scales.size() == 1);
    assert(q4.packed.size() == 2);
    assert(weight_loss::unpack_int4(q4, 0) == 7);
    assert(weight_loss::unpack_int4(q4, 1) == 4);
    assert(weight_loss::unpack_int4(q4, 2) == 0);
}

// INT4 negative half-step boundary using exactly representable values.
// max_abs=14.0f, scale=14.0f/7.0f=2.0f (exact in float32).
// weight=-7.0f → -7.0f/2.0f=-3.5f (exact in float32) → std::round(-3.5f)=-4.
static void test_int4_negative_half_step() {
    const float w[] = {-14.0f, -7.0f, 0.0f};
    const auto q4 = weight_loss::quantize_int4(w, 1, 3);
    assert(q4.row_scales.size() == 1);
    assert(q4.packed.size() == 2);
    assert(weight_loss::unpack_int4(q4, 0) == -7);
    assert(weight_loss::unpack_int4(q4, 1) == -4);
    assert(weight_loss::unpack_int4(q4, 2) == 0);
}

// INT4 max_abs weight maps to saturation boundary ±7.
// NOTE: Independent clamp testing is unsupported by the current public API:
// max_abs is derived from the same row being quantized, bounding
// |weights[i] / scale| <= 7 + epsilon through normal quantization. The clamp
// in quantize_int4() is defensive only; no public input has been identified
// that triggers it through quantize_int4(). This test verifies the strongest
// behavior the API supports: max_abs weight → ±7.
// max_abs=14.0f, scale=14.0f/7.0f=2.0f (exact) → weight=14.0f → 14.0f/2.0f=7.0f.
static void test_int4_saturation_boundary() {
    const float w[] = {14.0f, -14.0f, 0.0f};
    const auto q4 = weight_loss::quantize_int4(w, 1, 3);
    assert(q4.row_scales.size() == 1);
    assert(q4.packed.size() == 2);
    assert(weight_loss::unpack_int4(q4, 0) == 7);
    assert(weight_loss::unpack_int4(q4, 1) == -7);
    assert(weight_loss::unpack_int4(q4, 2) == 0);
}

// INT8 positive half-step boundary using exactly representable values.
// max_abs=254.0f, scale=254.0f/127.0f=2.0f (exact in float32).
// weight=127.0f → 127.0f/2.0f=63.5f (exact in float32) → std::round(63.5f)=64.
static void test_int8_positive_half_step() {
    const float w[] = {254.0f, 127.0f};
    const auto q8 = weight_loss::quantize_int8(w, 1, 2);
    assert(q8.row_scales.size() == 1);
    assert(q8.packed.size() == 2);
    assert(weight_loss::unpack_int8(q8, 0) == 127);
    assert(weight_loss::unpack_int8(q8, 1) == 64);
}

// INT8 negative half-step boundary using exactly representable values.
// max_abs=254.0f, scale=254.0f/127.0f=2.0f (exact in float32).
// weight=-127.0f → -127.0f/2.0f=-63.5f (exact in float32) → std::round(-63.5f)=-64.
static void test_int8_negative_half_step() {
    const float w[] = {-254.0f, -127.0f};
    const auto q8 = weight_loss::quantize_int8(w, 1, 2);
    assert(q8.row_scales.size() == 1);
    assert(q8.packed.size() == 2);
    assert(weight_loss::unpack_int8(q8, 0) == -127);
    assert(weight_loss::unpack_int8(q8, 1) == -64);
}

// INT8 max_abs weight maps to saturation boundary ±127.
// Independent clamp testing is unsupported by the current public API for the
// same reason: max_abs is derived from the same row being quantized, bounding
// |weights[i] / scale| <= 127 + epsilon through normal quantization. The clamp
// in quantize_int8() is defensive only; no public input has been identified
// that triggers it through quantize_int8(). This test verifies the strongest
// behavior the API supports: max_abs weight → ±127.
// max_abs=254.0f, scale=254.0f/127.0f=2.0f (exact) → weight=254.0f → 254.0f/2.0f=127.0f.
static void test_int8_saturation_boundary() {
    const float w[] = {254.0f, -254.0f, 0.0f};
    const auto q8 = weight_loss::quantize_int8(w, 1, 3);
    assert(q8.row_scales.size() == 1);
    assert(q8.packed.size() == 3);
    assert(weight_loss::unpack_int8(q8, 0) == 127);
    assert(weight_loss::unpack_int8(q8, 1) == -127);
    assert(weight_loss::unpack_int8(q8, 2) == 0);
}

int main() {
    // INT4: odd element count exercises the unused high nibble.
    const float w[] = {1.0f, -2.0f, 0.0f, 0.5f, 3.0f, -1.0f};
    const auto q4 = weight_loss::quantize_int4(w, 2, 3);
    assert(q4.packed.size() == 3);
    assert(q4.row_scales.size() == 2);
    assert(weight_loss::unpack_int4(q4, 0) == 4);
    assert(weight_loss::unpack_int4(q4, 1) == -7);

    // Half-ties are computed from the original values/max_abs, avoiding
    // a scale-division result that can land just below the mathematical tie.
    const float tie_weights[] = {2.0f, -1.0f, 0.0f, -2.0f, 1.0f, 0.0f};
    const auto qtie = weight_loss::quantize_int4(tie_weights, 1, 6);
    const int expected_ties[] = {7, -4, 0, -7, 4, 0};
    for (std::size_t i = 0; i < 6; ++i)
        assert(weight_loss::unpack_int4(qtie, i) == expected_ties[i]);

    // INT8 activation half-steps must round symmetrically away from zero.
    const float activation_weights[] = {0.0f, 1.0f};
    const auto qactivation = weight_loss::quantize_int4(activation_weights, 1, 2);
    const float positive_half_step[] = {1.0f, 0.5f};
    const float negative_half_step[] = {1.0f, -0.5f};
    float positive_output = 0.0f;
    float negative_output = 0.0f;
    weight_loss::matvec_int4(qactivation, positive_half_step, &positive_output);
    weight_loss::matvec_int4(qactivation, negative_half_step, &negative_output);
    assert(std::isfinite(positive_output) && positive_output > 0.5f);
    assert(std::isfinite(negative_output) && negative_output < -0.5f);
    bool out_of_range_rejected = false;
    try { (void)weight_loss::unpack_int4(q4, 6); }
    catch (const std::out_of_range&) { out_of_range_rejected = true; }
    assert(out_of_range_rejected);

    // Run INT4 weight fixture test
    test_int4_weight_fixture();

    const float x[] = {1.0f, -1.0f, 0.25f};
    float y4[2] = {};
    weight_loss::matvec_int4(q4, x, y4);
    for (int r = 0; r < 2; ++r) {
        const float reference = fp32_dot(w + r * 3, x, 3);
        assert(std::isfinite(y4[r]));
        assert(within_quantization_bound(y4[r], reference));
    }

    // INT8 activation fixture tests
    test_int8_activation_fixture();

    // New boundary-case tests: INT4/INT8 half-step rounding and saturation
    test_int4_positive_half_step();
    test_int4_negative_half_step();
    test_int4_saturation_boundary();
    test_int8_positive_half_step();
    test_int8_negative_half_step();
    test_int8_saturation_boundary();

    const float zeros[] = {0.0f, 0.0f, 0.0f};
    const auto qzero = weight_loss::quantize_int4(zeros, 1, 3);
    float yz = 1.0f;
    weight_loss::matvec_int4(qzero, x, &yz);
    assert(yz == 0.0f);

    // BIN1: 35 columns exercises a partial final word and compares with FP32.
    // Unit-magnitude weights and constant-magnitude activations make the sign
    // approximation exact, including the final three-bit tail.
    constexpr int binary_cols = 35;
    std::vector<float> wb(2 * binary_cols);
    std::vector<float> xb(binary_cols);
    for (int i = 0; i < 2 * binary_cols; ++i)
        wb[static_cast<std::size_t>(i)] = (i % 3 == 0) ? -1.0f : 1.0f;
    for (int i = 0; i < binary_cols; ++i)
        xb[static_cast<std::size_t>(i)] = (i % 2 == 0) ? 2.0f : -2.0f;
    const auto qb = weight_loss::quantize_binary(wb.data(), 2, binary_cols);
    float yb[2] = {};
    weight_loss::matvec_binary(qb, xb.data(), yb);
    for (int r = 0; r < 2; ++r) {
        const float reference = fp32_dot(wb.data() + r * binary_cols, xb.data(), binary_cols);
        assert(std::isfinite(yb[r]));
        assert(std::fabs(yb[r] - reference) <= 1e-5f);
    }

    const float bad[] = {1.0f, std::numeric_limits<float>::infinity()};
    bool rejected = false;
    try { (void)weight_loss::quantize_int4(bad, 1, 2); }
    catch (const std::invalid_argument&) { rejected = true; }
    assert(rejected);
    rejected = false;
    try { (void)weight_loss::quantize_binary(bad, 1, 2); }
    catch (const std::invalid_argument&) { rejected = true; }
    assert(rejected);

    float bad_activation[] = {1.0f, std::numeric_limits<float>::quiet_NaN(), 0.0f};
    rejected = false;
    try { weight_loss::matvec_int4(q4, bad_activation, y4); }
    catch (const std::invalid_argument&) { rejected = true; }
    assert(rejected);

    std::cout << "weight_loss tests passed\n";
    return 0;
}
