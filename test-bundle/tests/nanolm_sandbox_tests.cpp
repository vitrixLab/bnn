#define main nanolm_application_main
#include "nanolm.cpp"
#undef main
#include <filesystem>
#include <functional>
#include <iostream>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

static void require(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}
static std::filesystem::path test_dir(const std::string& name) {
    const auto path = std::filesystem::temp_directory_path() / ("bnn-nanolm-" + name);
    std::filesystem::remove_all(path);
    std::filesystem::create_directories(path);
    return path;
}
static float attention_objective(const Config& cfg, const AttentionWeights& weights,
                                 const std::vector<float>& input, const std::vector<float>& dy) {
    AttentionCache cache;
    attention_fwd(cfg, weights, input, cache);
    float value = 0.0f;
    for (std::size_t i = 0; i < cache.y.size(); ++i) value += cache.y[i] * dy[i];
    return value;
}
static void test_attention_head_cache_and_gradient() {
    Config cfg; cfg.block_size = 3; cfg.n_embd = 4; cfg.n_head = 2; cfg.n_layer = 1;
    AttentionWeights weights; weights.init(cfg.n_embd);
    for (std::size_t i = 0; i < weights.wq.size(); ++i) {
        weights.wq[i] = 0.013f * static_cast<float>(i + 1U) - 0.11f;
        weights.wk[i] = -0.009f * static_cast<float>(i + 2U) + 0.07f;
        weights.wv[i] = 0.017f * static_cast<float>(i + 1U) - 0.08f;
        weights.wo[i] = (i % 5U == 0U) ? 0.8f : 0.03f * static_cast<float>(i % 7U);
    }
    const std::vector<float> input{0.2f,-0.3f,0.5f,0.1f,-0.4f,0.7f,0.2f,-0.1f,0.6f,0.1f,-0.5f,0.3f};
    const std::vector<float> dy{0.4f,-0.2f,0.1f,0.3f,-0.1f,0.5f,-0.3f,0.2f,0.2f,0.1f,0.6f,-0.4f};
    AttentionCache cache; attention_fwd(cfg, weights, input, cache);
    require(cache.att.size() == static_cast<std::size_t>(cfg.n_head*cfg.block_size*cfg.block_size),
            "attention cache must retain one T x T matrix per head");
    AttentionWeights grads; grads.init(cfg.n_embd);
    zero(grads.dwq); zero(grads.dwk); zero(grads.dwv); zero(grads.dwo);
    std::vector<float> dx(input.size(),0.0f);
    attention_bwd(cfg,weights,grads,cache,dy,dx);
    const float epsilon=1.0e-3f, atol=4.0e-3f, rtol=3.0e-2f;
    const auto check=[&](std::vector<float>& p,const std::vector<float>& g,const std::string& name){
        require(p.size()==g.size(),name+" gradient size mismatch");
        for(std::size_t i=0;i<p.size();++i){
            const float original=p[i]; p[i]=original+epsilon;
            const float plus=attention_objective(cfg,weights,input,dy);
            p[i]=original-epsilon;
            const float minus=attention_objective(cfg,weights,input,dy); p[i]=original;
            const float numeric=(plus-minus)/(2.0f*epsilon);
            require(std::fabs(numeric-g[i])<=atol+rtol*std::fabs(numeric),
                    name+" finite-difference mismatch at element "+std::to_string(i));
        }
    };
    check(weights.wq,grads.dwq,"wq"); check(weights.wk,grads.dwk,"wk"); check(weights.wv,grads.dwv,"wv");
    std::cout<<"per-head cache and wq/wk/wv finite differences OK\n";
}
static void test_next_token_full_window_reference() {
    Config cfg; cfg.vocab_size=7; cfg.block_size=4; cfg.n_embd=4; cfg.n_head=2; cfg.n_layer=1;
    ModelWeights weights; weights.init(cfg);
    const std::vector<int> tokens{0,1,2,3,4};
    ForwardCache cache; const float loss=forward(cfg,weights,tokens,cache,true);
    float reference=0.0f;
    for(int t=0;t<cfg.block_size;++t)
        reference-=std::log(cache.probs[static_cast<std::size_t>(t*cfg.vocab_size+tokens[static_cast<std::size_t>(t+1)])]+1.0e-9f);
    reference/=static_cast<float>(cfg.block_size);
    require(std::isfinite(loss)&&std::fabs(loss-reference)<1.0e-6f,"loss must include all T targets");
    backward(cfg,weights,tokens,cache);
    float norm2=0.0f; const std::size_t offset=static_cast<std::size_t>((cfg.block_size-1)*cfg.n_embd);
    for(int e=0;e<cfg.n_embd;++e){const float g=weights.dwpe[offset+static_cast<std::size_t>(e)];norm2+=g*g;}
    require(norm2>1.0e-14f,"final positional embedding row must receive training gradient");
    std::cout<<"T loss terms and final-position gradient OK\n";
}
static void test_parameter_count_matches_allocations() {
    Config cfg; cfg.vocab_size=7; cfg.block_size=3; cfg.n_embd=4; cfg.n_head=2; cfg.n_layer=2;
    ModelWeights weights; weights.init(cfg);
    std::size_t actual=weights.wte.size()+weights.wpe.size();
    for(const auto& b:weights.blocks)
        actual+=b.attn.wq.size()+b.attn.wk.size()+b.attn.wv.size()+b.attn.wo.size()+b.mlp.fc1.size()+b.mlp.fc2.size();
    require(static_cast<std::size_t>(weights.count_params())==actual,"count_params must follow cfg.n_layer");
    std::cout<<"dynamic layer count matches allocated tensors ("<<actual<<")\n";
}
static void test_checkpoint_rejects_before_allocation() {
    const auto dir=test_dir("truncated-checkpoint"), path=dir/"truncated.bin";
    Config cfg;
    {std::ofstream out(path,std::ios::binary);out.write(reinterpret_cast<const char*>(&cfg),static_cast<std::streamsize>(sizeof(cfg)));
     const float partial=0.25f;out.write(reinterpret_cast<const char*>(&partial),static_cast<std::streamsize>(sizeof(partial)));}
    ModelWeights target;target.cfg.vocab_size=123;target.wte={9.0f,8.0f,7.0f};
    require(!load_weights(target,path.string()),"truncated checkpoint must be rejected");
    require(target.cfg.vocab_size==123&&target.wte==std::vector<float>({9.0f,8.0f,7.0f}),
            "rejected checkpoint must not mutate destination model");
    std::filesystem::remove_all(dir);
    std::cout<<"truncated checkpoint rejected before destination allocation/mutation\n";
}
static void test_cli_training_and_save_failures() {
    const auto dir=test_dir("save-failure");const auto old_cwd=std::filesystem::current_path();
    std::filesystem::current_path(dir);const auto data_path=dir/"corpus.txt";
    {std::ofstream out(data_path);for(int i=0;i<80;++i)out<<((i%2==0)?'a':'b');}
    char a0[]="nanolm",a1[]="train",a4[]="0",a5[]="0.003";
    std::string data=data_path.string(),bad=(dir/"missing-parent"/"model.bin").string();
    char* argv[]={a0,a1,data.data(),bad.data(),a4,a5};
    const int result=nanolm_application_main(6,argv);std::filesystem::current_path(old_cwd);
    require(result!=0,"CLI must return non-zero when checkpoint save fails");
    require(!std::filesystem::exists(dir/"missing-parent"/"model.bin"),"failed save must not create checkpoint");
    std::filesystem::remove_all(dir);
    std::cout<<"CLI propagates checkpoint save failure with non-zero exit\n";
}
static void test_isolated_deterministic_replay() {
    const auto dir=test_dir("deterministic-replay");
    Config cfg;cfg.vocab_size=5;cfg.block_size=4;cfg.n_embd=4;cfg.n_head=2;cfg.n_layer=1;
    ModelWeights original;original.init(cfg);const auto model=dir/"model.bin";
    require(save_weights(original,model.string()),"checkpoint save should succeed");
    ModelWeights loaded;require(load_weights(loaded,model.string()),"checkpoint load should succeed");
    Tokenizer tok;tok.build("abcde");
    const std::string first=generate(cfg,loaded,tok,"abc",8,0.0f),second=generate(cfg,loaded,tok,"abc",8,0.0f);
    require(first==second,"greedy generation must replay deterministically");
    require(first.size()==11U,"generation output length mismatch");
    std::filesystem::remove_all(dir);
    std::cout<<"save/load greedy replay is deterministic\n";
}
static void seed_deterministic_gradients(ModelWeights& W) {
    for (std::size_t i = 0; i < W.dwte.size(); ++i) W.dwte[i] = 0.01f * static_cast<float>((i * 7U) % 13U) - 0.06f;
    for (std::size_t i = 0; i < W.dwpe.size(); ++i) W.dwpe[i] = 0.02f * static_cast<float>((i * 5U) % 11U) - 0.09f;
    for (auto& b : W.blocks) {
        auto& a = b.attn;
        auto& m = b.mlp;
        for (std::size_t i = 0; i < a.dwq.size(); ++i) a.dwq[i] = 0.03f * static_cast<float>((i * 3U) % 17U) - 0.21f;
        for (std::size_t i = 0; i < a.dwk.size(); ++i) a.dwk[i] = 0.04f * static_cast<float>((i * 5U) % 19U) - 0.33f;
        for (std::size_t i = 0; i < a.dwv.size(); ++i) a.dwv[i] = 0.05f * static_cast<float>((i * 7U) % 23U) - 0.44f;
        for (std::size_t i = 0; i < a.dwo.size(); ++i) a.dwo[i] = 0.06f * static_cast<float>((i * 11U) % 29U) - 0.57f;
        for (std::size_t i = 0; i < m.dfc1.size(); ++i) m.dfc1[i] = 0.07f * static_cast<float>((i * 13U) % 31U) - 0.66f;
        for (std::size_t i = 0; i < m.dfc2.size(); ++i) m.dfc2[i] = 0.08f * static_cast<float>((i * 17U) % 37U) - 0.77f;
    }
}
static void require_moments_equal(const ModelWeights& a, const ModelWeights& b, const std::string& why) {
    require(a.mwte == b.mwte && a.vwte == b.vwte && a.mwpe == b.mwpe && a.vwpe == b.vwpe,
            why + " (embedding moments)");
    require(a.blocks.size() == b.blocks.size(), why + " (block count)");
    for (std::size_t bi = 0; bi < a.blocks.size(); ++bi) {
        const auto& x = a.blocks[bi].attn; const auto& y = b.blocks[bi].attn;
        const auto& p = a.blocks[bi].mlp;  const auto& q = b.blocks[bi].mlp;
        require(x.mwq == y.mwq && x.vwq == y.vwq && x.mwk == y.mwk && x.vwk == y.vwk &&
                x.mwv == y.mwv && x.vwv == y.vwv && x.mwo == y.mwo && x.vwo == y.vwo &&
                p.mfc1 == q.mfc1 && p.vfc1 == q.vfc1 && p.mfc2 == q.mfc2 && p.vfc2 == q.vfc2,
                why + " (block " + std::to_string(bi) + " moments)");
    }
}
static std::vector<char> read_file_bytes(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    return std::vector<char>((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
}
static void write_file_bytes(const std::string& path, const std::vector<char>& bytes) {
    std::ofstream f(path, std::ios::binary | std::ios::trunc);
    f.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
}
static void require_weights_equal(const ModelWeights& a, const ModelWeights& b, const char* why) {
    require(a.wte == b.wte && a.wpe == b.wpe, why);
    require(a.blocks.size() == b.blocks.size(), why);
    for (std::size_t bi = 0; bi < a.blocks.size(); ++bi) {
        const auto& x = a.blocks[bi];
        const auto& y = b.blocks[bi];
        require(x.attn.wq == y.attn.wq && x.attn.wk == y.attn.wk &&
                x.attn.wv == y.attn.wv && x.attn.wo == y.attn.wo &&
                x.mlp.fc1 == y.mlp.fc1 && x.mlp.fc2 == y.mlp.fc2, why);
    }
}
static void test_training_state_round_trip_exact() {
    const auto dir = test_dir("training-state-roundtrip");
    Config cfg; cfg.vocab_size = 5; cfg.block_size = 4; cfg.n_embd = 4; cfg.n_head = 2; cfg.n_layer = 1;
    ModelWeights W; W.init(cfg);
    Adam opt; opt.lr = 0.01f; opt.lr_final = 1.0e-3f; opt.lr_warmup_iters = 2.0f;
    opt.lr_total_iters = 10; opt.step = 1;
    seed_deterministic_gradients(W);
    opt.update_all(W); opt.update_all(W);
    std::mt19937 data_rng(7);
    for (int i = 0; i < 9; ++i) data_rng();
    std::mt19937 rng_copy = data_rng;
    const Adam opt_saved = opt;
    const ModelWeights moments_snapshot = W;
    const auto state_path = (dir / "state.bin").string();
    require(save_training_state(W, opt, data_rng, 1234U, state_path), "training-state save should succeed");

    for (float& x : W.mwte) x += 1.0f;
    for (float& x : W.vwte) x += 1.0f;
    for (float& x : W.mwpe) x += 1.0f;
    for (float& x : W.vwpe) x += 1.0f;
    for (auto& b : W.blocks) {
        b.attn.mwq[0] += 1.0f; b.attn.vwq[0] += 1.0f; b.attn.mwk[0] += 1.0f; b.attn.vwk[0] += 1.0f;
        b.attn.mwv[0] += 1.0f; b.attn.vwv[0] += 1.0f; b.attn.mwo[0] += 1.0f; b.attn.vwo[0] += 1.0f;
        b.mlp.mfc1[0] += 1.0f; b.mlp.vfc1[0] += 1.0f; b.mlp.mfc2[0] += 1.0f; b.mlp.vfc2[0] += 1.0f;
    }
    opt.step = 99; opt.lr = 5.0f; opt.lr_total_iters = 12345;
    opt.b1 = 0.5f; opt.b2 = 0.75f; opt.eps = 1.0e-3f;
    data_rng.discard(77);

    ModelWeights W2; W2.init(cfg);
    Adam opt2; std::mt19937 rng2(0); std::uint64_t tokens2 = 0;
    require(load_training_state(W2, opt2, rng2, tokens2, state_path), "training-state load should succeed");
    require(opt2.step == opt_saved.step, "Adam step must round-trip exactly");
    require(opt2.lr == opt_saved.lr && opt2.lr_final == opt_saved.lr_final &&
            opt2.lr_warmup_iters == opt_saved.lr_warmup_iters &&
            opt2.lr_total_iters == opt_saved.lr_total_iters &&
            opt2.b1 == opt_saved.b1 && opt2.b2 == opt_saved.b2 && opt2.eps == opt_saved.eps,
            "Adam hyperparameters must round-trip exactly");
    require(tokens2 == 1234U, "corpus token count must round-trip exactly");
    require_moments_equal(W2, moments_snapshot, "restored moments must match the saved model bitwise");
    for (int i = 0; i < 100; ++i)
        require(rng2() == rng_copy(), "restored sampler RNG must reproduce the saved stream exactly");
    std::filesystem::remove_all(dir);
    std::cout << "optimizer/sampler state round-trip is exact\n";
}
static void test_training_state_rejects_invalid_and_preserves_live_state() {
    const auto dir = test_dir("training-state-rejects");
    Config cfg; cfg.vocab_size = 5; cfg.block_size = 4; cfg.n_embd = 4; cfg.n_head = 2; cfg.n_layer = 1;
    ModelWeights W; W.init(cfg);
    Adam opt; opt.lr = 0.01f; opt.lr_total_iters = 10; opt.step = 1;
    seed_deterministic_gradients(W);
    opt.update_all(W);
    std::mt19937 data_rng(7);
    const auto state_path = (dir / "state.bin").string();
    require(save_training_state(W, opt, data_rng, 500U, state_path), "valid training-state save should succeed");
    const std::vector<char> valid = read_file_bytes(state_path);
    require(valid.size() > 64U, "state file should contain header and payload");

    ModelWeights target; target.init(cfg);
    Adam target_opt; target_opt.step = 42; target_opt.lr = 0.5f; target_opt.lr_total_iters = 77;
    std::mt19937 target_rng(3); std::uint64_t target_tokens = 999;
    const ModelWeights target_snapshot = target;
    const Adam opt_snapshot = target_opt;

    auto expect_reject = [&](const std::string& bad_path, const char* why) {
        std::mt19937 rng_canon = target_rng;
        require(!load_training_state(target, target_opt, target_rng, target_tokens, bad_path), why);
        require_moments_equal(target, target_snapshot, std::string(why) + ": live moments must survive rejection");
        require(target_opt.step == opt_snapshot.step && target_opt.lr == opt_snapshot.lr &&
                target_opt.lr_total_iters == opt_snapshot.lr_total_iters,
                std::string(why) + ": live optimizer must survive rejection");
        require(target_tokens == 999U, std::string(why) + ": live token count must survive rejection");
        require(target_rng() == rng_canon(), std::string(why) + ": live sampler RNG must survive rejection");
    };

    std::vector<char> truncated = valid; truncated.resize(valid.size() / 2);
    const auto truncated_path = (dir / "truncated.bin").string();
    write_file_bytes(truncated_path, truncated);
    expect_reject(truncated_path, "truncated training state must be rejected");

    std::vector<char> bad_magic = valid; bad_magic[0] = static_cast<char>(bad_magic[0] ^ 0x03);
    const auto bad_magic_path = (dir / "bad-magic.bin").string();
    write_file_bytes(bad_magic_path, bad_magic);
    expect_reject(bad_magic_path, "corrupt magic must be rejected");

    std::vector<char> bad_version = valid; bad_version[7] = 0x02;
    const auto bad_version_path = (dir / "bad-version.bin").string();
    write_file_bytes(bad_version_path, bad_version);
    expect_reject(bad_version_path, "unsupported version must be rejected");

    std::vector<char> bad_config = valid; bad_config[8] = 6;
    const auto bad_config_path = (dir / "bad-config.bin").string();
    write_file_bytes(bad_config_path, bad_config);
    expect_reject(bad_config_path, "configuration mismatch must be rejected");

    std::vector<char> trailing = valid; trailing.push_back('\x00');
    const auto trailing_path = (dir / "trailing.bin").string();
    write_file_bytes(trailing_path, trailing);
    expect_reject(trailing_path, "trailing bytes must be rejected");

    Config other_cfg = cfg; other_cfg.vocab_size = 6;
    ModelWeights other; other.init(other_cfg);
    const ModelWeights other_snapshot = other;
    std::mt19937 other_rng(1); std::uint64_t other_tokens = 321;
    const Adam other_opt_snapshot = target_opt;
    require(!load_training_state(other, target_opt, other_rng, other_tokens, state_path),
            "state file must be rejected against a dimension-mismatched live model");
    require(other.mwte == other_snapshot.mwte && other.vwte == other_snapshot.vwte,
            "rejected load must not mutate the mismatched live model");
    require(target_opt.step == other_opt_snapshot.step && target_opt.lr == other_opt_snapshot.lr &&
            target_opt.lr_total_iters == other_opt_snapshot.lr_total_iters,
            "rejected load must not mutate the optimizer");
    std::filesystem::remove_all(dir);
    std::cout << "training-state loader rejects invalid files without mutating live state\n";
}
static void test_resumed_training_matches_uninterrupted() {
    const auto dir = test_dir("resume-equivalence");
    Config cfg; cfg.vocab_size = 5; cfg.block_size = 4; cfg.n_embd = 4; cfg.n_head = 2; cfg.n_layer = 1;
    std::vector<int> data;
    for (int i = 0; i < 64; ++i) data.push_back(i % 5);
    const int T = cfg.block_size;
    auto run_iters = [&](ModelWeights& W, Adam& opt, std::mt19937& drng, int count, std::vector<float>& losses) {
        ForwardCache cache;
        for (int i = 0; i < count; ++i) {
            std::uniform_int_distribution<int> pos_dist(0, static_cast<int>(data.size()) - T - 2);
            const int start = pos_dist(drng);
            const std::vector<int> seq(data.begin() + start, data.begin() + start + T + 1);
            const float loss = forward(cfg, W, seq, cache, true);
            require(std::isfinite(loss), "training loss must stay finite");
            backward(cfg, W, seq, cache);
            opt.update_all(W);
            losses.push_back(loss);
        }
    };
    rng.seed(42); ModelWeights Wa; Wa.init(cfg);
    Adam optA; optA.lr = 0.01f; optA.lr_total_iters = 6;
    std::mt19937 rngA(0);
    std::vector<float> lossesA;
    run_iters(Wa, optA, rngA, 6, lossesA);

    rng.seed(42); ModelWeights Wb; Wb.init(cfg);
    Adam optB; optB.lr = 0.01f; optB.lr_total_iters = 6;
    std::mt19937 rngB(0);
    std::vector<float> lossesB;
    run_iters(Wb, optB, rngB, 2, lossesB);
    const auto model_path = (dir / "model.bin").string();
    const auto state_path = (dir / "model.bin.state").string();
    require(save_weights(Wb, model_path), "weights save should succeed");
    require(save_training_state(Wb, optB, rngB, static_cast<std::uint64_t>(data.size()), state_path),
            "state save should succeed");

    rng.seed(42); ModelWeights Wb2; Wb2.init(cfg);
    Adam optB2; optB2.lr_total_iters = 6;
    std::mt19937 rngB2(0);
    std::uint64_t tokens2 = 0;
    require(load_weights(Wb2, model_path), "weights load should succeed");
    require(load_training_state(Wb2, optB2, rngB2, tokens2, state_path), "state load should succeed");
    require(tokens2 == static_cast<std::uint64_t>(data.size()), "corpus token count must round-trip");
    std::vector<float> lossesB2;
    run_iters(Wb2, optB2, rngB2, 4, lossesB2);

    require_weights_equal(Wa, Wb2, "resumed training must reproduce uninterrupted weights bitwise");
    require(lossesB.size() + lossesB2.size() == lossesA.size(), "loss sequence length mismatch");
    for (std::size_t i = 0; i < lossesA.size(); ++i) {
        const float resumed_loss = (i < lossesB.size()) ? lossesB[i] : lossesB2[i - lossesB.size()];
        require(resumed_loss == lossesA[i], "resumed loss sequence must equal the uninterrupted sequence bitwise");
    }
    require(optB2.step == optA.step, "step counters must match after resume");
    std::filesystem::remove_all(dir);
    std::cout << "interrupted+resumed training is bitwise-equal to uninterrupted training\n";
}
static void test_cli_resume_paths() {
    const auto dir = test_dir("cli-resume-paths");
    const auto old_cwd = std::filesystem::current_path();
    std::filesystem::current_path(dir);
    const auto data_path = dir / "corpus.txt";
    { std::ofstream out(data_path); for (int i = 0; i < 80; ++i) out << ((i % 2 == 0) ? 'a' : 'b'); }
    std::string data = data_path.string();
    std::string model = (dir / "model.bin").string();
    std::string state = model + ".state";

    char a0[] = "nanolm", a1[] = "train", a3[] = "2", a4[] = "0.003",
         a5[] = "8", a6[] = "2", a7[] = "1", a8[] = "16", a9[] = "4";
    char* train_argv[] = {a0, a1, data.data(), model.data(), a3, a4, a5, a6, a7, a8, a9};
    require(nanolm_application_main(11, train_argv) == 0, "small CLI training run must succeed");
    require(std::filesystem::exists(model), "training must write model checkpoint");
    require(std::filesystem::exists(state), "training must write optimizer-state checkpoint");
    const std::vector<char> model_before = read_file_bytes(model);
    const std::vector<char> state_backup = read_file_bytes(state);

    char r1[] = "resume", a10[] = "2", a11[] = "4";
    char* resume_argv[] = {a0, r1, data.data(), model.data(), a10, a11};

    std::filesystem::remove(state);
    require(nanolm_application_main(6, resume_argv) != 0, "resume with missing state file must fail");
    require(read_file_bytes(model) == model_before, "failed resume must not rewrite model checkpoint");

    write_file_bytes(state, state_backup);
    require(nanolm_application_main(6, resume_argv) == 0, "resume with valid state must succeed");
    require(std::filesystem::exists(state), "resumed run must refresh the optimizer-state checkpoint");
    const std::vector<char> model_after_resume = read_file_bytes(model);

    std::vector<char> truncated = read_file_bytes(state);
    truncated.resize(truncated.size() / 2);
    write_file_bytes(state, truncated);
    require(nanolm_application_main(6, resume_argv) != 0, "resume with truncated state must fail");
    require(read_file_bytes(model) == model_after_resume, "failed resume must not rewrite model checkpoint");

    std::filesystem::current_path(old_cwd);
    std::filesystem::remove_all(dir);
    std::cout << "CLI resume success and failure paths behave correctly\n";
}
int main(int argc,char** argv) {
    const std::map<std::string,std::function<void()>> tests{
        {"attention_head_cache_and_gradient",test_attention_head_cache_and_gradient},
        {"next_token_full_window_reference",test_next_token_full_window_reference},
        {"parameter_count_matches_allocations",test_parameter_count_matches_allocations},
        {"checkpoint_rejects_before_allocation",test_checkpoint_rejects_before_allocation},
        {"cli_training_and_save_failures",test_cli_training_and_save_failures},
        {"isolated_deterministic_replay",test_isolated_deterministic_replay},
        {"training_state_round_trip_exact",test_training_state_round_trip_exact},
        {"training_state_rejects_invalid_and_preserves_live_state",test_training_state_rejects_invalid_and_preserves_live_state},
        {"resumed_training_matches_uninterrupted",test_resumed_training_matches_uninterrupted},
        {"cli_resume_paths",test_cli_resume_paths}};
    try{
        if(argc==3&&std::string(argv[1])=="--test"){
            const auto it=tests.find(argv[2]);if(it==tests.end())throw std::runtime_error("unknown test");
            std::cout<<"[TEST] "<<it->first<<"\n";it->second();std::cout<<"[PASS] "<<it->first<<"\n";return 0;
        }
        if(argc!=1)throw std::runtime_error("usage: run_tests [--test TEST_NAME]");
        int passed=0;for(const auto& test:tests){std::cout<<"[TEST] "<<test.first<<"\n";test.second();++passed;std::cout<<"[PASS] "<<test.first<<"\n";}
        std::cout<<"SUMMARY: "<<passed<<"/"<<tests.size()<<" passed\n";return 0;
    }catch(const std::exception& e){std::cerr<<"[FAIL] "<<e.what()<<"\n";return 1;}
}
