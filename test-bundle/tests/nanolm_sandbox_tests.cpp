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
int main(int argc,char** argv) {
    const std::map<std::string,std::function<void()>> tests{
        {"attention_head_cache_and_gradient",test_attention_head_cache_and_gradient},
        {"next_token_full_window_reference",test_next_token_full_window_reference},
        {"parameter_count_matches_allocations",test_parameter_count_matches_allocations},
        {"checkpoint_rejects_before_allocation",test_checkpoint_rejects_before_allocation},
        {"cli_training_and_save_failures",test_cli_training_and_save_failures},
        {"isolated_deterministic_replay",test_isolated_deterministic_replay}};
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
