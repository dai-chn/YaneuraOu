// NNUE評価関数の計算に関するコード

#include "../../config.h"

#if defined(EVAL_NNUE)

#include <fstream>
#include <sstream>
#include <vector>

#define INCBIN_SILENCE_BITCODE_WARNING
#include "../../incbin/incbin.h"

#include "../../types.h"
#include "../../evaluate.h"
#include "../../position.h"
#include "../../memory.h"
#include "../../usi.h"

#if defined(USE_EVAL_HASH)
#include "../evalhash.h"
#endif

#include "evaluate_nnue.h"

namespace YaneuraOu::Eval::NNUE {
extern int FV_SCALE;
extern int GTAIL_T;
extern int GTAIL_GAIN;
}

// ============================================================
// EvLog: 探索文脈つき評価呼び出しの統計記録 (Matryoshka 設計用, report20/33)
//
// search()/qsearch() のノード入口で窓文脈 (alpha/beta/depth/ply) を
// thread_local に置き、Eval::evaluate() の復路で 16B レコードを書く。
// 環境変数 EVAL_LOG_PATH が無ければ完全に無効 (分岐1つのみ)。
// EVAL_LOG_SAMPLE=N で 1/N サンプリング (既定 1 = 全件)。
// 計測は Threads=1 前提 (ファイルは thread_local に分離するので複数でも壊れない)。
// ============================================================
#include <cstdlib>
#include <cstdio>
#include <cstdint>

namespace YaneuraOu { namespace EvLog {
struct Ctx {
    int32_t alpha, beta;
    int16_t depth;
    uint8_t ply, flags;  // flags: bit0=PvNode, bit1=qsearch
};
thread_local Ctx ctx = {0, 0, 0, 0, 0};

#pragma pack(push, 1)
struct Rec {
    int32_t alpha, beta, value;
    int16_t depth;
    uint8_t ply, flags;  // flags: ctx.flags | (path << 2)  path: 0=fresh 1=acc 2=evalhash
};
#pragma pack(pop)
static_assert(sizeof(Rec) == 16, "Rec must be 16 bytes");

struct Sink {
    FILE* f = nullptr;
    uint32_t counter = 0;
    uint32_t sample = 1;
    bool checked = false;
    ~Sink() {
        if (f)
            fclose(f);
    }
    void ensure() {
        checked = true;
        const char* p = std::getenv("EVAL_LOG_PATH");
        if (!p)
            return;
        const char* s = std::getenv("EVAL_LOG_SAMPLE");
        if (s) {
            int v = std::atoi(s);
            if (v > 1)
                sample = (uint32_t)v;
        }
        f = std::fopen(p, "ab");
    }
};
thread_local Sink sink;

inline void log(int32_t value, uint8_t path) {
    if (!sink.checked)
        sink.ensure();
    if (!sink.f)
        return;
    if (++sink.counter % sink.sample != 0)
        return;
    Rec r{ctx.alpha, ctx.beta, value, ctx.depth, ctx.ply,
          (uint8_t)(ctx.flags | (path << 2))};
    std::fwrite(&r, sizeof(r), 1, sink.f);
}
}}  // namespace YaneuraOu::EvLog

// ============================================================
// EvalGateSim: 合成 small eval によるゲートのシミュレーション (report45 Phase-0)
//
// 狙い: 「区間型の軽 eval で境界判定を肩代わりさせたとき、探索の品質がどれだけ落ちるか」を
//       ネットも accumulator の改造も無しに測る。
//
//   small(P) = large(P) + noise(P)      noise は局面キーから決定的に生成 (標準偏差 GateE cp)
//   区間      = small ± (GateC/100)*GateE
//   hi <= alpha           → fail low  側で確定 → small を返す
//   lo >= beta            → fail high 側で確定 → small を返す
//   それ以外               → escalate           → large を返す
//
// ★返すのは点推定 (small) であって区間の端ではない。境界値を返すと親に渡る bound が
//   緩くなり木が膨らむ (report45 §5.3)。保守性は escalate の判断側で担保する。
// ★ノイズは局面キーのみに依存させる。訪問ごとに変わると設計に無い不安定性が混入して
//   測定が交絡する。
//
// 本シミュレーションは large を常に計算するので **速度は測れない**。測れるのは
// 「コストがゼロだったと仮定したときの品質劣化」= 損失側だけ。
// GateE=0 (既定) で完全に無効。ENABLE_EVAL_GATE_SIM 未定義なら存在しない。
// ============================================================
#if defined(ENABLE_EVAL_GATE_SIM)
#include <cmath>

namespace YaneuraOu { namespace EvalGateSim {

int E = 0;    // ノイズ標準偏差 [cp]。0 で無効
int C = 200;  // 区間半幅 = (C/100) * E

struct Stats {
    uint64_t decided_lo = 0, decided_hi = 0, escalated = 0, no_ctx = 0;
    uint64_t q_decided = 0, q_escalated = 0;
    ~Stats();
};
Stats stats;  // Threads=1 前提

// splitmix64: 局面キー → 決定的な擬似乱数
inline uint64_t mix(uint64_t x) {
    x += 0x9e3779b97f4a7c15ULL;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
    return x ^ (x >> 31);
}

// 標準正規 (Box-Muller)。局面キーのみに依存する。
inline double gauss(uint64_t key) {
    const uint64_t a = mix(key);
    const uint64_t b = mix(key ^ 0xdeadbeefcafef00dULL);
    const double u1 = double((a >> 11) + 1) * (1.0 / 9007199254740993.0);
    const double u2 = double((b >> 11) + 1) * (1.0 / 9007199254740993.0);
    return std::sqrt(-2.0 * std::log(u1)) * std::cos(6.283185307179586 * u2);
}

// large: 本物の評価値。返り値: 探索へ返す値。
//
// ★noinline 必須。inline 展開させると evaluate() → ComputeScore →
//   accumulator 更新の経路まで codegen が変わり、**ゲートが無効 (E=0) でも**
//   探索開始直後にアクセス違反で落ちた (2026-08-16 に実測)。
//   測定用コードが本体の最適化に干渉しないよう、呼び出し境界で切っておく。
__attribute__((noinline)) int32_t apply(int32_t large, uint64_t key) {
    if (E <= 0)
        return large;

    const EvLog::Ctx& c = EvLog::ctx;
    // 探索外からの呼び出し (Position::set / eval コマンド等) は窓が無いので素通し。
    // 探索中は必ず alpha < beta。
    if (c.alpha >= c.beta) {
        stats.no_ctx++;
        return large;
    }

    const int32_t noise = int32_t(std::lround(gauss(key) * double(E)));
    int32_t       small = large + noise;
    // mate スコア域に食い込ませない
    small = small < -30000 ? -30000 : (small > 30000 ? 30000 : small);

    const int32_t half = int32_t((int64_t)E * (int64_t)C / 100);
    const bool    q    = (c.flags & 2) != 0;

    if (small + half <= c.alpha) {  // 上界が alpha 以下 → fail low 確定
        stats.decided_lo++;
        if (q) stats.q_decided++;
        return small;
    }
    if (small - half >= c.beta) {   // 下界が beta 以上 → fail high 確定
        stats.decided_hi++;
        if (q) stats.q_decided++;
        return small;
    }
    stats.escalated++;
    if (q) stats.q_escalated++;
    return large;
}

Stats::~Stats() {
    const char* p = std::getenv("EVAL_GATE_STATS");
    if (!p || E <= 0)
        return;
    FILE* f = std::fopen(p, "a");
    if (!f)
        return;
    const uint64_t dec   = decided_lo + decided_hi;
    const uint64_t total = dec + escalated;
    std::fprintf(f,
                 "E=%d C=%d total=%llu decided=%llu decided_lo=%llu decided_hi=%llu "
                 "escalated=%llu escalate_rate=%.5f no_ctx=%llu "
                 "q_decided=%llu q_escalated=%llu q_escalate_rate=%.5f\n",
                 E, C, (unsigned long long)total, (unsigned long long)dec,
                 (unsigned long long)decided_lo, (unsigned long long)decided_hi,
                 (unsigned long long)escalated,
                 total ? double(escalated) / double(total) : 0.0,
                 (unsigned long long)no_ctx, (unsigned long long)q_decided,
                 (unsigned long long)q_escalated,
                 (q_decided + q_escalated)
                     ? double(q_escalated) / double(q_decided + q_escalated)
                     : 0.0);
    std::fclose(f);
}
}}  // namespace YaneuraOu::EvalGateSim
#endif  // ENABLE_EVAL_GATE_SIM

// ============================================================
//              旧評価関数のためのヘルパー
// ============================================================

#if defined(USE_CLASSIC_EVAL)
using namespace YaneuraOu;
void add_options_(OptionsMap& options, ThreadPool& threads);

namespace {
YaneuraOu::OptionsMap* options_ptr;
YaneuraOu::ThreadPool* threads_ptr;
}

// 📌 旧Options、旧Threadsとの互換性のための共通のマクロ 📌
#define Options (*options_ptr)
#define Threads (*threads_ptr)

namespace YaneuraOu::Eval {
void add_options(OptionsMap& options, ThreadPool& threads) {
    options_ptr = &options;
    threads_ptr = &threads;
    add_options_(options, threads);
}
}
// ============================================================

// 評価関数を読み込み済みであるか
bool        eval_loaded   = false;
std::string last_eval_dir = "None";

// 📌 この評価関数で追加したいエンジンオプションはここで追加する。
void add_options_(OptionsMap& options, ThreadPool& threads) {

#if defined(EVAL_LEARN)
    // isreadyタイミングで評価関数を読み込まれると、新しい評価関数の変換のために
    // test evalconvertコマンドを叩きたいのに、その新しい評価関数がないがために
    // このコマンドの実行前に異常終了してしまう。
    // そこでこの隠しオプションでisready時の評価関数の読み込みを抑制して、
    // test evalconvertコマンドを叩く。
    Options("SkipLoadingEval", Option(false));
#endif

#if defined(NNUE_EMBEDDING_OFF)
    const char* default_eval_dir = "eval";
#else
	// メモリから読み込む。
    const char* default_eval_dir = "<internal>";
#endif
    Options.add("EvalDir", Option(default_eval_dir, [](const Option& o) {
                    std::string eval_dir = std::string(o);
                    if (last_eval_dir != eval_dir)
                    {
                        // 評価関数フォルダ名の変更に際して、評価関数ファイルの読み込みフラグをクリアする。
                        last_eval_dir = eval_dir;
                        eval_loaded   = false;
                    }
                    return std::nullopt;
                }));

    // NNUEのFV_SCALEの値
    Options.add("FV_SCALE", Option(16, 1, 128, [&](const Option& o) {
                    YaneuraOu::Eval::NNUE::FV_SCALE = int(o);
                    return std::nullopt;
                }));

    // テールゲイン単調変換 g (report27: 極端度の後段調整)。GTAIL_GAIN=100 で恒等 (stock と同一)。
    Options.add("GTAIL_T", Option(400, 0, 8000, [&](const Option& o) {
                    YaneuraOu::Eval::NNUE::GTAIL_T = int(o);
                    return std::nullopt;
                }));
    Options.add("GTAIL_GAIN", Option(100, 100, 2000, [&](const Option& o) {
                    YaneuraOu::Eval::NNUE::GTAIL_GAIN = int(o);
                    return std::nullopt;
                }));

#if defined(ENABLE_EVAL_GATE_SIM)
    // 合成 small eval ゲートのシミュレーション (report45 Phase-0)。GateE=0 で無効。
    Options.add("GateE", Option(0, 0, 2000, [&](const Option& o) {
                    YaneuraOu::EvalGateSim::E = int(o);
                    return std::nullopt;
                }));
    Options.add("GateC", Option(200, 0, 1000, [&](const Option& o) {
                    YaneuraOu::EvalGateSim::C = int(o);
                    return std::nullopt;
                }));
#endif
}
#endif

// Macro to embed the default efficiently updatable neural network (NNUE) file
// data in the engine binary (using incbin.h, by Dale Weiler).
// This macro invocation will declare the following three variables
//     const unsigned char        gEmbeddedNNUEData[];  // a pointer to the embedded data
//     const unsigned char *const gEmbeddedNNUEEnd;     // a marker to the end
//     const unsigned int         gEmbeddedNNUESize;    // the size of the embedded file
// Note that this does not work in Microsoft Visual Studio.

// デフォルトの効率的に更新可能なニューラルネットワーク（NNUE）ファイルの
// データをエンジンのバイナリに埋め込むためのマクロ
// （Dale Weiler 氏の incbin.h を使用）。
// このマクロを使うことで、以下の3つの変数が宣言されます：
//     const unsigned char        gEmbeddedNNUEData[];  // 埋め込まれたデータへのポインタ
//     const unsigned char *const gEmbeddedNNUEEnd;     // データの終端を示すマーカー
//     const unsigned int         gEmbeddedNNUESize;    // 埋め込まれたファイルのサイズ
// なお、この方法は Microsoft Visual Studio では動作しません。

#if !defined(_MSC_VER) && !defined(NNUE_EMBEDDING_OFF)
INCBIN(EmbeddedNNUE, EvalFileDefaultName);
#else
const unsigned char        gEmbeddedNNUEData[1] = { 0x0 };
const unsigned char* const gEmbeddedNNUEEnd = &gEmbeddedNNUEData[1];
const unsigned int         gEmbeddedNNUESize = 1;
#endif

// NNUEの埋め込みデータ型

namespace {

	struct EmbeddedNNUE {
		EmbeddedNNUE(const unsigned char* embeddedData,
			const unsigned char* embeddedEnd,
			const unsigned int   embeddedSize) :
			data(embeddedData),
			end(embeddedEnd),
			size(embeddedSize) {
		}
		const unsigned char* data;
		const unsigned char* end;
		const unsigned int   size;
	};

	//EmbeddedNNUE get_embedded(EmbeddedNNUEType type) {
	//	if (type == EmbeddedNNUEType::BIG)
	//		return EmbeddedNNUE(gEmbeddedNNUEBigData, gEmbeddedNNUEBigEnd, gEmbeddedNNUEBigSize);
	//	else
	//		return EmbeddedNNUE(gEmbeddedNNUESmallData, gEmbeddedNNUESmallEnd, gEmbeddedNNUESmallSize);
	//}

	// ⇨  StockfishはNNUEとして大きなnetworkと小さなnetworkがある。

	EmbeddedNNUE get_embedded() {
		return EmbeddedNNUE(gEmbeddedNNUEData, gEmbeddedNNUEEnd, gEmbeddedNNUESize);
	}
}


namespace YaneuraOu {
namespace Eval {
namespace NNUE {

	int FV_SCALE = 16; // 水匠5では24がベストらしいのでエンジンオプション"FV_SCALE"で変更可能にした。

	// テールゲイン単調変換 g のパラメータ (report27)。GTAIL_GAIN=100 (=1.0倍) で恒等。
	int GTAIL_T = 400;
	int GTAIL_GAIN = 100;

    // NNUE評価関数パラメーター（共有メモリまたはローカルメモリ上に配置）
    SystemWideSharedConstant<NnueNetworks> shared_networks;

    // 評価関数ファイル名
    const char* const kFileName = EvalFileDefaultName;

    // 評価関数の構造を表す文字列を取得する
    std::string GetArchitectureString() {
        const std::string base = "Features=" + FeatureTransformer::GetStructureString() +
			",Network=" + Network::GetStructureString();
#if defined(SFNNwoPSQT)
		return "ModelType=SFNNWithoutPsqt;" + base + "{LayerStack=" + std::to_string(kLayerStacks) + "}";
#else
		return base;
#endif
    }

namespace {
	namespace Detail {

		// 評価関数パラメータを読み込む（参照版）
		template <typename T>
		Tools::Result ReadParameters(std::istream& stream, T& obj) {
			std::uint32_t header;
			stream.read(reinterpret_cast<char*>(&header), sizeof(header));
			if (!stream) return Tools::ResultCode::FileReadError;
			// hash値、古い評価関数ファイルに対して一致するとは限らないので、警告に変更する。
			if (header != T::GetHashValue())
				sync_cout << "info string Warning : nn.bin hash mismatch." << sync_endl;
			return obj.ReadParameters(stream);
		}

		// 評価関数パラメータを書き込む（参照版）
		template <typename T>
		bool WriteParameters(std::ostream& stream, const T& obj) {
			constexpr std::uint32_t header = T::GetHashValue();
			stream.write(reinterpret_cast<const char*>(&header), sizeof(header));
			return obj.WriteParameters(stream);
		}

	}  // namespace Detail

	// テンポラリにパラメータを読み込み、共有メモリに配置する。
	// 同じパラメータを持つ他プロセスが既に共有メモリを作成済みなら、そちらを参照する。
	Tools::Result LoadAndShare(std::istream& stream) {
		// テンポラリ領域にパラメータを読み込む
		auto tmp = make_unique_large_page<NnueNetworks>();

		std::uint32_t hash_value;
		std::string architecture;
		Tools::Result result = ReadHeader(stream, &hash_value, &architecture, nullptr);
		if (result.is_not_ok()) return result;
		if (hash_value != kHashValue) {
			sync_cout << "info string Warning: NNUE hash mismatch: expected " << kHashValue
				<< " got " << hash_value
				<< " arch_in_file=" << architecture
				<< " arch_expected=" << GetArchitectureString()
				<< sync_endl;
#if defined(NNUE_FT_SCRELU) || defined(NNUE_FT_PAIRWISE)
			// exp013 arm1/arm3: マーカー XOR により hash は必ず不一致になる。
			// 異種ネットとの混載を hard error で拒否する (spec §3 C1)。
			sync_cout << "info string Error : NNUE hash mismatch is fatal for this build (refusing to load)." << sync_endl;
			return Tools::ResultCode::FileMismatch;
#endif
		}

		result = Detail::ReadParameters<FeatureTransformer>(stream, tmp->feature_transformer);
		if (result.is_not_ok()) {
			sync_cout << "info string NNUE feature params read failed: " << result.to_string() << sync_endl;
			return result;
		}
		for (int i = 0; i < kLayerStacks; ++i) {
			result = Detail::ReadParameters<Network>(stream, tmp->network[i]);
			if (result.is_not_ok()) {
				sync_cout << "info string NNUE network params read failed at stack " << i << ": " << result.to_string() << sync_endl;
				return result;
			}
		}

		if (!stream || stream.peek() != std::ios::traits_type::eof())
			return Tools::ResultCode::FileCloseError;

		// 共有メモリに配置（同一ハッシュの共有メモリが既に存在すればそちらを参照）
		shared_networks = SystemWideSharedConstant<NnueNetworks>(*tmp);

		auto status = shared_networks.get_status();
		if (status == SystemWideSharedConstantAllocationStatus::SharedMemory)
			sync_cout << "info string NNUE shared memory: using shared memory" << sync_endl;
		else if (status == SystemWideSharedConstantAllocationStatus::LocalMemory)
			sync_cout << "info string NNUE shared memory: fallback to local memory" << sync_endl;

		return Tools::ResultCode::Ok;
	}

	}  // namespace
    // ヘッダを読み込む
    Tools::Result ReadHeader(std::istream& stream,
        std::uint32_t* hash_value, std::string* architecture, std::uint32_t* version_out) {
        std::uint32_t version = 0, size = 0;
        stream.read(reinterpret_cast<char*>(&version), sizeof(version));
        stream.read(reinterpret_cast<char*>(hash_value), sizeof(*hash_value));
        stream.read(reinterpret_cast<char*>(&size), sizeof(size));
		if (!stream) return Tools::ResultCode::FileReadError;
		if (version_out)
			*version_out = version;
        if (version != kVersion) {
			sync_cout << "info string NNUE header version mismatch: expected " << kVersion
				<< " got " << version << sync_endl;
			return Tools::ResultCode::FileMismatch;
		}
        architecture->resize(size);
        stream.read(&(*architecture)[0], size);
		return !stream.fail() ? Tools::ResultCode::Ok : Tools::ResultCode::FileReadError;
    }

    // ヘッダを書き込む
    bool WriteHeader(std::ostream& stream,
        std::uint32_t hash_value, const std::string& architecture) {
        stream.write(reinterpret_cast<const char*>(&kVersion), sizeof(kVersion));
        stream.write(reinterpret_cast<const char*>(&hash_value), sizeof(hash_value));
        const std::uint32_t size = static_cast<std::uint32_t>(architecture.size());
        stream.write(reinterpret_cast<const char*>(&size), sizeof(size));
        stream.write(architecture.data(), size);
        return !stream.fail();
    }

    	// 評価関数パラメータを読み込む
    	Tools::Result ReadParameters(std::istream& stream) {
    		return LoadAndShare(stream);
    	}
    // 評価関数パラメータを書き込む
    bool WriteParameters(std::ostream& stream) {
        if (!WriteHeader(stream, kHashValue, GetArchitectureString())) return false;
        if (!Detail::WriteParameters<FeatureTransformer>(stream, networks().feature_transformer)) return false;
        for (int i = 0; i < kLayerStacks; ++i) {
            if (!Detail::WriteParameters<Network>(stream, networks().network[i])) return false;
        }
        return !stream.fail();
    }

    // 差分計算ができるなら進める
    static void UpdateAccumulatorIfPossible(const Position& pos) {
        networks().feature_transformer.UpdateAccumulatorIfPossible(pos);
    }

#if defined(SFNNwoPSQT)
    // レイヤースタックの選択。双方の玉の段に応じて9通りに分岐させる。
    static int stack_index_for_nnue(const Position& pos) {
        constexpr int kFToIndex[] = { 0, 0, 0, 3, 3, 3, 6, 6, 6 };
        constexpr int kEToIndex[] = { 0, 0, 0, 1, 1, 1, 2, 2, 2 };
        const auto stm = pos.side_to_move();
        const auto f_king = pos.square<KING>(stm);
        const auto e_king = pos.square<KING>(~stm);
        const auto f_rank = stm == BLACK ? rank_of(f_king) : rank_of(Inv(f_king));
        const auto e_rank = stm == BLACK ? rank_of(Inv(e_king)) : rank_of(e_king);
        int idx = kFToIndex[f_rank] + kEToIndex[e_rank];
        if (idx < 0) idx = 0;
        if (idx >= kLayerStacks) idx = kLayerStacks - 1;
        return idx;
    }
#endif

    // 評価値を計算する
    static Value ComputeScore(const Position& pos, bool refresh = false) {
        auto& accumulator = pos.state()->accumulator;
        if (!refresh && accumulator.computed_score) {
            return accumulator.score;
        }

        alignas(kCacheLineSize) TransformedFeatureType
            transformed_features[FeatureTransformer::kBufferSize];
        networks().feature_transformer.Transform(pos, transformed_features, refresh);
        alignas(kCacheLineSize) char buffer[Network::kBufferSize];
#if defined(SFNNwoPSQT)
        const auto bucket = stack_index_for_nnue(pos);
        const auto output = networks().network[bucket].Propagate(transformed_features, buffer);
#else
        const auto output = networks().network[0].Propagate(transformed_features, buffer);
#endif

        // VALUE_MAX_EVALより大きな値が返ってくるとaspiration searchがfail highして
        // 探索が終わらなくなるのでVALUE_MAX_EVAL以下であることを保証すべき。

        // この現象が起きても、対局時に秒固定などだとそこで探索が打ち切られるので、
        // 1つ前のiterationのときの最善手がbestmoveとして指されるので見かけ上、
        // 問題ない。このVALUE_MAX_EVALが返ってくるような状況は、ほぼ詰みの局面であり、
        // そのような詰みの局面が出現するのは終盤で形勢に大差がついていることが多いので
        // 勝敗にはあまり影響しない。

        // しかし、教師生成時などdepth固定で探索するときに探索から戻ってこなくなるので
        // そのスレッドの計算時間を無駄にする。またdepth固定対局でtime-outするようになる。

        auto score = static_cast<Value>(output[0] / FV_SCALE);

        // テールゲイン単調変換 g: |score| > GTAIL_T の超過分を (GTAIL_GAIN/100) 倍する単調変換。
        // GTAIL_GAIN=100 のとき恒等 (stock と同一)。中盤(|score|<=GTAIL_T)は不変、テールだけ非線形に伸縮。
        if (GTAIL_GAIN != 100) {
            int s = (int)score;
            int a = s < 0 ? -s : s;
            if (a > GTAIL_T) {
                long long gv = (long long)GTAIL_T + (long long)(a - GTAIL_T) * (long long)GTAIL_GAIN / 100;
                score = (Value)(s < 0 ? -(int)gv : (int)gv);
            }
        }

        // 1) ここ、下手にclipすると学習時には影響があるような気もするが…。
        // 2) accumulator.scoreは、差分計算の時に用いないので書き換えて問題ない。
        score = Math::clamp(score, -VALUE_MAX_EVAL, VALUE_MAX_EVAL);

        accumulator.score = score;
        accumulator.computed_score = true;
        return accumulator.score;
    }

}  // namespace NNUE

#if defined(USE_EVAL_HASH)

// HashTableに評価値を保存するために利用するクラス
struct alignas(16) ScoreKeyValue {
#if defined(USE_SSE2)
    ScoreKeyValue() = default;
    ScoreKeyValue(const ScoreKeyValue & other) {
        static_assert(sizeof(ScoreKeyValue) == sizeof(__m128i),
            "sizeof(ScoreKeyValue) should be equal to sizeof(__m128i)");
        _mm_store_si128(&as_m128i, other.as_m128i);
    }
    ScoreKeyValue& operator=(const ScoreKeyValue & other) {
        _mm_store_si128(&as_m128i, other.as_m128i);
        return *this;
    }
#endif

    // evaluate hashでatomicに操作できる必要があるのでそのための操作子
    void encode() {
#if defined(USE_SSE2)
        // ScoreKeyValue は atomic にコピーされるので key が合っていればデータも合っている。
#else
        key ^= score;
#endif
    }
    // decode()はencode()の逆変換だが、xorなので逆変換も同じ変換。
    void decode() { encode(); }

    union {
        struct {
            std::uint64_t key;
            std::uint64_t score;
        };
#if defined(USE_SSE2)
        __m128i as_m128i;
#endif
    };
};

// evaluateしたものを保存しておくHashTable(俗にいうehash)

struct EvaluateHashTable : HashTable<ScoreKeyValue> {};

EvaluateHashTable g_evalTable;
void EvalHash_Resize(size_t mbSize) { g_evalTable.resize(mbSize); }
void EvalHash_Clear() { g_evalTable.clear(); };

// prefetchする関数も用意しておく。
void prefetch_evalhash(const Key key) {
    constexpr auto mask = ~((u64)0x1f);
    prefetch((void*)((u64)g_evalTable[key] & mask));
}
#endif

// 評価関数ファイルを読み込む
void load_eval() {
    // 評価関数パラメーターを読み込み済みであるなら帰る。
    if (eval_loaded)
        return;

#if defined(EVAL_LEARN)
    if (!Options["SkipLoadingEval"])
#endif
    {
        const std::string dir_name = Options["EvalDir"];
    #if !defined(__EMSCRIPTEN__)
		const std::string file_name = NNUE::kFileName;
#else
		// WASM
        const std::string file_name = Options["EvalFile"];
    #endif
        const Tools::Result result = [&] {
            if (dir_name != "<internal>") {
#if defined(_WIN32)
                // 非ASCII (CP932 外) パス耐性: 起動フォルダ相対のファイルを wide(UTF-16) API で読む。
                // narrow fopen の ANSI コードページ変換を回避する (探索/評価の数値には影響しない)。
                std::vector<char> eval_data;
                const bool eval_ok = Directory::ReadBinaryFolderRelativeFileW(dir_name, file_name, eval_data);
                sync_cout << "info string loading eval file (wide) : "
                          << dir_name << "\\" << file_name << (eval_ok ? " [ok]" : " [not found]") << sync_endl;
                if (!eval_ok)
                    return Tools::Result(Tools::ResultCode::FileNotFound);
                class MemoryBuffer : public std::basic_streambuf<char> {
                    public: MemoryBuffer(char* p, size_t n) {
                        std::streambuf::setg(p, p, p + n);
                        std::streambuf::setp(p, p + n);
                    }
                };
                MemoryBuffer eval_buffer(eval_data.data(), eval_data.size());
                std::istream stream(&eval_buffer);
                return NNUE::ReadParameters(stream);
#else
                auto abs_eval_path = Path::Combine(Directory::GetBinaryFolder(), dir_name);
                const std::string file_path = Path::Combine(abs_eval_path, file_name);
                std::ifstream stream(file_path, std::ios::binary);
                sync_cout << "info string loading eval file : " << file_path << sync_endl;
                if (!stream.is_open())
                    return Tools::Result(Tools::ResultCode::FileNotFound);
                return NNUE::ReadParameters(stream);
#endif
            }
            else {
                // C++ way to prepare a buffer for a memory stream
                class MemoryBuffer : public std::basic_streambuf<char> {
                    public: MemoryBuffer(char* p, size_t n) {
                        std::streambuf::setg(p, p, p + n);
                        std::streambuf::setp(p, p + n);
                    }
                };

			    const auto embedded = get_embedded(/* embeddedType */);

                MemoryBuffer buffer(
                              const_cast<char*>(reinterpret_cast<const char*>(embedded.data)),
                              size_t(embedded.size));

                std::istream stream(&buffer);
                sync_cout << "info string loading eval file : <internal>" << sync_endl;

                return NNUE::ReadParameters(stream);
            }
        }();

        //      ASSERT(result);

        if (result.is_not_ok())
        {
            // 読み込みエラーのとき終了してくれないと困る。
            sync_cout << "Error! : failed to read " << file_name << " : " << result.to_string() << sync_endl;
            Tools::exit();
        }

		// 評価関数ファイルの読み込みが完了した。
		eval_loaded = true;
    }
}


// 評価関数。差分計算ではなく全計算する。
// Position::set()で一度だけ呼び出される。(以降は差分計算)
// 手番側から見た評価値を返すので注意。(他の評価関数とは設計がこの点において異なる)
// なので、この関数の最適化は頑張らない。
Value compute_eval(const Position& pos) {
    return NNUE::ComputeScore(pos, true);
}

// 合成 small ゲートの適用 (report45 Phase-0)。
// ENABLE_EVAL_GATE_SIM 未定義なら恒等 = 配布ビルドには存在しない。
inline Value eval_gate(Value v, const Position& pos) {
#if defined(ENABLE_EVAL_GATE_SIM)
    // ★局面キーの取得はゲートが有効なときだけ行う。引数として書くと呼び出し側で
    //   常に評価され、無効時にもコストと副作用が乗る。
    if (EvalGateSim::E <= 0)
        return v;
    return Value(EvalGateSim::apply((int32_t)v, (uint64_t)pos.state()->key()));
#else
    (void)pos;
    return v;
#endif
}

// 評価関数
Value evaluate(const Position& pos) {
    const auto& accumulator = pos.state()->accumulator;
    if (accumulator.computed_score) {
        EvLog::log((int32_t)accumulator.score, 1);
        return eval_gate(accumulator.score, pos);
    }

#if defined(USE_GLOBAL_OPTIONS)
    // GlobalOptionsでeval hashを用いない設定になっているなら
    // eval hashへの照会をskipする。
    if (!GlobalOptions.use_eval_hash) {
        ASSERT_LV5(pos.state()->materialValue == Eval::material(pos));
        return NNUE::ComputeScore(pos);
    }
#endif

#if defined(USE_EVAL_HASH)
    // evaluate hash tableにはあるかも。
    const Key key = pos.state()->key();
    ScoreKeyValue entry = *g_evalTable[key];
    entry.decode();
    if (entry.key == key) {
        // あった！
        EvLog::log((int32_t)entry.score, 2);
        return eval_gate(Value(entry.score), pos);
    }
#endif

    Value score = NNUE::ComputeScore(pos);
    EvLog::log((int32_t)score, 0);
#if defined(USE_EVAL_HASH)
    // せっかく計算したのでevaluate hash tableに保存しておく。
    entry.key = key;
    entry.score = score;
    entry.encode();
    *g_evalTable[key] = entry;
#endif

    // ★ eval hash には素の score を保存し、ゲートは返り値にだけ掛ける。
    //    ゲート後の値を保存すると、窓が変わった次の訪問で「粗い値」が素通しで
    //    再利用されてしまい、シミュレーションが設計と乖離する。
    return eval_gate(score, pos);
}

// 差分計算ができるなら進める
void evaluate_with_no_return(const Position& pos) {
    NNUE::UpdateAccumulatorIfPossible(pos);
}

// 現在の局面の評価値の内訳を表示する
void print_eval_stat(Position& /*pos*/) {
    std::cout << "--- EVAL STAT: not implemented" << std::endl;
}

} // namespace Eval
} // namespace YaneuraOu

#endif  // defined(EVAL_NNUE)
