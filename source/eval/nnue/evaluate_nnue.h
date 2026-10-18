// header used in NNUE evaluation function
// NNUE評価関数で用いるheader

#ifndef CLASSIC_NNUE_EVALUATE_NNUE_H_INCLUDED
#define CLASSIC_NNUE_EVALUATE_NNUE_H_INCLUDED

#include "../../config.h"

#if defined(EVAL_NNUE)

#include "nnue_feature_transformer.h"
#include "nnue_architecture.h"
#include "../../misc.h"
#include "../../memory.h"
#include "../../shm.h"

#if defined(SFNNwoPSQT)
#ifndef NNUE_SFNN_HAND_BUCKETS
#define NNUE_SFNN_HAND_BUCKETS 1
#endif
#ifndef NNUE_SFNN_KING_BUCKETS
#define NNUE_SFNN_KING_BUCKETS 9
#endif
#ifndef NNUE_SFNN_PROGRESS_BUCKETS
#define NNUE_SFNN_PROGRESS_BUCKETS 1
#endif
#endif

namespace YaneuraOu {
class NumaConfig;
class NumaReplicatedAccessToken;
class Position;

namespace Eval::NNUE {

	#define EvalFileDefaultName "nn.bin"

#if defined(SFNNwoPSQT) && NNUE_SFNN_PROGRESS_BUCKETS != 1
namespace Progress {

	// SFNNのLayerStack選択に使う進行度計算パラメーター。
	// EvalDir/progress.bin: headerless little-endian doubles, without bias.
	struct Parameters {
		static constexpr int kWeightCount = int(SQ_NB) * int(Eval::fe_end);

		Tools::Result ReadParameters(std::istream& stream);

		std::int64_t SumQ16(const Position& pos) const;
		int BucketIndex(const Position& pos, int bucket_count) const;

		std::int32_t weights_q16_[SQ_NB][Eval::fe_end] = {};
	};

} // namespace Progress
#endif

	// Hash value of evaluation function structure
	// 評価関数の構造のハッシュ値
#if defined(SFNNwoPSQT)
	constexpr std::uint32_t kSfnnBaseHashValue = 0x3c203b32u;
	constexpr std::uint32_t kHashValue = kSfnnBaseHashValue;
	constexpr int kLayerStacks = LayerStacks;
#else
	constexpr std::uint32_t kHashValue =
	    FeatureTransformer::GetHashValue() ^ Network::GetHashValue();
	constexpr int kLayerStacks = 1;
#endif

	// NNUE評価関数パラメーターを格納する統合構造体。
	// 全メンバーが生配列で構成されており trivially copyable であるため、
	// プロセス間共有メモリに直接配置できる。
	struct NnueNetworks {
		FeatureTransformer feature_transformer;
#if defined(SFNNwoPSQT) && NNUE_SFNN_PROGRESS_BUCKETS != 1
		Progress::Parameters progress;
#endif
		Network network[kLayerStacks];
	};
	static_assert(std::is_trivially_copyable_v<NnueNetworks>,
		"NnueNetworks must be trivially copyable for shared memory support");

	// NNUE評価関数パラメーター（共有メモリまたはローカルメモリ上に配置）
	extern SystemWideSharedConstant<NnueNetworks> shared_networks;

	// 🌈 #97 NUMA 複製: 探索スレッドは自分の NUMA ノード上のコピーを thread_local で指す (bind_networks_to_numa)。
	//     指していない (メインスレッド、1 ノード構成、再読込直後) なら共有メモリのコピー。generation は評価関数の再読込で増え、
	//     古いコピーを指したままのスレッドが共有コピーへ落ちるようにする (ポインタ比較 + 世代比較だけなので評価のたびに呼んでも軽い)。
	extern thread_local const NnueNetworks* tl_networks;
	extern thread_local unsigned tl_networks_generation;
	extern unsigned networks_generation;
	inline const NnueNetworks& networks() {
		return (tl_networks != nullptr && tl_networks_generation == networks_generation) ? *tl_networks : *shared_networks;
	}
	// NumaConfig を登録する (resize_threads から。ノード数が変われば古いコピーは捨てる)。nullptr なら複製しない。
	void set_numa_config(const NumaConfig* cfg);
	// このスレッドを token のノードのコピーに結び付ける (無ければそのノード上で確保して作る)。Worker::ensure_network_replicated() から呼ぶ。
	void bind_networks_to_numa(const NumaReplicatedAccessToken& token);
	// コピーを全部捨てる (評価関数の再読込)。
	void drop_numa_networks();

	// 評価関数ファイル名
	extern const char* const kFileName;

	// 評価関数の構造を表す文字列を取得する
	std::string GetArchitectureString();

	// ヘッダを読み込む
	Tools::Result ReadHeader(std::istream& stream,
	    std::uint32_t* hash_value, std::string* architecture, std::uint32_t* version_out = nullptr);

	// ヘッダを書き込む
	bool WriteHeader(std::ostream& stream,
	    std::uint32_t hash_value, const std::string& architecture);

	// 評価関数パラメータを読み込む
	// On a sidecar read failure, replace failed_file with its path.
	Tools::Result ReadParameters(std::istream& stream, std::string* failed_file = nullptr);

	// 評価関数パラメータを書き込む
	bool WriteParameters(std::ostream& stream);

} // namespace Eval::NNUE
} // namespace YaneuraOu

// NnueNetworks のコンテンツハッシュ。共有メモリの名前に使われる。
// 同一の評価関数パラメーターを持つプロセス同士で自動的にメモリが共有される。
template<>
struct std::hash<YaneuraOu::Eval::NNUE::NnueNetworks> {
	std::size_t operator()(const YaneuraOu::Eval::NNUE::NnueNetworks& n) const noexcept {
		return static_cast<std::size_t>(
			YaneuraOu::hash_bytes(reinterpret_cast<const char*>(&n), sizeof(n)));
	}
};

#endif  // defined(EVAL_NNUE)

#endif // #ifndef NNUE_EVALUATE_NNUE_H_INCLUDED
