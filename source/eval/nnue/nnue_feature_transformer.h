// A class that converts the input features of the NNUE evaluation function
// NNUE評価関数の入力特徴量の変換を行うクラス

#ifndef CLASSIC_NNUE_FEATURE_TRANSFORMER_H_INCLUDED
#define CLASSIC_NNUE_FEATURE_TRANSFORMER_H_INCLUDED

#include "../../config.h"

#if defined(EVAL_NNUE)

#if defined(SFNNwoPSQT)
#define USE_ELEMENT_WISE_MULTIPLY
#endif

// NNUE_FT_PAIRWISE (exp013 arm3): 既存の element-wise multiply Transform 経路を流用する。
// このフラグは line 16 の nnue_architecture.h include より前に見える必要があるため
// コンパイルフラグ (-DNNUE_FT_PAIRWISE) で渡す (arch ヘッダ内 #define では間に合わない)。
#if defined(NNUE_FT_PAIRWISE)
#define USE_ELEMENT_WISE_MULTIPLY
#endif

// NNUE_FT_SCRELU (exp013 arm1) は classic 経路の AVX2 / scalar のみ実装。
#if defined(NNUE_FT_SCRELU)
#if defined(USE_ELEMENT_WISE_MULTIPLY) || defined(USE_AVX512) || defined(USE_MMX) \
	|| defined(USE_NEON) || (defined(USE_SSE2) && !defined(USE_AVX2))
#error "NNUE_FT_SCRELU is implemented only for the AVX2 and scalar Transform paths"
#endif
#endif

// NNUE_FT_PAIRWISE (exp013 arm3) は element-wise 経路の AVX2 / scalar のみ検証済み。
#if defined(NNUE_FT_PAIRWISE)
#if defined(USE_AVX512) || defined(USE_MMX) || defined(USE_NEON) \
	|| (defined(USE_SSE2) && !defined(USE_AVX2))
#error "NNUE_FT_PAIRWISE is verified only for the AVX2 and scalar element-wise paths"
#endif
#endif

#include "nnue_common.h"
#include "nnue_architecture.h"
#include "features/index_list.h"

#include <algorithm>  // std::clamp
#include <cstring>  // std::memset()

namespace YaneuraOu {
namespace Eval::NNUE {

#if defined(ENABLE_FT_TRAFFIC_STAT)
// ============================================================
//   FT のメモリトラフィック内訳を数える (task#45 / report/49)
// ============================================================
// ★動機: 我々の NPS は FT の行 gather でメモリ律速 (report/23)。
//   その行読みが「差分 (2〜4 行)」と「全再構築 (~38 行)」のどちらに
//   使われているかで、打つべき手が変わる:
//     - 全再構築が支配的 → Finny table (王位置ごとの accumulator キャッシュ) が効く
//                          = V970 の +15% NPS の正体という仮説
//     - 差分が支配的     → 幅を削る / 疎化するしか無い
//
// 全再構築が起きる経路は 2 つある:
//   (a) refresh_accumulator … 差分連鎖が切れた (親の accumulator が未計算)
//   (b) update_accumulator 内の reset … **王が動いた**。HalfKP は王相対なので
//       その手番側の全特徴が張り替わる。removed は空で added が全 active になる
//
// ★計測は「呼び出し回数」でなく **行数** で数える。コストは行数に比例するため
//   (report/33 の教訓: コストは MAC でなくメモリで数えろ)。
struct FtStat {
    uint64_t n_transform = 0;   // Transform 呼び出し (= evaluate 相当)
    uint64_t n_refresh   = 0;   // (a) 連鎖断絶による全再構築
    uint64_t n_update    = 0;   // 差分更新の呼び出し
    uint64_t n_reset     = 0;   // (b) 王移動で reset された perspective 数
    uint64_t rows_full   = 0;   // (a)+(b) で読んだ行数
    uint64_t rows_inc    = 0;   // 差分で読んだ行数
};
// C++17 の inline 変数。計測用ビルドのみなので TU をまたぐ定義の手間を省く。
// Threads=1 前提 (計測用ビルドのみ。並列では数え落とす)。
inline FtStat g_ft_stat;
#endif

// If vector instructions are enabled, we update and refresh the
// accumulator tile by tile such that each tile fits in the CPU's
// vector registers.
// ベクトル命令が有効な場合、変数のタイルを、
// 各タイルがCPUのベクトルレジスタに収まるように、更新してリフレッシュする。
#define VECTOR

#if defined(USE_AVX512)
using vec_t = __m512i;
#define vec_load(a) _mm512_load_si512(a)
#define vec_store(a, b) _mm512_store_si512(a, b)
#define vec_add_16(a, b) _mm512_add_epi16(a, b)
#define vec_sub_16(a, b) _mm512_sub_epi16(a, b)
#define vec_mulhi_16(a, b) _mm512_mulhi_epi16(a, b)
#define vec_set_16(a) _mm512_set1_epi16(a)
#define vec_max_16(a, b) _mm512_max_epi16(a, b)
#define vec_min_16(a, b) _mm512_min_epi16(a, b)
#define vec_slli_16(a, b) _mm512_slli_epi16(a, b)
#define vec_packus_16(a, b) _mm512_packus_epi16(a, b)
#define vec_zero() _mm512_setzero_si512()
static constexpr IndexType kNumRegs = 8;  // only 8 are needed

#elif defined(USE_AVX2)
using vec_t = __m256i;
#define vec_load(a) _mm256_load_si256(a)
#define vec_store(a, b) _mm256_store_si256(a, b)
#define vec_add_16(a, b) _mm256_add_epi16(a, b)
#define vec_sub_16(a, b) _mm256_sub_epi16(a, b)
#define vec_mulhi_16(a, b) _mm256_mulhi_epi16(a, b)
#define vec_set_16(a) _mm256_set1_epi16(a)
#define vec_max_16(a, b) _mm256_max_epi16(a, b)
#define vec_min_16(a, b) _mm256_min_epi16(a, b)
#define vec_slli_16(a, b) _mm256_slli_epi16(a, b)
#define vec_packus_16(a, b) _mm256_packus_epi16(a, b)
#define vec_zero() _mm256_setzero_si256()
static constexpr IndexType kNumRegs = 16;

#elif defined(USE_SSE2)
using vec_t = __m128i;
#define vec_load(a) (*(a))
#define vec_store(a, b) *(a) = (b)
#define vec_add_16(a, b) _mm_add_epi16(a, b)
#define vec_sub_16(a, b) _mm_sub_epi16(a, b)
#define vec_mulhi_16(a, b) _mm_mulhi_epi16(a, b)
#define vec_set_16(a) _mm_set1_epi16(a)
#define vec_max_16(a, b) _mm_max_epi16(a, b)
#define vec_min_16(a, b) _mm_min_epi16(a, b)
#define vec_slli_16(a, b) _mm_slli_epi16(a, b)
#define vec_packus_16(a, b) _mm_packus_epi16(a, b)
#define vec_zero() _mm_setzero_si128()
static constexpr IndexType kNumRegs = Is64Bit ? 16 : 8;

#elif defined(USE_MMX)
using vec_t = __m64;
#define vec_load(a) (*(a))
#define vec_store(a, b) *(a) = (b)
#define vec_add_16(a, b) _mm_add_pi16(a, b)
#define vec_sub_16(a, b) _mm_sub_pi16(a, b)
#define vec_zero() _mm_setzero_si64()
static constexpr IndexType kNumRegs = 8;

#elif defined(USE_NEON)
using vec_t = int16x8_t;
#define vec_load(a) (*(a))
#define vec_store(a, b) *(a) = (b)
#define vec_add_16(a, b) vaddq_s16(a, b)
#define vec_sub_16(a, b) vsubq_s16(a, b)
#define vec_mulhi_16(a, b) vqdmulhq_s16(a, b)
#define vec_set_16(a) vdupq_n_s16(a)
#define vec_max_16(a, b) vmaxq_s16(a, b)
#define vec_min_16(a, b) vminq_s16(a, b)
#define vec_slli_16(a, b) vshlq_s16(a, vec_set_16(b))
#define vec_packus_16(a, b) reinterpret_cast<vec_t>(vcombine_u8(vqmovun_s16(a), vqmovun_s16(b)))
#define vec_zero() \
	vec_t { 0 }
static constexpr IndexType kNumRegs = 16;

#else
#undef VECTOR

#endif

/*
 例) SFNNwop-1536のときのkNumChunksの計算

┌─────────┬───────────────┬─────────────────┬────────────┐
│  SIMD            │ sizeof(vec_t)                │ / sizeof(int16)                  │ kNumChunks             │
├─────────┼───────────────┼─────────────────┼────────────┤
│ AVX-512          │ 64                           │ 32                               │ 1536/32=48             │
├─────────┼───────────────┼─────────────────┼────────────┤
│ AVX2             │ 32                           │ 16                               │ 1536/16=96             │
├─────────┼───────────────┼─────────────────┼────────────┤
│ SSE2             │ 16                           │ 8                                │ 1536/8=192             │
├─────────┼───────────────┼─────────────────┼────────────┤
│ NEON             │ 16                           │ 8                                │ 1536/8=192             │
└─────────┴───────────────┴─────────────────┴────────────┘
*/

constexpr IndexType MaxChunkSize = 16;

// Input feature converter
// 入力特徴量変換器
class FeatureTransformer {
   private:
	// Number of output dimensions for one side
	// 片側分の出力の次元数
	static constexpr IndexType kHalfDimensions = kTransformedFeatureDimensions;

#if defined(VECTOR)
	//static constexpr IndexType kTileHeight = kNumRegs * sizeof(vec_t) / 2;
	//static_assert(kHalfDimensions % kTileHeight == 0, "kTileHeight must divide kHalfDimensions");
	// ⇨  AVX-512でこの制約守れないっぽ。
#endif

   public:
	// Output type
	// 出力の型
	using OutputType = TransformedFeatureType;
	using BiasType   = std::int16_t;
	using WeightType = std::int16_t;

	// Number of input/output dimensions
	// 入出力の次元数
	static constexpr IndexType kInputDimensions  = RawFeatures::kDimensions;
#if defined(USE_ELEMENT_WISE_MULTIPLY)
	static constexpr IndexType kOutputDimensions = kHalfDimensions;
#else
	static constexpr IndexType kOutputDimensions = kHalfDimensions * 2;
#endif

	// Size of forward propagation buffer
	// 順伝播用バッファのサイズ
	static constexpr std::size_t kBufferSize = kOutputDimensions * sizeof(OutputType);

#if defined(NNUE_FT_SCRELU)
	// SCReLU ネット識別マーカー。exp013 trainer の SCRELU_HASH_MARKER /
	// experiments/013-arch-ladder/check_headers.py の MARKER と一致必須。
	static constexpr std::uint32_t kFtSCReLUHashMarker = 0x5C12E1D;
#endif
#if defined(NNUE_FT_PAIRWISE)
	// pairwise ネット識別マーカー。exp013 trainer の PAIRWISE_HASH_MARKER /
	// experiments/013-arch-ladder の nnue_eval.py PAIRWISE_HASH_MARKER と一致必須。
	static constexpr std::uint32_t kFtPairwiseHashMarker = 0x9A1E70;
#endif

	// Hash value embedded in the evaluation file
	// 評価関数ファイルに埋め込むハッシュ値
	static constexpr std::uint32_t GetHashValue() {
#if defined(SFNNwoPSQT)
		// 学習部と整合性とるの面倒なのでSFNNwoPSQTのときはこれに固定しておく。
		return 0x5f134ab8u;
#elif defined(NNUE_FT_SCRELU)
		// CReLU ネットとの取り違えをロード時に hash mismatch で検出する。
		return (RawFeatures::kHashValue ^ kOutputDimensions) ^ kFtSCReLUHashMarker;
#elif defined(NNUE_FT_PAIRWISE)
		// CReLU ネットとの取り違えをロード時に hash mismatch で検出する。
		// kOutputDimensions は USE_ELEMENT_WISE_MULTIPLY 下で kHalfDimensions (256)。
		return (RawFeatures::kHashValue ^ kOutputDimensions) ^ kFtPairwiseHashMarker;
#else
		return RawFeatures::kHashValue ^ kOutputDimensions;
#endif
	}

	// A string that represents the structure
	// 構造を表す文字列
	static std::string GetStructureString() {
		return RawFeatures::GetName() + "[" + std::to_string(kInputDimensions) + "->"
		       + std::to_string(kHalfDimensions) + "x2]";
	}

	// Read network parameters
	// パラメータを読み込む
	Tools::Result ReadParameters(std::istream& stream) {
#if defined(NNUE_FT_PAIRWISE)
		// exp013 arm3 pairwise: trainer は SavedFormat::quantise::<i16> = raw little-endian を出力。
		// SFNNwoPSQT の LEB128 形式とは異なるため、pairwise は標準 CReLU と同じ raw 読み出し。
		// Transform 経路 (USE_ELEMENT_WISE_MULTIPLY) の AVX2 layout 整合のため permute は必要。
		// ただし scale_weights(true) (FT 重み×2) は不要: SFNNwoPSQT は FT を half-scale で
		// export するため×2 で復元するが、exp013 trainer は qa=255 full-scale で export する。
		// ここで×2 すると acc が 2倍になり、(a*c)>>9 が ~4倍 + clamp254 飽和でネットが壊れる。
		// nnue_eval.py --verify (意図クオンタイズ: raw weight, clamp254, >>9) で 1:1 一致を確認済み。
		for (std::size_t i = 0; i < kHalfDimensions; ++i) biases_[i] = read_little_endian<BiasType>(stream);
		for (std::size_t i = 0; i < kHalfDimensions * kInputDimensions; ++i)
			weights_[i] = read_little_endian<WeightType>(stream);
#if defined(VECTOR)
		permute_weights(inverse_order_packs);
#endif
#elif defined(USE_ELEMENT_WISE_MULTIPLY)
		read_leb_128<BiasType>(stream, biases_, kHalfDimensions);
		read_leb_128<WeightType>(stream, weights_, kHalfDimensions * kInputDimensions);

#if defined(VECTOR)
		permute_weights(inverse_order_packs);
#endif
		scale_weights(true);
#else
		for (std::size_t i = 0; i < kHalfDimensions; ++i) biases_[i] = read_little_endian<BiasType>(stream);
		for (std::size_t i = 0; i < kHalfDimensions * kInputDimensions; ++i)
			weights_[i] = read_little_endian<WeightType>(stream);
#endif
#if defined(THREAT_ATTACKER_MAJOR)
		// threat 行を attacker-major へロード時置換 (task#59 / report/51 §7.6.2)。
		// RawFeatures = FeatureSet<Threat, HalfKP> なので threat の先頭行 = HalfKP::kDimensions。
		if (!stream.fail())
			Features::Threat::PermuteRows(weights_, kHalfDimensions,
			                              Features::HalfKP<Features::Side::kFriend>::kDimensions);
#endif
		return !stream.fail() ? Tools::ResultCode::Ok : Tools::ResultCode::FileReadError;
	}

	// Write network parameters
	// パラメータを書き込む
	bool WriteParameters(std::ostream& stream) const {
#if defined(THREAT_ATTACKER_MAJOR)
		// 並び替え済み配置の保存は未対応 (標準 pair-major へ逆置換していないため、
		// このまま書くと他ビルドで読めない nn.bin ができる)。学習系はこのビルドで使わないこと。
		return false;
#else
		stream.write(reinterpret_cast<const char*>(biases_), kHalfDimensions * sizeof(BiasType));
		stream.write(reinterpret_cast<const char*>(weights_), kHalfDimensions * kInputDimensions * sizeof(WeightType));
		return !stream.fail();
#endif
	}

	// Proceed with the difference calculation if possible
	// 可能なら差分計算を進める
	bool UpdateAccumulatorIfPossible(const Position& pos) const {
		const auto now = pos.state();
		if (now->accumulator.computed_accumulation) {
			return true;
		}
		const auto prev = now->previous;
		if (prev && prev->accumulator.computed_accumulation) {
			update_accumulator(pos);
			return true;
		}
		return false;
	}

	// Convert input features
	// 入力特徴量を変換する
	void Transform(const Position& pos, OutputType* output, bool refresh) const {
#if defined(ENABLE_FT_TRAFFIC_STAT)
		g_ft_stat.n_transform++;
#endif
		if (refresh || !UpdateAccumulatorIfPossible(pos)) {
			refresh_accumulator(pos);
		}
		const auto& accumulation = pos.state()->accumulator.accumulation;

#if defined(USE_ELEMENT_WISE_MULTIPLY)

#if defined(VECTOR)
			// Packed output is sizeof(vec_t) bytes for each SIMD register
#if defined(USE_AVX512)
			constexpr IndexType OutputChunkSize = 64;
#else
			constexpr IndexType OutputChunkSize = kSimdWidth;
#endif
		static_assert((kHalfDimensions / 2) % OutputChunkSize == 0);
		constexpr IndexType NumOutputChunks = kHalfDimensions / 2 / OutputChunkSize;

		vec_t Zero = vec_zero();
		vec_t One = vec_set_16(127 * 2);

		const Color perspectives[2] = { pos.side_to_move(), ~pos.side_to_move() };
		for (IndexType p = 0; p < 2; ++p) {
			const IndexType offset = (kHalfDimensions / 2) * p;

			// ★複数 refresh trigger (halfka2t 等) では平面を全て合算する。
			//   平面 0 固定読みだと threat 等の追加平面が出力に乗らない (task#54 で実害)。
			const vec_t* in0 = reinterpret_cast<const vec_t*>(&(accumulation[perspectives[p]][0][0]));
			const vec_t* in1 = reinterpret_cast<const vec_t*>(&(accumulation[perspectives[p]][0][kHalfDimensions / 2]));
			vec_t* out = reinterpret_cast<vec_t*>(output + offset);
			auto load0 = [&](IndexType idx) {
				vec_t v = in0[idx];
				for (IndexType t = 1; t < kRefreshTriggers.size(); ++t)
					v = vec_add_16(v, reinterpret_cast<const vec_t*>(
						&(accumulation[perspectives[p]][t][0]))[idx]);
				return v;
			};
			auto load1 = [&](IndexType idx) {
				vec_t v = in1[idx];
				for (IndexType t = 1; t < kRefreshTriggers.size(); ++t)
					v = vec_add_16(v, reinterpret_cast<const vec_t*>(
						&(accumulation[perspectives[p]][t][kHalfDimensions / 2]))[idx]);
				return v;
			};

			constexpr int shift =
#if defined(USE_SSE2)
				7;
#else
				6;
#endif

			for (IndexType j = 0; j < NumOutputChunks; ++j)
			{
				const vec_t sum0a =
					vec_slli_16(vec_max_16(vec_min_16(load0(j * 2 + 0), One), Zero), shift);
				const vec_t sum0b =
					vec_slli_16(vec_max_16(vec_min_16(load0(j * 2 + 1), One), Zero), shift);
				const vec_t sum1a = vec_min_16(load1(j * 2 + 0), One);
				const vec_t sum1b = vec_min_16(load1(j * 2 + 1), One);

				const vec_t pa = vec_mulhi_16(sum0a, sum1a);
				const vec_t pb = vec_mulhi_16(sum0b, sum1b);

				out[j] = vec_packus_16(pa, pb);
			}

		}

#else
		const Color perspectives[2] = { pos.side_to_move(), ~pos.side_to_move() };
		for (IndexType p = 0; p < 2; ++p) {
			const IndexType offset = (kHalfDimensions / 2) * p;

			for (IndexType j = 0; j < kHalfDimensions / 2; ++j)
			{
				BiasType sum0 = accumulation[perspectives[p]][0][j];
				BiasType sum1 = accumulation[perspectives[p]][0][j + kHalfDimensions / 2];
				for (IndexType t = 1; t < kRefreshTriggers.size(); ++t) {
					sum0 += accumulation[perspectives[p]][t][j];
					sum1 += accumulation[perspectives[p]][t][j + kHalfDimensions / 2];
				}
				sum0 = std::clamp<BiasType>(sum0, 0, 127 * 2);
				sum1 = std::clamp<BiasType>(sum1, 0, 127 * 2);
				output[offset + j] = static_cast<OutputType>(unsigned(sum0 * sum1) / 512);
			}

		}
#endif

#else

		// 以下は旧NNUEのコード。
		// ループ本体がx86とNEONで異なる（2入力→1出力 vs 1入力→1出力）ため、
		// kNumChunksの意味自体がアーキテクチャごとに違うため、共通化しにくい。触らないことにする。

#if defined(USE_AVX512)
		constexpr IndexType kNumChunks = kHalfDimensions / (kSimdWidth * 2);
		static_assert(kHalfDimensions % (kSimdWidth * 2) == 0);
		const __m512i kControl = _mm512_setr_epi64(0, 2, 4, 6, 1, 3, 5, 7);
		const __m512i kZero    = _mm512_setzero_si512();

#elif defined(USE_AVX2)
		constexpr IndexType kNumChunks = kHalfDimensions / kSimdWidth;
		constexpr int       kControl   = 0b11011000;
		const __m256i       kZero      = _mm256_setzero_si256();

#elif defined(USE_SSE2)
		constexpr IndexType kNumChunks = kHalfDimensions / kSimdWidth;
#if defined(USE_SSE41)
		const __m128i kZero = _mm_setzero_si128();
#else  // SSE41非対応だがSSE2は使える環境
		const __m128i k0x80s = _mm_set1_epi8(-128);
#endif

#elif defined(USE_MMX)
		// USE_MMX を config.h では現状、有効化することがないので dead code
		constexpr IndexType kNumChunks = kHalfDimensions / kSimdWidth;
		const __m64         k0x80s     = _mm_set1_pi8(-128);

#elif defined(USE_NEON)
		constexpr IndexType kNumChunks = kHalfDimensions / (kSimdWidth / 2);
		const int8x8_t      kZero      = {0};
#endif
		const Color perspectives[2] = {pos.side_to_move(), ~pos.side_to_move()};
		for (IndexType p = 0; p < 2; ++p) {
			const IndexType offset = kHalfDimensions * p;
#if defined(USE_AVX512)
			auto out = reinterpret_cast<__m512i*>(&output[offset]);
			for (IndexType j = 0; j < kNumChunks; ++j) {
				__m512i sum0 =
				    _mm512_load_si512(&reinterpret_cast<const __m512i*>(accumulation[perspectives[p]][0])[j * 2 + 0]);
				__m512i sum1 =
				    _mm512_load_si512(&reinterpret_cast<const __m512i*>(accumulation[perspectives[p]][0])[j * 2 + 1]);
				for (IndexType i = 1; i < kRefreshTriggers.size(); ++i) {
					sum0 = _mm512_add_epi16(
					    sum0,
					    reinterpret_cast<const __m512i*>(accumulation[perspectives[p]][i])[j * 2 + 0]);
					sum1 = _mm512_add_epi16(
					    sum1,
					    reinterpret_cast<const __m512i*>(accumulation[perspectives[p]][i])[j * 2 + 1]);
				}
				_mm512_store_si512(&out[j], _mm512_permutexvar_epi64(
								 kControl, _mm512_max_epi8(_mm512_packs_epi16(sum0, sum1), kZero)));
			}

#elif defined(USE_AVX2)
			auto out = reinterpret_cast<__m256i*>(&output[offset]);
			for (IndexType j = 0; j < kNumChunks; ++j) {
					__m256i sum0 =
					    _mm256_loadu_si256(&reinterpret_cast<const __m256i*>(accumulation[perspectives[p]][0])[j * 2 + 0]);
					__m256i sum1 =
					    _mm256_loadu_si256(&reinterpret_cast<const __m256i*>(accumulation[perspectives[p]][0])[j * 2 + 1]);
					for (IndexType i = 1; i < kRefreshTriggers.size(); ++i) {
						sum0 = _mm256_add_epi16(
							sum0,
							_mm256_loadu_si256(&reinterpret_cast<const __m256i*>(accumulation[perspectives[p]][i])[j * 2 + 0]));
						sum1 = _mm256_add_epi16(
							sum1,
							_mm256_loadu_si256(&reinterpret_cast<const __m256i*>(accumulation[perspectives[p]][i])[j * 2 + 1]));
					}
#if defined(NNUE_FT_SCRELU)
					// SCReLU: (clamp(acc,0,127))^2 >> 7。127*127=16129 < 32767 なので
					// mullo_epi16 で正確。結果 0..126 は packs の飽和に届かない。
					const __m256i kMax127 = _mm256_set1_epi16(127);
					sum0 = _mm256_min_epi16(_mm256_max_epi16(sum0, kZero), kMax127);
					sum1 = _mm256_min_epi16(_mm256_max_epi16(sum1, kZero), kMax127);
					sum0 = _mm256_srli_epi16(_mm256_mullo_epi16(sum0, sum0), 7);
					sum1 = _mm256_srli_epi16(_mm256_mullo_epi16(sum1, sum1), 7);
					_mm256_store_si256(&out[j], _mm256_permute4x64_epi64(
									 _mm256_packs_epi16(sum0, sum1), kControl));
#else
					_mm256_store_si256(&out[j], _mm256_permute4x64_epi64(
									 _mm256_max_epi8(_mm256_packs_epi16(sum0, sum1), kZero), kControl));
#endif
			}

#elif defined(USE_SSE2)
			auto out = reinterpret_cast<__m128i*>(&output[offset]);
			for (IndexType j = 0; j < kNumChunks; ++j) {
				__m128i sum0 =
				    _mm_load_si128(&reinterpret_cast<const __m128i*>(accumulation[perspectives[p]][0])[j * 2 + 0]);
				__m128i sum1 =
				    _mm_load_si128(&reinterpret_cast<const __m128i*>(accumulation[perspectives[p]][0])[j * 2 + 1]);
				for (IndexType i = 1; i < kRefreshTriggers.size(); ++i) {
					sum0 = _mm_add_epi16(sum0,
					                     reinterpret_cast<const __m128i*>(accumulation[perspectives[p]][i])[j * 2 + 0]);
					sum1 = _mm_add_epi16(sum1,
					                     reinterpret_cast<const __m128i*>(accumulation[perspectives[p]][i])[j * 2 + 1]);
				}

				const __m128i packedbytes = _mm_packs_epi16(sum0, sum1);
				_mm_store_si128(&out[j],
#if defined(USE_SSE41)
				                _mm_max_epi8(packedbytes, kZero)
#else  // SSE41非対応だがSSE2は使える環境
				                _mm_subs_epi8(_mm_adds_epi8(packedbytes, k0x80s), k0x80s)
#endif
				);
			}

#elif defined(USE_MMX)
			// USE_MMX を config.h では現状、有効化することがないので dead code
			auto out = reinterpret_cast<__m64*>(&output[offset]);
			for (IndexType j = 0; j < kNumChunks; ++j) {
				__m64       sum0 = *(&reinterpret_cast<const __m64*>(accumulation[perspectives[p]][0])[j * 2 + 0]);
				__m64       sum1 = *(&reinterpret_cast<const __m64*>(accumulation[perspectives[p]][0])[j * 2 + 1]);
				const __m64 packedbytes = _mm_packs_pi16(sum0, sum1);
				out[j]                  = _mm_subs_pi8(_mm_adds_pi8(packedbytes, k0x80s), k0x80s);
			}

#elif defined(USE_NEON)
			const auto out = reinterpret_cast<int8x8_t*>(&output[offset]);
			for (IndexType j = 0; j < kNumChunks; ++j) {
				int16x8_t sum = reinterpret_cast<const int16x8_t*>(accumulation[perspectives[p]][0])[j];
				for (IndexType i = 1; i < kRefreshTriggers.size(); ++i) {
					sum = vaddq_s16(sum, reinterpret_cast<const int16x8_t*>(accumulation[perspectives[p]][i])[j]);
				}
				out[j] = vmax_s8(vqmovn_s16(sum), kZero);
			}
#else
			for (IndexType j = 0; j < kHalfDimensions; ++j) {
				BiasType sum = accumulation[perspectives[p]][0][j];
				for (IndexType i = 1; i < kRefreshTriggers.size(); ++i) {
					sum += accumulation[perspectives[p]][i][j];
				}
#if defined(NNUE_FT_SCRELU)
				const int s = std::clamp<int>(sum, 0, 127);
				output[offset + j] = static_cast<OutputType>((s * s) >> 7);
#else
				output[offset + j] = static_cast<OutputType>(std::clamp<int>(sum, 0, 127));
#endif
			}
#endif
		}
#if defined(USE_MMX)
		// USE_MMX を config.h では現状、有効化することがないので dead code
		_mm_empty();
#endif
#endif
	}

   private:
	static void order_packs([[maybe_unused]] uint64_t* v) {
#if defined(USE_AVX512)  // _mm512_set_epi32 packs in the order [15 11 7 3 14 10 6 2 13 9 5 1 12 8 4 0]
		uint64_t tmp0 = v[4], tmp1 = v[5];
		v[4] = v[6], v[5] = v[7];
		v[6] = tmp0, v[7] = tmp1;
		tmp0 = v[8], tmp1 = v[9];
		v[8] = v[12], v[9] = v[13];
		v[12] = v[10], v[13] = v[11];
		v[10] = tmp0, v[11] = tmp1;
#elif defined(USE_AVX2)  // _mm256_set_epi32 packs in the order [7 3 6 2 5 1 4 0]
		uint64_t tmp0 = v[2], tmp1 = v[3];
		v[2] = v[4], v[3] = v[5];
		v[4] = tmp0, v[5] = tmp1;
#endif
	}

	static void inverse_order_packs([[maybe_unused]] uint64_t* v) {
#if defined(USE_AVX512)
		uint64_t tmp0 = v[2], tmp1 = v[3];
		v[2] = v[4], v[3] = v[5];
		v[4] = v[8], v[5] = v[9];
		v[8] = tmp0, v[9] = tmp1;
		tmp0 = v[6], tmp1 = v[7];
		v[6] = v[12], v[7] = v[13];
		v[12] = v[10], v[13] = v[11];
		v[10] = tmp0, v[11] = tmp1;
#elif defined(USE_AVX2)  // Inverse _mm256_packs_epi16 ordering
		uint64_t tmp0 = v[2], tmp1 = v[3];
		v[2] = v[4], v[3] = v[5];
		v[4] = tmp0, v[5] = tmp1;
#endif
	}

	void permute_weights([[maybe_unused]] void (*order_fn)(uint64_t*)) const {
#if defined(USE_AVX2)
#if defined(USE_AVX512)
		constexpr IndexType di = 16;
#else
		constexpr IndexType di = 8;
#endif
		uint64_t* b = reinterpret_cast<uint64_t*>(const_cast<BiasType*>(&biases_[0]));
		for (IndexType i = 0; i < kHalfDimensions * sizeof(BiasType) / sizeof(uint64_t); i += di)
			order_fn(&b[i]);

		for (IndexType j = 0; j < kInputDimensions; ++j)
		{
			uint64_t* w =
				reinterpret_cast<uint64_t*>(const_cast<WeightType*>(&weights_[j * kHalfDimensions]));
			for (IndexType i = 0; i < kHalfDimensions * sizeof(WeightType) / sizeof(uint64_t);
					i += di)
				order_fn(&w[i]);
		}
#endif
	}

	inline void scale_weights(bool read) const {
		for (IndexType j = 0; j < kInputDimensions; ++j)
		{
			WeightType* w = const_cast<WeightType*>(&weights_[j * kHalfDimensions]);
			for (IndexType i = 0; i < kHalfDimensions; ++i)
				w[i] = read ? w[i] * 2 : w[i] / 2;
		}

		BiasType* b = const_cast<BiasType*>(biases_);
		for (IndexType i = 0; i < kHalfDimensions; ++i)
			b[i] = read ? b[i] * 2 : b[i] / 2;
	}

	// Calculate cumulative value without using difference calculation
	// 差分計算を用いずに累積値を計算する
	void refresh_accumulator(const Position& pos) const {
		auto& accumulator = pos.state()->accumulator;
		for (IndexType i = 0; i < kRefreshTriggers.size(); ++i) {
			Features::IndexList active_indices[2];
			RawFeatures::AppendActiveIndices(pos, kRefreshTriggers[i], active_indices);
#if defined(ENABLE_FT_TRAFFIC_STAT)
			if (i == 0) g_ft_stat.n_refresh++;
			g_ft_stat.rows_full += active_indices[BLACK].size() + active_indices[WHITE].size();
#endif
			for (Color perspective : {BLACK, WHITE}) {
#if defined(VECTOR)
				if (i == 0) {
					std::memcpy(accumulator.accumulation[perspective][i], biases_, kHalfDimensions * sizeof(BiasType));
				} else {
					std::memset(accumulator.accumulation[perspective][i], 0, kHalfDimensions * sizeof(BiasType));
				}
				for (const auto index : active_indices[perspective]) {
					const IndexType offset = kHalfDimensions * index;
					auto accumulation      = reinterpret_cast<vec_t*>(&accumulator.accumulation[perspective][i][0]);
					auto column            = reinterpret_cast<const vec_t*>(&weights_[offset]);
					constexpr IndexType kNumChunks = kHalfDimensions / (sizeof(vec_t) / sizeof(BiasType));
					for (IndexType j = 0; j < kNumChunks; ++j) {
						accumulation[j] = vec_add_16(accumulation[j], column[j]);
					}
				}
#else
				if (i == 0) {
					std::memcpy(accumulator.accumulation[perspective][i], biases_, kHalfDimensions * sizeof(BiasType));
				} else {
					std::memset(accumulator.accumulation[perspective][i], 0, kHalfDimensions * sizeof(BiasType));
				}
				for (const auto index : active_indices[perspective]) {
					const IndexType offset = kHalfDimensions * index;

					for (IndexType j = 0; j < kHalfDimensions; ++j) {
						accumulator.accumulation[perspective][i][j] += weights_[offset + j];
					}
				}
#endif
			}
		}

		accumulator.computed_accumulation = true;
		// Stockfishでは fc27d15(2020-09-07) にcomputed_scoreが排除されているので確認
		accumulator.computed_score = false;
	}

	// Calculate cumulative value using difference calculation
	// 差分計算を用いて累積値を計算する
	void update_accumulator(const Position& pos) const {
		const auto prev_accumulator = pos.state()->previous->accumulator;
		auto&      accumulator      = pos.state()->accumulator;
		for (IndexType i = 0; i < kRefreshTriggers.size(); ++i) {
			Features::IndexList removed_indices[2], added_indices[2];
			bool                reset[2];
			RawFeatures::AppendChangedIndices(pos, kRefreshTriggers[i], removed_indices, added_indices, reset);
#if defined(ENABLE_FT_TRAFFIC_STAT)
			if (i == 0) g_ft_stat.n_update++;
			for (Color pc : {BLACK, WHITE}) {
				const uint64_t rows = removed_indices[pc].size() + added_indices[pc].size();
				// ★reset は王移動。added が全 active になるので「全再構築」に数える
				if (reset[pc]) { g_ft_stat.n_reset++; g_ft_stat.rows_full += rows; }
				else           { g_ft_stat.rows_inc += rows; }
			}
#endif
#if defined(FT_ROW_PREFETCH)
			// 差分行の先出しプリフェッチ (task#59): index は確定済みなので、accumulate に入る前に
			// 全行 (両視点) のフェッチを重ねて発行する。行 = kHalfDimensions×2B、先頭と中間の
			// 2 ライン → 残りは HW ストリームプリフェッチに任せる。意味論は不変。
			for (Color pf_p : {BLACK, WHITE}) {
				for (const auto index : removed_indices[pf_p]) {
					const auto* row = reinterpret_cast<const char*>(&weights_[kHalfDimensions * index]);
					_mm_prefetch(row, _MM_HINT_T0);
					_mm_prefetch(row + kHalfDimensions, _MM_HINT_T0);   // 行の中間 (bytes = dims*2/2)
				}
				for (const auto index : added_indices[pf_p]) {
					const auto* row = reinterpret_cast<const char*>(&weights_[kHalfDimensions * index]);
					_mm_prefetch(row, _MM_HINT_T0);
					_mm_prefetch(row + kHalfDimensions, _MM_HINT_T0);
				}
			}
#endif
			for (Color perspective : {BLACK, WHITE}) {
#if defined(VECTOR)
				constexpr IndexType kNumChunks = kHalfDimensions / (sizeof(vec_t) / sizeof(BiasType));
				auto accumulation              = reinterpret_cast<vec_t*>(&accumulator.accumulation[perspective][i][0]);
#endif
				if (reset[perspective]) {
					if (i == 0) {
						std::memcpy(accumulator.accumulation[perspective][i], biases_,
						            kHalfDimensions * sizeof(BiasType));
					} else {
						std::memset(accumulator.accumulation[perspective][i], 0, kHalfDimensions * sizeof(BiasType));
					}
				} else {
					// Difference calculation for the feature amount changed from 1 to 0
					// 1から0に変化した特徴量に関する差分計算
					std::memcpy(accumulator.accumulation[perspective][i], prev_accumulator.accumulation[perspective][i],
					            kHalfDimensions * sizeof(BiasType));
					for (const auto index : removed_indices[perspective]) {
						const IndexType offset = kHalfDimensions * index;
#if defined(VECTOR)
						auto column = reinterpret_cast<const vec_t*>(&weights_[offset]);
						for (IndexType j = 0; j < kNumChunks; ++j) {
							accumulation[j] = vec_sub_16(accumulation[j], column[j]);
						}
#else
						for (IndexType j = 0; j < kHalfDimensions; ++j) {
							accumulator.accumulation[perspective][i][j] -= weights_[offset + j];
						}
#endif
					}
				}
				{
					// Difference calculation for features that changed from 0 to 1
					// 0から1に変化した特徴量に関する差分計算
					for (const auto index : added_indices[perspective]) {
						const IndexType offset = kHalfDimensions * index;
#if defined(VECTOR)
						auto column = reinterpret_cast<const vec_t*>(&weights_[offset]);
						for (IndexType j = 0; j < kNumChunks; ++j) {
							accumulation[j] = vec_add_16(accumulation[j], column[j]);
						}
#else
						for (IndexType j = 0; j < kHalfDimensions; ++j) {
							accumulator.accumulation[perspective][i][j] += weights_[offset + j];
						}
#endif
					}
				}
			}
		}

		accumulator.computed_accumulation = true;
		// Stockfishでは fc27d15(2020-09-07) にcomputed_scoreが排除されているので確認
		accumulator.computed_score = false;
	}

	// parameter type
	// パラメータの型

	// Make the learning class a friend
	// 学習用クラスをfriendにする
	friend class Trainer<FeatureTransformer>;

	// parameter
	// パラメータ
	alignas(kCacheLineSize) BiasType biases_[kHalfDimensions];
	alignas(kCacheLineSize) WeightType weights_[kHalfDimensions * kInputDimensions];
};

} // namespace Eval::NNUE
} // namespace YaneuraOu

#endif  // defined(EVAL_NNUE)

#endif  // #ifndef NNUE_FEATURE_TRANSFORMER_H_INCLUDED
