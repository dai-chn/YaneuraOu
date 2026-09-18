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
#include <vector>   // NNUE_FT_INT8_ROWS のロード時一時領域
#include <algorithm>  // std::lower_bound (NNUE_FT_INT8_ROWS の残差表)
#include <cstdio>   // std::fprintf (narrow threat ロード検査のエラー出力)
#if defined(ENABLE_FT_TRAFFIC_STAT)
#include <x86intrin.h>  // __rdtsc (行の足し引きのサイクル計測、2026-09-18)
#endif

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
    uint64_t rows_cache  = 0;   // (c) accumulator キャッシュ (ENABLE_ACC_CACHE) との集合差分で読んだ行数
    uint64_t n_cache_hit = 0;   // (c) キャッシュが有効だった (全再構築を差分に置き換えた) 回数
    // 2026-09-18 (report/52 §20): 行の足し引きにかかる rdtsc サイクル (メモリ側の税の直接計測)。
    //   差分更新の 1 行ごとに計測し、index が RawFeatures::kTailDimensions 以上 (= Head 特徴、threat 系では threat 行) と
    //   それ未満 (Tail = KP/KA2 行) に分けて集計する。全再構築はループ全体で計測。
    uint64_t cyc_inc_head = 0, rows_inc_head = 0;   // 差分更新: Head (threat) 行
    uint64_t cyc_inc_tail = 0, rows_inc_tail = 0;   // 差分更新: Tail (KP/KA2) 行
    uint64_t cyc_full     = 0;                      // 全再構築 (refresh + reset) の行加算ループ全体
    uint64_t cyc_update   = 0;                      // update_accumulator 全体 (index 収集 + memcpy + 行加減算)
    uint64_t cyc_collect  = 0;                      // update_accumulator 内の AppendChangedIndices (index 収集 = 列挙 + 写像) だけ
    uint64_t cyc_refresh  = 0;                      // refresh_accumulator 全体
    // 2026-09-18 (report/52 §23.4): 評価 (ComputeScore = Transform + 密層) 全体と、走行の経過サイクル (最初の評価から最後の評価まで)。
    //   ノード数は呼び出し側 (go nodes N × 局面数) が知っているので、eval 外 (探索側) の 1 ノードあたりコストは
    //   (elapsed − cyc_eval) / nodes で出す。
    uint64_t cyc_eval     = 0, n_eval = 0;
    uint64_t t_first_eval = 0, t_last_eval = 0;
    // 注: 行ごとの rdtsc は非直列化命令なので OoO 実行でロード待ちが後続に付け替わり得る (下限値)。
    //     メモリ側の税は update_total − collect で読む (行の加減算 + memcpy + キャッシュ差分の合計)。
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

// ★レジスタタイル差分更新 (task#82、2026-09-18、report/52 §23): VECTOR ビルドの既定。-DNNUE_FT_NO_TILING で従来経路
//   (memcpy + 行ごとに accumulator 全幅を読み書き) に戻す。行ごと rdtsc 計測 (ENABLE_FT_TRAFFIC_STAT で FT_STAT_NO_ROW_TIMING
//   なし) は行単位の経路が要るので自動的に従来経路。
#if defined(VECTOR) && !defined(NNUE_FT_NO_TILING) && !(defined(ENABLE_FT_TRAFFIC_STAT) && !defined(FT_STAT_NO_ROW_TIMING))
#define NNUE_FT_TILED
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
#if defined(NNUE_FT_INT8_ROWS)
	// ★int8 行 (task#82 / report/52 §23.6): FT 重みを nn.bin の値 (w×127、|v| ≤ 127 に clip) のまま int8 で格納し、
	//   従来の「ロード時 ×2 (scale_weights)」を掛けない。accumulator は従来の 1/2 スケールになり、Transform で
	//   (a'<<8)·(c'<<1) >> 16 = a'c' >> 7 = (2a')(2c') >> 9 として従来と同じ出力を得る (|v|>127 で clip された重み以外はビット一致)。
	//   行の読みが半分 (2 KB → 1 KB) になり、L2 帯域律速の行ループ (§23.4: update の 8 割) が軽くなる。VECTOR (AVX2/AVX-512) +
	//   USE_ELEMENT_WISE_MULTIPLY 専用。int8 → int16 の拡張は行ロード時 (vpmovsxbw)。
	using RowType = std::int8_t;
	static constexpr int kFtHalfScale = 1;
#if !defined(USE_ELEMENT_WISE_MULTIPLY) || !defined(USE_AVX2)
#error "NNUE_FT_INT8_ROWS needs USE_ELEMENT_WISE_MULTIPLY (SFNN) and AVX2/AVX-512"
#endif
#else
	using RowType = WeightType;
	static constexpr int kFtHalfScale = 0;
#endif
#if defined(VECTOR)
	// 重み行の 1 ベクトル分 (int16 × kSimdWidth/2 個) をロードする。int8 行なら符号拡張しながら。
	static inline vec_t vec_load_row(const RowType* p) {
#if defined(NNUE_FT_INT8_ROWS)
#if defined(USE_AVX512)
		return _mm512_cvtepi8_epi16(_mm256_load_si256(reinterpret_cast<const __m256i*>(p)));
#else
		return _mm256_cvtepi8_epi16(_mm_load_si128(reinterpret_cast<const __m128i*>(p)));
#endif
#else
		return vec_load(reinterpret_cast<const vec_t*>(p));
#endif
	}
	static constexpr IndexType kVecElems = sizeof(vec_t) / sizeof(BiasType);
#endif

#if defined(NNUE_FT_INT8_ROWS)
	// ★残差表: |v| > 127 で clip した重みの差分 (v − clip(v)) を行ごとに疎に持ち、行の加減算の後に足す。
	//   これで int16 経路とビット一致になる (王者では 6,900 個 / 5,291 行、行ごと平均 1.3 個。列 398/194/171 等の少数のニューロンに
	//   集中しているので、clip したままだと多くの局面で評価が動く: 87 局面で一致 61%、|差| 最大 44cp)。
	//   行に残差があるかはビット表 (kInputDimensions bits ≈ 43 KB) で見て、あれば行 index の二分探索で範囲を引く。
	//   (FeatureTransformer は SystemWideSharedConstant に載る = 自明コピー可能でないといけないので、表は固定長配列で持つ)
	//   ★行に残差があるかの判定は「行の先頭バイトが −128 (clip 範囲 ±127 に無い番兵)」で行う。行データは行ループが読んだばかりで
	//   キャッシュにあるので判定はほぼ無料 (別のビット表を引く版は表がキャッシュから追い出されて 1 update 630 cycles = NPS −8% だった)。
	//   番兵にした先頭要素の本当の値は残差表の col 0 に (delta = v0 + 128) 入れる。
	struct FixEntry { std::uint16_t col; std::int16_t delta; };
	static constexpr std::size_t kMaxFixRows = std::size_t(1) << 16, kMaxFix = std::size_t(1) << 17;
	static constexpr RowType kFixSentinel = RowType(-128);
	std::uint32_t fix_idx_[RawFeatures::kDimensions]; // 行 index → 残差ブロック番号 k (残差の無い行は未使用)
	std::uint32_t fix_start_[kMaxFixRows + 1];       // ブロック k の残差 = fix_[fix_start_[k] .. fix_start_[k+1])
	FixEntry      fix_[kMaxFix];
	std::uint32_t n_fix_rows_ = 0, n_fix_ = 0;
	// 行内の格納位置 col → その行が足される accumulator 上の位置
	inline IndexType fix_acc_pos(IndexType index, IndexType col) const {
#if defined(NNUE_THREAT_NARROW_COLS)
		if (kNarrowBlocks > 1 && index >= kTailRows) {
			const IndexType b = narrow_block(index);
			return col < kNarrowCols ? b * kNarrowCols + col : kHalfDimensions / 2 + b * kNarrowCols + (col - kNarrowCols);
		}
#endif
		(void)index;
		return col;
	}
	// 残差のある行だけが来る遅い経路 (二分探索 + 疎な加減算)。行ループの codegen を汚さないよう noinline。
	template <bool kAdd>
	__attribute__((noinline)) void fix_row_slow(BiasType* acc, IndexType index) const {
		const std::size_t k = fix_idx_[index];
		for (std::uint32_t e = fix_start_[k]; e < fix_start_[k + 1]; ++e) {
			const IndexType pos = fix_acc_pos(index, fix_[e].col);
			acc[pos] = static_cast<BiasType>(kAdd ? acc[pos] + fix_[e].delta : acc[pos] - fix_[e].delta);
		}
	}
	template <bool kAdd>
	inline void fix_row(BiasType* acc, IndexType index) const {
#if defined(NNUE_FT_INT8_NO_RESIDUAL)
		// 計測用: 残差表を引かない (clip したままの評価 = NNUE_FT_CLIP127_TEST の int16 ビルドと探索一致、残差表のコストを切り分ける)
		(void)acc; (void)index; return;
#endif
		if (__builtin_expect(weights_[row_off(index)] == kFixSentinel, 0))
			fix_row_slow<kAdd>(acc, index);
	}
#else
	template <bool kAdd>
	inline void fix_row(BiasType*, IndexType) const {}
#endif

#if defined(ENABLE_ACC_CACHE)
	// ★accumulator キャッシュ ("Finny table"、task#50 / report/52 §18.13)
	//   玉移動トリガ (kFriendKingMoved) の slot は玉が動くたびに全再構築 (HalfKP で ~39 行/視点) になる。
	//   視点 × 自玉升ごとに「その時の accumulator と active index (ソート済)」を thread_local に覚えておき、
	//   次に同じ玉升で全再構築が要るときはキャッシュとの集合差分 (removed/added の数行) で作る。
	//   特徴集合の種類に依らず index 列だけで動くので HalfKP / HalfKA2 共通。整数加算の順序非依存性により
	//   結果はビット一致 (探索一致で確認)。ネットを読み直したら generation で無効化する。
	struct AccCacheEntry {
		alignas(kCacheLineSize) BiasType accumulation[kHalfDimensions];
		IndexType indices[RawFeatures::kMaxActiveDimensions];
		std::uint16_t n = 0;
		bool valid = false;
	};
	struct AccCache {
		AccCacheEntry e[COLOR_NB][SQ_NB];   // [視点][その視点の自玉升]
		std::uint32_t generation = 0;
		void clear() { for (auto& row : e) for (auto& ent : row) ent.valid = false; }
	};
	static AccCache& acc_cache() {
		static thread_local AccCache cache;
		return cache;
	}
#endif

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

#if defined(NNUE_THREAT_NARROW_COLS)
	// ★threat 専用スライス (2026-09-18、report/52 §21.1): Head 特徴 (threat、kRefreshTriggers[0] == kNone の slot) の行を
	//   pairwise 前半 [0,N) と後半 [kHalf/2, kHalf/2+N) の 2N 列だけ持つ (行 2N×2 B、行列 kHeadRows×2N×2 B)。
	//   列マスクで訓練した nn.bin (マスク外 = 0) 専用で、ロード時にマスク外が 0 であることを検査して違反なら読込失敗にする。
	//   accumulator の narrow slot は先頭 2N 要素だけ使い、Transform では full slot の先頭 N 列 (各半分) にだけ足す。
	//   バイアスは full slot (KA2) に置く。評価は全幅ビルドとビット一致 (マスク外が 0 なので和が同じ)。
	static constexpr IndexType kNarrowCols  = NNUE_THREAT_NARROW_COLS;
	static constexpr IndexType kNarrowWidth = 2 * kNarrowCols;
	static constexpr IndexType kNarrowSlot  = 0;
	static constexpr IndexType kBiasSlot    = 1;
	static constexpr IndexType kTailRows    = RawFeatures::kTailDimensions;
	static constexpr IndexType kHeadRows    = kInputDimensions - kTailRows;
	static constexpr std::size_t kWeightsCount =
		std::size_t(kHalfDimensions) * kTailRows + std::size_t(kNarrowWidth) * kHeadRows;
	static_assert(kRefreshTriggers.size() == 2 && kRefreshTriggers[0] == Features::TriggerEvent::kNone,
	              "NNUE_THREAT_NARROW_COLS needs FeatureSet<Threat (kNone), Base (kFriendKingMoved)>");
	static_assert(kNarrowCols % 32 == 0 && kNarrowCols * 2 <= kHalfDimensions, "narrow cols must be a multiple of 32");
	static_assert(kHeadRows > 0, "no Head feature");
	// ブロック疎 (report/52 §21.5、NNUE_THREAT_NARROW_BLOCKS=B > 1): threat 行 t はブロック b = t % B を持ち、その 2N 値を
	// full 幅 accumulator の列 [b*N, b*N+N) と [kHalf/2 + b*N, ...) に足す。行の格納は B=1 と同じ 2N 幅。narrow slot は full 幅になり、
	// Transform は通常の全 slot 合算 (B=1 のときだけ先頭 N 列の短縮経路)。
#if !defined(NNUE_THREAT_NARROW_BLOCKS)
#define NNUE_THREAT_NARROW_BLOCKS 1
#endif
	static constexpr IndexType kNarrowBlocks = NNUE_THREAT_NARROW_BLOCKS;
	static_assert(kNarrowBlocks >= 1 && kNarrowCols * kNarrowBlocks * 2 <= kHalfDimensions, "narrow blocks x cols must fit in kHalf/2");
	static constexpr bool narrow_blocked() { return kNarrowBlocks > 1; }
	static constexpr IndexType narrow_block(IndexType index) { return kNarrowBlocks > 1 ? (index - kTailRows) % kNarrowBlocks : 0; }
	// accumulator 上の slot 幅 (B>1 の narrow slot は full 幅) と、格納行の幅 (narrow 行は常に 2N)
	static constexpr IndexType slot_width(IndexType slot) { return (slot == kNarrowSlot && kNarrowBlocks == 1) ? kNarrowWidth : kHalfDimensions; }
	static constexpr IndexType row_width(IndexType slot) { return slot == kNarrowSlot ? kNarrowWidth : kHalfDimensions; }
	// B>1 用: narrow 行 (2N 値) を full 幅 accumulator のブロック位置に足す (kAdd) / 引く
	template <bool kAdd>
	inline void apply_narrow_row(BiasType* acc, IndexType index) const {
		const RowType* row = &weights_[row_off(index)];
		const IndexType b = narrow_block(index);
		BiasType* a0 = acc + b * kNarrowCols;
		BiasType* a1 = acc + kHalfDimensions / 2 + b * kNarrowCols;
#if defined(VECTOR)
		constexpr IndexType kChunks = kNarrowCols / kVecElems;
		auto v0 = reinterpret_cast<vec_t*>(a0); auto v1 = reinterpret_cast<vec_t*>(a1);
		for (IndexType j = 0; j < kChunks; ++j) {
			const vec_t c0 = vec_load_row(row + j * kVecElems), c1 = vec_load_row(row + kNarrowCols + j * kVecElems);
			v0[j] = kAdd ? vec_add_16(v0[j], c0) : vec_sub_16(v0[j], c0);
			v1[j] = kAdd ? vec_add_16(v1[j], c1) : vec_sub_16(v1[j], c1);
		}
#else
		for (IndexType j = 0; j < kNarrowCols; ++j) {
			if (kAdd) { a0[j] += row[j]; a1[j] += row[kNarrowCols + j]; }
			else      { a0[j] -= row[j]; a1[j] -= row[kNarrowCols + j]; }
		}
#endif
		fix_row<kAdd>(acc, index);
	}
#else
	static constexpr IndexType kBiasSlot = 0;
	static constexpr std::size_t kWeightsCount = std::size_t(kHalfDimensions) * kInputDimensions;
	static constexpr IndexType slot_width(IndexType /*slot*/) { return kHalfDimensions; }
	static constexpr IndexType row_width(IndexType /*slot*/) { return kHalfDimensions; }
#endif

	// タイル更新 (NNUE_FT_TILED) の対象 slot か: B>1 の narrow slot は行がブロック位置に散るので従来経路 (apply_narrow_row)
#if defined(NNUE_THREAT_NARROW_COLS)
	static constexpr bool tiled_slot(IndexType slot) { return !(narrow_blocked() && slot == kNarrowSlot); }
#else
	static constexpr bool tiled_slot(IndexType /*slot*/) { return true; }
#endif
#if defined(NNUE_FT_TILED)
	// ★レジスタタイル差分更新 (task#82、report/52 §23): accumulator を kNumRegs × vec_t のタイル (AVX2: 16 × 16 = 256 値 = 512 B)
	//   に分け、タイルをレジスタに載せたまま removed/added の全行のそのタイル部分を加減算してから 1 回だけ書き戻す。
	//   従来 (memcpy + 行ごとに accumulator 全幅を読み書き) は行 1 本あたり「acc 読み + acc 書き + 行読み」= 3 × 幅 の
	//   トラフィックだったのが「行読み」= 1 × 幅 になる。src = 出発点 (前局面の accumulator / バイアス / nullptr = 0)。
	//   整数の加減算は順序に依らないので結果はビット一致 (探索一致で確認)。
	inline void apply_rows_tiled(BiasType* acc, const BiasType* src, IndexType width,
	                             const IndexType* rem, std::size_t n_rem,
	                             const IndexType* add, std::size_t n_add) const {
		constexpr IndexType kChunk = sizeof(vec_t) / sizeof(BiasType);
		constexpr IndexType kTile  = kNumRegs * kChunk;
#if defined(NNUE_FT_INT8_ROWS) && !defined(NNUE_FT_INT8_NO_RESIDUAL)
		// 残差のある行 (先頭バイトが番兵) をタイルループの前に拾う。この読みは行の先頭ラインの prefetch を兼ねる。
		IndexType frem[RawFeatures::kMaxActiveDimensions], fadd[RawFeatures::kMaxActiveDimensions];
		std::size_t nfr = 0, nfa = 0;
		for (std::size_t r = 0; r < n_rem; ++r) if (__builtin_expect(weights_[row_off(rem[r])] == kFixSentinel, 0)) frem[nfr++] = rem[r];
		for (std::size_t r = 0; r < n_add; ++r) if (__builtin_expect(weights_[row_off(add[r])] == kFixSentinel, 0)) fadd[nfa++] = add[r];
#endif
		IndexType j = 0;
		for (; j + kTile <= width; j += kTile) {
			vec_t regs[kNumRegs];
			if (src) {
				const vec_t* s = reinterpret_cast<const vec_t*>(src + j);
				for (IndexType k = 0; k < kNumRegs; ++k) regs[k] = vec_load(&s[k]);
			} else {
				for (IndexType k = 0; k < kNumRegs; ++k) regs[k] = vec_zero();
			}
			for (std::size_t r = 0; r < n_rem; ++r) {
				const RowType* col = &weights_[row_off(rem[r]) + j];
				for (IndexType k = 0; k < kNumRegs; ++k) regs[k] = vec_sub_16(regs[k], vec_load_row(col + k * kChunk));
			}
			for (std::size_t r = 0; r < n_add; ++r) {
				const RowType* col = &weights_[row_off(add[r]) + j];
				for (IndexType k = 0; k < kNumRegs; ++k) regs[k] = vec_add_16(regs[k], vec_load_row(col + k * kChunk));
			}
			vec_t* d = reinterpret_cast<vec_t*>(acc + j);
			for (IndexType k = 0; k < kNumRegs; ++k) vec_store(&d[k], regs[k]);
		}
		// 端数 (幅がタイルの倍数でないとき): チャンク単位
		for (; j < width; j += kChunk) {
			vec_t v = src ? vec_load(reinterpret_cast<const vec_t*>(src + j)) : vec_zero();
			for (std::size_t r = 0; r < n_rem; ++r)
				v = vec_sub_16(v, vec_load_row(&weights_[row_off(rem[r]) + j]));
			for (std::size_t r = 0; r < n_add; ++r)
				v = vec_add_16(v, vec_load_row(&weights_[row_off(add[r]) + j]));
			vec_store(reinterpret_cast<vec_t*>(acc + j), v);
		}
#if defined(NNUE_FT_INT8_ROWS) && !defined(NNUE_FT_INT8_NO_RESIDUAL)
		for (std::size_t k = 0; k < nfr; ++k) fix_row_slow<false>(acc, frem[k]);
		for (std::size_t k = 0; k < nfa; ++k) fix_row_slow<true>(acc, fadd[k]);
#endif
	}
#endif

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
#if defined(ENABLE_ACC_CACHE)
		++load_generation_;   // 重みが変わるので各スレッドの accumulator キャッシュを無効化する
#endif
#if defined(NNUE_FT_INT8_ROWS)
		// int16 で読んで並び替えてから int8 へ落とす (一時領域 kWeightsCount × 2 B)
		std::vector<WeightType> wtmp(kWeightsCount);
		WeightType* const wdst = wtmp.data();
#else
		[[maybe_unused]] WeightType* const wdst = weights_;
#endif
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
			wdst[i] = read_little_endian<WeightType>(stream);
#if defined(VECTOR)
		permute_weights(inverse_order_packs, wdst);
#endif
#elif defined(USE_ELEMENT_WISE_MULTIPLY)
		read_leb_128<BiasType>(stream, biases_, kHalfDimensions);
#if defined(NNUE_THREAT_NARROW_COLS)
		{
			// 全幅 (kHalf 列) で書かれた LEB128 ブロックをストリーム復号し、Tail 行はそのまま、Head (threat) 行はスライス列だけ
			// 圧縮格納する。スライス外に非零があれば列マスク訓練のネットでないので読込失敗にする (静かな評価破壊を防ぐ)。
			const std::size_t tail_count = std::size_t(kHalfDimensions) * kTailRows;
			std::size_t violations = 0;
			read_leb_128_sink<WeightType>(stream, std::size_t(kHalfDimensions) * kInputDimensions,
				[&](std::size_t i, WeightType v) {
					if (i < tail_count) { wdst[i] = v; return; }
					const std::size_t k = i - tail_count;
					const std::size_t r = k / kHalfDimensions, c = k % kHalfDimensions;
					WeightType* row = &wdst[tail_count + r * kNarrowWidth];
					const std::size_t lo0 = (kNarrowBlocks > 1 ? (r % kNarrowBlocks) : 0) * kNarrowCols;
					const std::size_t lo1 = kHalfDimensions / 2 + lo0;
					if (c >= lo0 && c < lo0 + kNarrowCols)
						row[c - lo0] = v;
					else if (c >= lo1 && c < lo1 + kNarrowCols)
						row[kNarrowCols + (c - lo1)] = v;
					else if (v != 0)
						++violations;
				});
			if (violations != 0) {
				std::fprintf(stderr, "Error! narrow threat build (N=%u, blocks=%u): %llu nonzero weights outside the slice — not a slice-trained net\n",
				             unsigned(kNarrowCols), unsigned(kNarrowBlocks), (unsigned long long)violations);
				return Tools::ResultCode::FileReadError;
			}
		}
#else
		read_leb_128<WeightType>(stream, wdst, kHalfDimensions * kInputDimensions);
#endif

#if defined(VECTOR)
		permute_weights(inverse_order_packs, wdst);
#endif
#if defined(NNUE_FT_CLIP127_TEST)
		// 検証用: int16 経路のまま |v| > 127 を clip して、NNUE_FT_INT8_ROWS ビルドと探索一致するかを見る (clip と算術の切り分け)
		for (std::size_t i = 0; i < kWeightsCount; ++i)
			wdst[i] = static_cast<WeightType>(wdst[i] > 127 ? 127 : (wdst[i] < -127 ? -127 : wdst[i]));
#endif
#if !defined(NNUE_FT_INT8_ROWS)
		scale_weights(true);
#endif
#else
		for (std::size_t i = 0; i < kHalfDimensions; ++i) biases_[i] = read_little_endian<BiasType>(stream);
		for (std::size_t i = 0; i < kHalfDimensions * kInputDimensions; ++i)
			wdst[i] = read_little_endian<WeightType>(stream);
#endif
#if defined(THREAT_ATTACKER_MAJOR)
		// threat 行を attacker-major へロード時置換 (task#59 / report/51 §7.6.2)。
		// RawFeatures = FeatureSet<Threat, HalfKP> なので threat の先頭行 = HalfKP::kDimensions。
		if (!stream.fail())
			Features::Threat::PermuteRows(wdst, kHalfDimensions,
			                              Features::HalfKP<Features::Side::kFriend>::kDimensions);
#endif
#if defined(NNUE_FT_INT8_ROWS)
		if (!stream.fail()) {
			// nn.bin の値 (w×127) をそのまま int8 に。|v| > 127 は clip し、差分を残差表に (王者 tsfnn-526 で 6,900 / 357M = 0.002%)。
			std::size_t clipped = 0;
			for (std::size_t i = 0; i < kWeightsCount; ++i) {
				const int v = wtmp[i];
				const int c = v > 127 ? 127 : (v < -127 ? -127 : v);
				clipped += std::size_t(c != v);
				weights_[i] = static_cast<RowType>(c);
			}
			n_fix_rows_ = 0; n_fix_ = 0;
			for (IndexType r = 0; r < kInputDimensions; ++r) {
				const std::size_t base = row_off(r);
#if defined(NNUE_THREAT_NARROW_COLS)
				const IndexType width = r < kTailRows ? kHalfDimensions : kNarrowWidth;
#else
				const IndexType width = kHalfDimensions;
#endif
				bool any = false;
				for (IndexType c = 0; c < width; ++c) {
					const int v = wtmp[base + c];
					if (v > 127 || v < -127) {
						if (n_fix_ + 2 > kMaxFix || n_fix_rows_ >= kMaxFixRows) {
							std::fprintf(stderr, "Error! NNUE_FT_INT8_ROWS: residual table overflow (rows %u, entries %u)\n", n_fix_rows_, n_fix_);
							return Tools::ResultCode::FileReadError;
						}
						if (!any) {
							// この行は残差あり: 先頭要素を番兵 −128 にし、その本当の値 v0 (clip 済み) との差を col 0 の残差として先に積む
							any = true;
							fix_idx_[r] = n_fix_rows_; fix_start_[n_fix_rows_] = n_fix_; ++n_fix_rows_;
							const int v0 = wtmp[base];
							const int c0 = v0 > 127 ? 127 : (v0 < -127 ? -127 : v0);
							fix_[n_fix_++] = FixEntry{std::uint16_t(0), std::int16_t(v0 - int(kFixSentinel))};
							weights_[base] = kFixSentinel;
							if (c == 0) continue;   // col 0 の残差は上で (clip 分も込みで) 積んだ
							(void)c0;
						}
						fix_[n_fix_++] = FixEntry{std::uint16_t(c), std::int16_t(v - (v > 127 ? 127 : -127))};
					}
				}
			}
			fix_start_[n_fix_rows_] = n_fix_;
			if (clipped)
				std::fprintf(stderr, "NNUE_FT_INT8_ROWS: %llu of %llu weights beyond [-127, 127] -> int8 + residual table (%u rows)\n",
				             (unsigned long long)clipped, (unsigned long long)kWeightsCount, n_fix_rows_);
		}
#endif
		return !stream.fail() ? Tools::ResultCode::Ok : Tools::ResultCode::FileReadError;
	}

	// Write network parameters
	// パラメータを書き込む
	bool WriteParameters(std::ostream& stream) const {
#if defined(THREAT_ATTACKER_MAJOR) || defined(NNUE_THREAT_NARROW_COLS) || defined(NNUE_FT_INT8_ROWS)
		// 並び替え済み配置 / int8 格納の保存は未対応 (標準 pair-major へ逆置換していないため、
		// このまま書くと他ビルドで読めない nn.bin ができる)。学習系はこのビルドで使わないこと。
		return false;
#else
		stream.write(reinterpret_cast<const char*>(biases_), kHalfDimensions * sizeof(BiasType));
		stream.write(reinterpret_cast<const char*>(weights_), kHalfDimensions * kInputDimensions * sizeof(WeightType));
		return !stream.fail();
#endif
	}

	// 特徴 index → 重み行列内オフセット。
	// 計測ビルド FT_STAT_ALIAS_THREAT_ROWS (report/52 §21.2): Head (threat) 行を 4096 行に畳んで作業集合を 8 MB にし、
	// DRAM 分のコストを直接引く (評価は変わるが行/update の統計は同じ)。通常ビルドでは恒等。
	static inline IndexType row_off(IndexType index) {
#if defined(FT_STAT_ALIAS_THREAT_ROWS)
		if (index >= RawFeatures::kTailDimensions)
			index = RawFeatures::kTailDimensions + ((index - RawFeatures::kTailDimensions) & 4095u);
#endif
#if defined(NNUE_THREAT_NARROW_COLS)
		if (index >= kTailRows)
			return kHalfDimensions * kTailRows + kNarrowWidth * (index - kTailRows);
#endif
		return kHalfDimensions * index;
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
		vec_t One = vec_set_16(kFtHalfScale ? 127 : 127 * 2);   // int8 行では acc が半スケール

		const Color perspectives[2] = { pos.side_to_move(), ~pos.side_to_move() };
		for (IndexType p = 0; p < 2; ++p) {
			const IndexType offset = (kHalfDimensions / 2) * p;

			// ★複数 refresh trigger (halfka2t 等) では平面を全て合算する。
			//   平面 0 固定読みだと threat 等の追加平面が出力に乗らない (task#54 で実害)。
			vec_t* out = reinterpret_cast<vec_t*>(output + offset);
#if defined(NNUE_THREAT_NARROW_COLS) && NNUE_THREAT_NARROW_BLOCKS == 1
			// narrow slot (threat) は先頭 2N 要素 = [前半の先頭 N 列][後半の先頭 N 列]。full slot (KA2 + バイアス) に
			// 各半分の先頭 N 列 (= kNarrowChunks ベクトル) だけ足す。(ブロック疎 B>1 では narrow slot が full 幅なので通常経路)
			constexpr IndexType kVecElems     = sizeof(vec_t) / sizeof(BiasType);
			constexpr IndexType kNarrowChunks = kNarrowCols / kVecElems;
			static_assert(kNarrowCols % kVecElems == 0, "narrow cols must be a multiple of the vector width");
			const vec_t* in0 = reinterpret_cast<const vec_t*>(&(accumulation[perspectives[p]][kBiasSlot][0]));
			const vec_t* in1 = reinterpret_cast<const vec_t*>(&(accumulation[perspectives[p]][kBiasSlot][kHalfDimensions / 2]));
			const vec_t* nr  = reinterpret_cast<const vec_t*>(&(accumulation[perspectives[p]][kNarrowSlot][0]));
			auto load0 = [&](IndexType idx) {
				vec_t v = in0[idx];
				if (idx < kNarrowChunks) v = vec_add_16(v, nr[idx]);
				return v;
			};
			auto load1 = [&](IndexType idx) {
				vec_t v = in1[idx];
				if (idx < kNarrowChunks) v = vec_add_16(v, nr[kNarrowChunks + idx]);
				return v;
			};
#else
			const vec_t* in0 = reinterpret_cast<const vec_t*>(&(accumulation[perspectives[p]][0][0]));
			const vec_t* in1 = reinterpret_cast<const vec_t*>(&(accumulation[perspectives[p]][0][kHalfDimensions / 2]));
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
#endif

			// int8 行 (kFtHalfScale): a', c' は従来の 1/2 なので (a'<<(s+1))·(c'<<1) >> 16 = (2a')(2c') >> (16−s) で従来と一致
			constexpr int shift =
#if defined(USE_SSE2)
				7 + kFtHalfScale;
#else
				6 + kFtHalfScale;
#endif

			for (IndexType j = 0; j < NumOutputChunks; ++j)
			{
				const vec_t sum0a =
					vec_slli_16(vec_max_16(vec_min_16(load0(j * 2 + 0), One), Zero), shift);
				const vec_t sum0b =
					vec_slli_16(vec_max_16(vec_min_16(load0(j * 2 + 1), One), Zero), shift);
				vec_t sum1a = vec_min_16(load1(j * 2 + 0), One);
				vec_t sum1b = vec_min_16(load1(j * 2 + 1), One);
				if constexpr (kFtHalfScale) { sum1a = vec_slli_16(sum1a, 1); sum1b = vec_slli_16(sum1b, 1); }

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
#if defined(NNUE_THREAT_NARROW_COLS) && NNUE_THREAT_NARROW_BLOCKS == 1
				BiasType sum0 = accumulation[perspectives[p]][kBiasSlot][j];
				BiasType sum1 = accumulation[perspectives[p]][kBiasSlot][j + kHalfDimensions / 2];
				if (j < kNarrowCols) {
					sum0 += accumulation[perspectives[p]][kNarrowSlot][j];
					sum1 += accumulation[perspectives[p]][kNarrowSlot][kNarrowCols + j];
				}
#else
				BiasType sum0 = accumulation[perspectives[p]][0][j];
				BiasType sum1 = accumulation[perspectives[p]][0][j + kHalfDimensions / 2];
				for (IndexType t = 1; t < kRefreshTriggers.size(); ++t) {
					sum0 += accumulation[perspectives[p]][t][j];
					sum1 += accumulation[perspectives[p]][t][j + kHalfDimensions / 2];
				}
#endif
				sum0 = std::clamp<BiasType>(sum0, 0, kFtHalfScale ? 127 : 127 * 2);
				sum1 = std::clamp<BiasType>(sum1, 0, kFtHalfScale ? 127 : 127 * 2);
				output[offset + j] = static_cast<OutputType>(unsigned(sum0 * sum1) / (kFtHalfScale ? 128 : 512));
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
#if defined(NNUE_THREAT_NARROW_COLS)
	// read_leb_128 (nnue_common.h) と同じ符号化を、配列に置かずに sink(i, value) へ流す版 (narrow 格納用)。
	template <typename IntType, typename Sink>
	static void read_leb_128_sink(std::istream& stream, std::size_t count, Sink sink) {
		char leb128MagicString[Leb128MagicStringSize];
		stream.read(leb128MagicString, Leb128MagicStringSize);
		static_assert(std::is_signed_v<IntType>, "Not implemented for unsigned types");
		const std::uint32_t BUF_SIZE = 4096;
		std::uint8_t        buf[BUF_SIZE];
		auto bytes_left = read_little_endian<std::uint32_t>(stream);
		std::uint32_t buf_pos = BUF_SIZE;
		for (std::size_t i = 0; i < count; ++i) {
			IntType result = 0;
			size_t  shift = 0;
			do {
				if (buf_pos == BUF_SIZE) {
					stream.read(reinterpret_cast<char*>(buf), std::min(bytes_left, BUF_SIZE));
					buf_pos = 0;
				}
				std::uint8_t byte = buf[buf_pos++];
				--bytes_left;
				result |= (byte & 0x7f) << shift;
				shift += 7;
				if ((byte & 0x80) == 0) {
					sink(i, IntType((sizeof(IntType) * 8 <= shift || (byte & 0x40) == 0) ? result : result | ~((1 << shift) - 1)));
					break;
				}
			} while (shift < sizeof(IntType) * 8);
		}
	}
#endif

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

	// wts = 並び替える int16 の重み配列 (通常は weights_、NNUE_FT_INT8_ROWS ではロード時の int16 一時領域)
	void permute_weights([[maybe_unused]] void (*order_fn)(uint64_t*), [[maybe_unused]] WeightType* wts) const {
#if defined(USE_AVX2)
#if defined(USE_AVX512)
		constexpr IndexType di = 16;
#else
		constexpr IndexType di = 8;
#endif
		uint64_t* b = reinterpret_cast<uint64_t*>(const_cast<BiasType*>(&biases_[0]));
		for (IndexType i = 0; i < kHalfDimensions * sizeof(BiasType) / sizeof(uint64_t); i += di)
			order_fn(&b[i]);

#if defined(NNUE_THREAT_NARROW_COLS)
		// Tail 行は全幅、Head 行は 2N 幅。並び替えは 64 B (AVX2) / 128 B (AVX-512) の群内で閉じているので、
		// スライス列 (群境界に揃う) を圧縮した行にも同じ群単位で適用できる。
		for (IndexType j = 0; j < kTailRows; ++j) {
			uint64_t* w = reinterpret_cast<uint64_t*>(&wts[std::size_t(j) * kHalfDimensions]);
			for (IndexType i = 0; i < kHalfDimensions * sizeof(WeightType) / sizeof(uint64_t); i += di)
				order_fn(&w[i]);
		}
		static_assert((kNarrowWidth * sizeof(WeightType) / sizeof(uint64_t)) % di == 0, "narrow row must be whole permute groups");
		for (IndexType j = 0; j < kHeadRows; ++j) {
			uint64_t* w = reinterpret_cast<uint64_t*>(
				&wts[std::size_t(kHalfDimensions) * kTailRows + std::size_t(j) * kNarrowWidth]);
			for (IndexType i = 0; i < kNarrowWidth * sizeof(WeightType) / sizeof(uint64_t); i += di)
				order_fn(&w[i]);
		}
#else
		for (IndexType j = 0; j < kInputDimensions; ++j)
		{
			uint64_t* w = reinterpret_cast<uint64_t*>(&wts[std::size_t(j) * kHalfDimensions]);
			for (IndexType i = 0; i < kHalfDimensions * sizeof(WeightType) / sizeof(uint64_t);
					i += di)
				order_fn(&w[i]);
		}
#endif
#endif
	}

#if !defined(NNUE_FT_INT8_ROWS)
	inline void scale_weights(bool read) const {
		// 全要素一様なので行構造 (narrow の Head 行含む) に依らず配列全体を走査する
		WeightType* w = const_cast<WeightType*>(weights_);
		for (std::size_t i = 0; i < kWeightsCount; ++i)
			w[i] = read ? w[i] * 2 : w[i] / 2;

		BiasType* b = const_cast<BiasType*>(biases_);
		for (IndexType i = 0; i < kHalfDimensions; ++i)
			b[i] = read ? b[i] * 2 : b[i] / 2;
	}
#endif

	// Calculate cumulative value without using difference calculation
	// 差分計算を用いずに累積値を計算する
	void refresh_accumulator(const Position& pos) const {
#if defined(ENABLE_FT_TRAFFIC_STAT)
		const uint64_t t_ref0 = __rdtsc();
		struct RefreshTimer { uint64_t t0; ~RefreshTimer() { g_ft_stat.cyc_refresh += __rdtsc() - t0; } } t_ref_timer{t_ref0};
#endif
		auto& accumulator = pos.state()->accumulator;
		for (IndexType i = 0; i < kRefreshTriggers.size(); ++i) {
			Features::IndexList active_indices[2];
			RawFeatures::AppendActiveIndices(pos, kRefreshTriggers[i], active_indices);
#if defined(ENABLE_FT_TRAFFIC_STAT)
			if (i == 0) g_ft_stat.n_refresh++;
#endif
			for (Color perspective : {BLACK, WHITE}) {
#if defined(ENABLE_ACC_CACHE)
				if (kRefreshTriggers[i] == Features::TriggerEvent::kFriendKingMoved) {
					build_from_cache(perspective, pos.square<KING>(perspective), active_indices[perspective],
					                 accumulator.accumulation[perspective][i], i == kBiasSlot);
					continue;
				}
#endif
#if defined(ENABLE_FT_TRAFFIC_STAT)
				g_ft_stat.rows_full += active_indices[perspective].size();
				const uint64_t t_full0 = __rdtsc();
#endif
#if defined(VECTOR)
#if defined(NNUE_FT_TILED)
				if (tiled_slot(i)) {
					apply_rows_tiled(&accumulator.accumulation[perspective][i][0], i == kBiasSlot ? biases_ : nullptr, slot_width(i),
					                 nullptr, 0, active_indices[perspective].begin(), active_indices[perspective].size());
				} else
#endif
				{
					if (i == kBiasSlot) {
						std::memcpy(accumulator.accumulation[perspective][i], biases_, kHalfDimensions * sizeof(BiasType));
					} else {
						std::memset(accumulator.accumulation[perspective][i], 0, slot_width(i) * sizeof(BiasType));
					}
					for (const auto index : active_indices[perspective]) {
#if defined(NNUE_THREAT_NARROW_COLS)
						if (narrow_blocked() && i == kNarrowSlot) { apply_narrow_row<true>(&accumulator.accumulation[perspective][i][0], index); continue; }
#endif
						const IndexType offset = row_off(index);
						auto accumulation      = reinterpret_cast<vec_t*>(&accumulator.accumulation[perspective][i][0]);
						const RowType* column  = &weights_[offset];
						const IndexType kNumChunks = slot_width(i) / kVecElems;
						for (IndexType j = 0; j < kNumChunks; ++j) {
							accumulation[j] = vec_add_16(accumulation[j], vec_load_row(column + j * kVecElems));
						}
						fix_row<true>(&accumulator.accumulation[perspective][i][0], index);
					}
				}
#if defined(ENABLE_FT_TRAFFIC_STAT)
				g_ft_stat.cyc_full += __rdtsc() - t_full0;
#endif
#else
				if (i == kBiasSlot) {
					std::memcpy(accumulator.accumulation[perspective][i], biases_, kHalfDimensions * sizeof(BiasType));
				} else {
					std::memset(accumulator.accumulation[perspective][i], 0, slot_width(i) * sizeof(BiasType));
				}
				for (const auto index : active_indices[perspective]) {
#if defined(NNUE_THREAT_NARROW_COLS)
					if (narrow_blocked() && i == kNarrowSlot) { apply_narrow_row<true>(&accumulator.accumulation[perspective][i][0], index); continue; }
#endif
					const IndexType offset = row_off(index);

					for (IndexType j = 0; j < slot_width(i); ++j) {
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
#if defined(ENABLE_FT_TRAFFIC_STAT)
		const uint64_t t_upd0 = __rdtsc();
#endif
		// ★参照 (2026-09-18): 従来は値コピーで accumulator 構造体 (全 slot × 両視点、8 KB 超) を update のたびに丸写ししていた
		const auto& prev_accumulator = pos.state()->previous->accumulator;
		auto&       accumulator      = pos.state()->accumulator;
		for (IndexType i = 0; i < kRefreshTriggers.size(); ++i) {
			Features::IndexList removed_indices[2], added_indices[2];
			bool                reset[2];
#if defined(ENABLE_FT_TRAFFIC_STAT)
			const uint64_t t_col0 = __rdtsc();
#endif
			RawFeatures::AppendChangedIndices(pos, kRefreshTriggers[i], removed_indices, added_indices, reset);
#if defined(ENABLE_FT_TRAFFIC_STAT)
			g_ft_stat.cyc_collect += __rdtsc() - t_col0;
			if (i == 0) g_ft_stat.n_update++;
			for (Color pc : {BLACK, WHITE}) {
				const uint64_t rows = removed_indices[pc].size() + added_indices[pc].size();
				// ★reset は王移動。added が全 active になるので「全再構築」に数える
				//   (ENABLE_ACC_CACHE で玉移動 slot がキャッシュ経由になるときは rows_cache 側で数える)
				if (reset[pc]) {
					g_ft_stat.n_reset++;
#if defined(ENABLE_ACC_CACHE)
					if (kRefreshTriggers[i] != Features::TriggerEvent::kFriendKingMoved)
#endif
					g_ft_stat.rows_full += rows;
				}
				else           { g_ft_stat.rows_inc += rows; }
			}
#endif
#if defined(FT_ROW_PREFETCH)
			// 差分行の先出しプリフェッチ (task#59): index は確定済みなので、accumulate に入る前に
			// 全行 (両視点) のフェッチを重ねて発行する。行 = kHalfDimensions×2B、先頭と中間の
			// 2 ライン → 残りは HW ストリームプリフェッチに任せる。意味論は不変。
			for (Color pf_p : {BLACK, WHITE}) {
				for (const auto index : removed_indices[pf_p]) {
					const auto* row = reinterpret_cast<const char*>(&weights_[row_off(index)]);
					_mm_prefetch(row, _MM_HINT_T0);
					_mm_prefetch(row + row_width(i) * sizeof(RowType) / 2, _MM_HINT_T0);   // 行の中間 (bytes = dims*2/2)
				}
				for (const auto index : added_indices[pf_p]) {
					const auto* row = reinterpret_cast<const char*>(&weights_[row_off(index)]);
					_mm_prefetch(row, _MM_HINT_T0);
					_mm_prefetch(row + row_width(i) * sizeof(RowType) / 2, _MM_HINT_T0);
				}
			}
#endif
			for (Color perspective : {BLACK, WHITE}) {
#if defined(ENABLE_ACC_CACHE)
				// 玉移動 slot の reset = 全 active (added_indices に入っている) をキャッシュとの集合差分で作る
				if (reset[perspective] && kRefreshTriggers[i] == Features::TriggerEvent::kFriendKingMoved) {
					build_from_cache(perspective, pos.square<KING>(perspective), added_indices[perspective],
					                 accumulator.accumulation[perspective][i], i == kBiasSlot);
					continue;
				}
#endif
#if defined(NNUE_FT_TILED)
				if (tiled_slot(i)) {
					BiasType* out = &accumulator.accumulation[perspective][i][0];
					if (reset[perspective])
						apply_rows_tiled(out, i == kBiasSlot ? biases_ : nullptr, slot_width(i), nullptr, 0,
						                 added_indices[perspective].begin(), added_indices[perspective].size());
					else
						apply_rows_tiled(out, &prev_accumulator.accumulation[perspective][i][0], slot_width(i),
						                 removed_indices[perspective].begin(), removed_indices[perspective].size(),
						                 added_indices[perspective].begin(), added_indices[perspective].size());
					continue;
				}
#endif
#if defined(VECTOR)
				const IndexType kNumChunks = slot_width(i) / (sizeof(vec_t) / sizeof(BiasType));
				auto accumulation          = reinterpret_cast<vec_t*>(&accumulator.accumulation[perspective][i][0]);
#endif
				if (reset[perspective]) {
					if (i == kBiasSlot) {
						std::memcpy(accumulator.accumulation[perspective][i], biases_,
						            kHalfDimensions * sizeof(BiasType));
					} else {
						std::memset(accumulator.accumulation[perspective][i], 0, slot_width(i) * sizeof(BiasType));
					}
				} else {
					// Difference calculation for the feature amount changed from 1 to 0
					// 1から0に変化した特徴量に関する差分計算
					std::memcpy(accumulator.accumulation[perspective][i], prev_accumulator.accumulation[perspective][i],
					            slot_width(i) * sizeof(BiasType));
					for (const auto index : removed_indices[perspective]) {
#if defined(NNUE_THREAT_NARROW_COLS)
						if (narrow_blocked() && i == kNarrowSlot) { apply_narrow_row<false>(&accumulator.accumulation[perspective][i][0], index); continue; }
#endif
						const IndexType offset = row_off(index);
#if defined(ENABLE_FT_TRAFFIC_STAT) && !defined(FT_STAT_NO_ROW_TIMING)
						const uint64_t t_row0 = __rdtsc();
#endif
#if defined(VECTOR)
						const RowType* column = &weights_[offset];
						for (IndexType j = 0; j < kNumChunks; ++j) {
							accumulation[j] = vec_sub_16(accumulation[j], vec_load_row(column + j * kVecElems));
						}
						fix_row<false>(&accumulator.accumulation[perspective][i][0], index);
#else
						for (IndexType j = 0; j < slot_width(i); ++j) {
							accumulator.accumulation[perspective][i][j] -= weights_[offset + j];
						}
#endif
#if defined(ENABLE_FT_TRAFFIC_STAT) && !defined(FT_STAT_NO_ROW_TIMING)
						{
							const uint64_t dt = __rdtsc() - t_row0;
							if (index >= RawFeatures::kTailDimensions) { g_ft_stat.cyc_inc_head += dt; g_ft_stat.rows_inc_head++; }
							else                                       { g_ft_stat.cyc_inc_tail += dt; g_ft_stat.rows_inc_tail++; }
						}
#endif
					}
				}
				{
					// Difference calculation for features that changed from 0 to 1
					// 0から1に変化した特徴量に関する差分計算
					for (const auto index : added_indices[perspective]) {
#if defined(NNUE_THREAT_NARROW_COLS)
						if (narrow_blocked() && i == kNarrowSlot) { apply_narrow_row<true>(&accumulator.accumulation[perspective][i][0], index); continue; }
#endif
						const IndexType offset = row_off(index);
#if defined(ENABLE_FT_TRAFFIC_STAT) && !defined(FT_STAT_NO_ROW_TIMING)
						const uint64_t t_row0 = __rdtsc();
#endif
#if defined(VECTOR)
						const RowType* column = &weights_[offset];
						for (IndexType j = 0; j < kNumChunks; ++j) {
							accumulation[j] = vec_add_16(accumulation[j], vec_load_row(column + j * kVecElems));
						}
						fix_row<true>(&accumulator.accumulation[perspective][i][0], index);
#else
						for (IndexType j = 0; j < slot_width(i); ++j) {
							accumulator.accumulation[perspective][i][j] += weights_[offset + j];
						}
#endif
#if defined(ENABLE_FT_TRAFFIC_STAT) && !defined(FT_STAT_NO_ROW_TIMING)
						{
							const uint64_t dt = __rdtsc() - t_row0;
							if (reset[perspective]) { g_ft_stat.cyc_full += dt; }
							else if (index >= RawFeatures::kTailDimensions) { g_ft_stat.cyc_inc_head += dt; g_ft_stat.rows_inc_head++; }
							else                                            { g_ft_stat.cyc_inc_tail += dt; g_ft_stat.rows_inc_tail++; }
						}
#endif
					}
				}
			}
		}
#if defined(ENABLE_FT_TRAFFIC_STAT)
		g_ft_stat.cyc_update += __rdtsc() - t_upd0;
#endif

		accumulator.computed_accumulation = true;
		// Stockfishでは fc27d15(2020-09-07) にcomputed_scoreが排除されているので確認
		accumulator.computed_score = false;
	}

#if defined(ENABLE_ACC_CACHE)
	// 1 行の加算/減算 (キャッシュ差分用)
	inline void add_row(BiasType* acc, IndexType index) const {
		const IndexType offset = row_off(index);
#if defined(VECTOR)
		constexpr IndexType kNumChunks = kHalfDimensions / kVecElems;
		auto a = reinterpret_cast<vec_t*>(acc);
		const RowType* column = &weights_[offset];
		for (IndexType j = 0; j < kNumChunks; ++j) a[j] = vec_add_16(a[j], vec_load_row(column + j * kVecElems));
#else
		for (IndexType j = 0; j < kHalfDimensions; ++j) acc[j] += weights_[offset + j];
#endif
		fix_row<true>(acc, index);
	}
	inline void sub_row(BiasType* acc, IndexType index) const {
		const IndexType offset = row_off(index);
#if defined(VECTOR)
		constexpr IndexType kNumChunks = kHalfDimensions / kVecElems;
		auto a = reinterpret_cast<vec_t*>(acc);
		const RowType* column = &weights_[offset];
		for (IndexType j = 0; j < kNumChunks; ++j) a[j] = vec_sub_16(a[j], vec_load_row(column + j * kVecElems));
#else
		for (IndexType j = 0; j < kHalfDimensions; ++j) acc[j] -= weights_[offset + j];
#endif
		fix_row<false>(acc, index);
	}

	// active (未ソート、この視点の玉移動 slot の全 active index) から out を作る。
	// キャッシュ [perspective][ksq] が有効ならその accumulator との集合差分、無効なら通常の全加算。
	// 作った結果と active をキャッシュに書き戻す。with_bias = slot 0 (バイアスを含む) かどうか。
	void build_from_cache(Color perspective, Square ksq, Features::IndexList& active, BiasType* out,
	                      bool with_bias) const {
		AccCache& C = acc_cache();
		if (C.generation != load_generation_) {
			C.clear();
			C.generation = load_generation_;
		}
		AccCacheEntry& ent = C.e[perspective][ksq];
		std::sort(active.begin(), active.end());
		if (!ent.valid) {
#if defined(NNUE_FT_TILED)
			apply_rows_tiled(out, with_bias ? biases_ : nullptr, kHalfDimensions, nullptr, 0, active.begin(), active.size());
#else
			if (with_bias)
				std::memcpy(out, biases_, kHalfDimensions * sizeof(BiasType));
			else
				std::memset(out, 0, kHalfDimensions * sizeof(BiasType));
			for (const auto index : active) add_row(out, index);
#endif
#if defined(ENABLE_FT_TRAFFIC_STAT)
			g_ft_stat.rows_full += active.size();
#endif
		} else {
			// ソート済み多重集合の差分 (merge walk): キャッシュにあって今無い → 減算、今あってキャッシュに無い → 加算
			std::size_t a = 0, b = 0;
			const std::size_t na = ent.n, nb = active.size();
			[[maybe_unused]] uint64_t applied = 0;
#if defined(NNUE_FT_TILED)
			IndexType rem[RawFeatures::kMaxActiveDimensions], add[RawFeatures::kMaxActiveDimensions];
			std::size_t n_rem = 0, n_add = 0;
			while (a < na || b < nb) {
				if (b >= nb || (a < na && ent.indices[a] < active[b])) {
					rem[n_rem++] = ent.indices[a]; ++a; ++applied;
				} else if (a >= na || active[b] < ent.indices[a]) {
					add[n_add++] = active[b]; ++b; ++applied;
				} else {
					++a; ++b;
				}
			}
			apply_rows_tiled(out, ent.accumulation, kHalfDimensions, rem, n_rem, add, n_add);
#else
			std::memcpy(out, ent.accumulation, kHalfDimensions * sizeof(BiasType));
			while (a < na || b < nb) {
				if (b >= nb || (a < na && ent.indices[a] < active[b])) {
					sub_row(out, ent.indices[a]); ++a; ++applied;
				} else if (a >= na || active[b] < ent.indices[a]) {
					add_row(out, active[b]); ++b; ++applied;
				} else {
					++a; ++b;
				}
			}
#endif
#if defined(ENABLE_FT_TRAFFIC_STAT)
			g_ft_stat.rows_cache += applied;
			g_ft_stat.n_cache_hit++;
#endif
		}
		std::memcpy(ent.accumulation, out, kHalfDimensions * sizeof(BiasType));
		for (std::size_t k = 0; k < active.size(); ++k) ent.indices[k] = active[k];
		ent.n = static_cast<std::uint16_t>(active.size());
		ent.valid = true;
	}

	mutable std::uint32_t load_generation_ = 1;
#endif

	// parameter type
	// パラメータの型

	// Make the learning class a friend
	// 学習用クラスをfriendにする
	friend class Trainer<FeatureTransformer>;

	// parameter
	// パラメータ
	alignas(kCacheLineSize) BiasType biases_[kHalfDimensions];
	// narrow (NNUE_THREAT_NARROW_COLS) では [Tail 行 × kHalf][Head 行 × kNarrowWidth] の連結 (row_off で引く)
	// NNUE_FT_INT8_ROWS では int8 (nn.bin の値そのまま、半スケール)、通常は int16 (ロード時 ×2 済)
	alignas(kCacheLineSize) RowType weights_[kWeightsCount];
};

} // namespace Eval::NNUE
} // namespace YaneuraOu

#endif  // defined(EVAL_NNUE)

#endif  // #ifndef NNUE_FEATURE_TRANSFORMER_H_INCLUDED
