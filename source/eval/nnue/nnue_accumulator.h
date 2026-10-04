// NNUE評価関数の差分計算用のクラス

#ifndef CLASSIC_NNUE_ACCUMULATOR_H_
#define CLASSIC_NNUE_ACCUMULATOR_H_

#include "../../config.h"

#if defined(EVAL_NNUE)

#include "nnue_architecture.h"

namespace YaneuraOu {
namespace Eval::NNUE {

// 入力特徴量をアフィン変換した結果を保持するクラス
// 最終的な出力である評価値も一緒に持たせておく
// AVX-512命令を使用する場合に64bytesのアライメントが要求される。
struct alignas(64) Accumulator {
  std::int16_t
      accumulation[2][kRefreshTriggers.size()][kTransformedFeatureDimensions];
  Value score = VALUE_ZERO;
  bool computed_accumulation = false;
  bool computed_score = false;
#if defined(ENABLE_LAZY_THREAT)
  // lazy threat gate (A2、docs/superpowers/specs/2026-10-04-lazy-threat-a2.md):
  //   slot 0 (threat) の行加算を先送りする。pend_* = この手で列挙した threat 差分 (前局面との差)。
  static constexpr int kLazyPendMax = 128;
  bool          threat_computed = false;   // accumulation[*][0] が有効か
  bool          pend_valid      = false;   // pend_* がこの手の差分として有効か (do_move で false、列挙時に true、null move で空の true)
  bool          pend_overflow   = false;   // 差分が kLazyPendMax を超えた (追いつきは全再構築)
  std::uint16_t pend_n_rem[2] = {0, 0}, pend_n_add[2] = {0, 0};
  std::uint32_t pend_rem[2][kLazyPendMax], pend_add[2][kLazyPendMax];
#endif
};

} // namespace Eval::NNUE
} // namespace YaneuraOu

#endif  // defined(EVAL_NNUE)

#endif
