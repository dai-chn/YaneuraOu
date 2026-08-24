// NNUE 入力特徴量 ThreatLite の定義 (task#59 / report/51 §7.4-7.5)
//
// Threat (216,720 次元) の from/幾何を落とした縮約版:
//   feature = (attacker_side, attacker_class, defender_side, defender_class, to_sq)
//   次元 = 324 pair × 81 to = 26,244
// bullet-shogi 側 `shogi_halfkp_threatlite.rs` と**同一 index 仕様**。
//
// ★count 意味論: 同じ (pair, to) に複数の攻撃駒が居る場合、同一 index を
//   重複して push する (特徴値 = 攻撃駒数)。accumulator の行加算が重複分
//   積まれることで自然に count になる。bullet 側の sparse gather と同一。
//
// ★当面はナイーブ全再構築 (kAnyPieceMoved)。差分更新 (純加減算で参照カウント
//   不要になる設計 — report/51 §7.5) は Elo 保持率 A/B が通ってから実装する。
//   判定対局は固定ノードなので NPS は結果に影響しない。

#ifndef CLASSIC_NNUE_FEATURES_THREAT_LITE_H
#define CLASSIC_NNUE_FEATURES_THREAT_LITE_H

#include "../../../config.h"

#if defined(EVAL_NNUE)

#include "../../../evaluate.h"
#include "features_common.h"

namespace YaneuraOu {
namespace Eval::NNUE::Features {

// 特徴量 ThreatLite: 駒の実利きが駒に当たっている関係 (from-drop 縮約)
class ThreatLite {
 public:
  // 特徴量名
  static constexpr const char* kName = "ThreatLite(FromDrop)";

  // 評価関数ファイルに埋め込むハッシュ値。
  // bullet 側 (exp004v main.rs) のヘッダ hash = HalfKP_hash ^ "TLTE"
  //   = 0x5D69D5B8 ^ 0x544C5445 = 0x092581FD
  // YO の FeatureSet<ThreatLite, HalfKP> の合成式
  //   composite = ThreatLite::kHashValue ^ (KP << 1) ^ (KP >> 31)
  // がその値になるよう逆算 (KP = 0x5D69D5B8, KP<<1 = 0xBAD3AB70, KP>>31 = 0):
  //   0x092581FD ^ 0xBAD3AB70 = 0xB3F62A8D
  static constexpr std::uint32_t kHashValue = 0xB3F62A8Du;

  // 特徴量の次元数 (2*9*2*9 pair × 81 to)
  static constexpr IndexType kDimensions = 26244;

  // 同時にアクティブになりうる最大特徴数 (重複 emit 込み。full threat と同じ上限)
  static constexpr IndexType kMaxActiveDimensions = 320;

  // ナイーブ全再構築 (毎手 reset)。差分化は A/B 通過後 (ヘッダ冒頭コメント参照)。
  static constexpr TriggerEvent kRefreshTrigger = TriggerEvent::kAnyPieceMoved;

  // 特徴量のインデックスのリストを取得する (★重複 push あり = count 意味論)
  static void AppendActiveIndices(const Position& pos, Color perspective,
                                  IndexList* active);

  // kAnyPieceMoved は常に reset になるので呼ばれない
  static void AppendChangedIndices(const Position& pos, Color perspective,
                                   IndexList* removed, IndexList* added);
};

} // namespace Eval::NNUE::Features
} // namespace YaneuraOu

#endif  // defined(EVAL_NNUE)

#endif
