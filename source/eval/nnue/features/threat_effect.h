// NNUE 入力特徴量 ThreatEffect の定義 (task#73 v2 / report/52 §18.10)
//
// 「駒 to に side as の利きが何本当たっているか」を長い利き/短い利きに分けてバケット化した特徴:
//   feature = (attacker_side, defender_side, defender_class, to_sq, long 0/1/2+, short 0/1/2+)
//   次元 = 2 * 2 * 9 * 81 * 3 * 3 = 26,244
// bullet-shogi 側 `shogi_halfkp_threat_effect.rs` と**同一 index 仕様**。
//
// 定義は LONG_EFFECT_LIBRARY (board_effect / long_effect) と一致させてある:
//   - 攻撃側は玉を含む。被弾側は玉を除外 (threat 系と同じ)。
//   - 長い利き = 香/角/飛の射程 + 馬の斜め射程 + 龍の縦横射程 (隣接升も含む)。
//     短い利き = 歩/桂/銀/金類/玉の 1 歩 + 馬の縦横 1 歩 + 龍の斜め 1 歩。
//   → 将来 board_effect (利き数) と long_effect (方向 popcount) から列挙なしで計算できる。
// ★本ファイルは Elo 保持率 A/B 用の判定実装: 列挙で利き数を数え、ナイーブ全再構築 (kAnyPieceMoved)。
//
// 意味論: 被弾駒 1 つ × 攻撃側 2 色 = 常に 2 特徴 (利き 0 本の (0,0) 状態も emit)。重複 emit なし。

#ifndef CLASSIC_NNUE_FEATURES_THREAT_EFFECT_H
#define CLASSIC_NNUE_FEATURES_THREAT_EFFECT_H

#include "../../../config.h"

#if defined(EVAL_NNUE)

#include "../../../evaluate.h"
#include "features_common.h"

namespace YaneuraOu {
namespace Eval::NNUE::Features {

// 特徴量 ThreatEffect: 駒に当たっている利き数 (長/短) の状態
class ThreatEffect {
 public:
  // 特徴量名
  static constexpr const char* kName = "ThreatEffect(LongShort)";

  // 評価関数ファイルに埋め込むハッシュ値。
  // bullet 側 (exp004z main.rs) のヘッダ hash = HalfKP_hash ^ "TEF2"
  //   = 0x5D69D5B8 ^ 0x54454632 = 0x092C938A
  // YO の FeatureSet<ThreatEffect, HalfKP> の合成式
  //   composite = ThreatEffect::kHashValue ^ (KP << 1) ^ (KP >> 31)
  // がその値になるよう逆算 (KP<<1 = 0xBAD3AB70, KP>>31 = 0):
  //   0x092C938A ^ 0xBAD3AB70 = 0xB3FF38FA
  static constexpr std::uint32_t kHashValue = 0xB3FF38FAu;

  // 特徴量の次元数
  static constexpr IndexType kDimensions = 26244;

  // 同時にアクティブになりうる最大特徴数 (玉以外の駒 ≤ 38 × 攻撃側 2 色)
  static constexpr IndexType kMaxActiveDimensions = 80;

  // ナイーブ全再構築 (毎手 reset)。差分化 (board_effect 差分) は A/B 通過後
  static constexpr TriggerEvent kRefreshTrigger = TriggerEvent::kAnyPieceMoved;

  // 特徴量のインデックスのリストを取得する
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
