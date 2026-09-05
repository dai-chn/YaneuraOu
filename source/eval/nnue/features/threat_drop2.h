// NNUE 入力特徴量 ThreatDrop2 の定義 (task#73 / report/52 §18.8)
//
// ThreatLite (from-drop) からさらに **攻撃駒種 (attacker class) も落とした** 縮約版:
//   feature = (attacker_side, defender_side, defender_class, to_sq)
//   次元 = 2 * 2 * 9 * 81 = 2,916
// bullet-shogi 側 `shogi_halfkp_threat_drop2.rs` と**同一 index 仕様**。
//
// この粒度は「to の駒に side as の利きが何本当たっているか」に等しいので、将来は
// LONG_EFFECT_LIBRARY の利き数 (board_effect) から列挙なしで計算できる (税ゼロ化の本命)。
// ★本ファイルは Elo 保持率 A/B 用の判定実装: 列挙経路は threat_lite と同一のナイーブ全再構築
//   (kAnyPieceMoved)。固定ノード判定なので NPS は結果に影響しない。
//
// ★count 意味論: 同一 (as, ds, dc, to) への複数攻撃は同一 index を重複 push する。
//   攻撃側の玉は full/lite と同様に除外 (利き数で実装するときは玉利きの控除が要る)。

#ifndef CLASSIC_NNUE_FEATURES_THREAT_DROP2_H
#define CLASSIC_NNUE_FEATURES_THREAT_DROP2_H

#include "../../../config.h"

#if defined(EVAL_NNUE)

#include "../../../evaluate.h"
#include "features_common.h"

namespace YaneuraOu {
namespace Eval::NNUE::Features {

// 特徴量 ThreatDrop2: 駒に当たっている利きの (攻撃側, 被弾側, 被弾クラス, 升)
class ThreatDrop2 {
 public:
  // 特徴量名
  static constexpr const char* kName = "ThreatDrop2(ClassDrop)";

  // 評価関数ファイルに埋め込むハッシュ値。
  // bullet 側 (exp004y main.rs) のヘッダ hash = HalfKP_hash ^ "TDR2"
  //   = 0x5D69D5B8 ^ 0x54445232 = 0x092D878A
  // YO の FeatureSet<ThreatDrop2, HalfKP> の合成式
  //   composite = ThreatDrop2::kHashValue ^ (KP << 1) ^ (KP >> 31)
  // がその値になるよう逆算 (KP<<1 = 0xBAD3AB70, KP>>31 = 0):
  //   0x092D878A ^ 0xBAD3AB70 = 0xB3FE2CFA
  static constexpr std::uint32_t kHashValue = 0xB3FE2CFAu;

  // 特徴量の次元数 (2 as × 2 ds × 9 dc × 81 to)
  static constexpr IndexType kDimensions = 2916;

  // 同時にアクティブになりうる最大特徴数 (重複 emit 込み。full threat と同じ上限)
  static constexpr IndexType kMaxActiveDimensions = 320;

  // ナイーブ全再構築 (毎手 reset)。差分化/利き数実装は A/B 通過後
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
