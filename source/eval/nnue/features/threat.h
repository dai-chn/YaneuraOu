// NNUE 入力特徴量 Threat の定義 (task#52 Phase-1 / report/51)
//
// 「駒 A の実利きが駒 B の升に届いている」関係を特徴化する。
// bullet-shogi 側 `shogi_halfkp_threat.rs` (rshogi threat_spec 系) と**同一 index 仕様**。
//
// - 攻撃側/被弾側とも玉は除外。9 クラス (P/L/N/S/金類/B/R/馬/龍) × 敵味方
// - (攻撃クラス, from, to) は空盤幾何で圧縮 (pair ごとに from_offset + attack_order)
// - profile は Full (除外なし) 固定。次元 = 216,720
// - ★差分更新はしない: kRefreshTrigger = kAnyPieceMoved (毎手、既存機構が全再構築)。
//   固定ノード判定用のナイーブ実装。採用が決まったら差分化する (report/51 §4)。

#ifndef CLASSIC_NNUE_FEATURES_THREAT_H
#define CLASSIC_NNUE_FEATURES_THREAT_H

#include "../../../config.h"

#if defined(EVAL_NNUE)

#include "../../../evaluate.h"
#include "features_common.h"

namespace YaneuraOu {
namespace Eval::NNUE::Features {

// 特徴量 Threat: 駒の実利きが駒に当たっている関係
class Threat {
 public:
  // 特徴量名
  static constexpr const char* kName = "Threat(Full)";

  // 評価関数ファイルに埋め込むハッシュ値。
  // ★bullet 側 (exp004u main.rs) は composite = HalfKP_hash ^ 0x54485254 ^ profile_id を
  //   ヘッダに書く。YO の FeatureSet<Threat, HalfKP> の合成式
  //     composite = Threat::kHashValue ^ (KP << 1) ^ (KP >> 31)
  //   がその値になるよう逆算した定数 (KP = 0x5D69D5B8, profile = full = 0):
  //     0x092187EC ^ 0xBAD3AB70 ^ 0 = 0xB3F22C9C
  static constexpr std::uint32_t kHashValue = 0xB3F22C9Cu;

  // 特徴量の次元数 (full profile: 2 * 9 * 2 * 9 pair に幾何圧縮次元を掛けた総和)
  static constexpr IndexType kDimensions = 216720;

  // 同時にアクティブになりうる最大特徴数 (bullet 側と同じ安全上限。実測は ~38)
  static constexpr IndexType kMaxActiveDimensions = 320;

  // 差分計算の代わりに全計算を行うタイミング。
  // 既定は kNone = 常に差分計算 (task#37)。threat の index は玉位置に依存しないので
  // リフレッシュ不要、毎手 AppendChangedIndices で ~5.3 行/手だけ更新する。
  // (素朴全再構築は NPS 0.3988× と実測され不可 — report/51 §7.3)
  // 検証用に -DTHREAT_NAIVE_REBUILD で旧挙動 (毎手全再構築) に戻せる。
#if defined(THREAT_NAIVE_REBUILD) || !defined(KEEP_LAST_MOVE)
  // 差分更新は lastMove を要するため、KEEP_LAST_MOVE の無いビルドは自動で naive に落とす
  static constexpr TriggerEvent kRefreshTrigger = TriggerEvent::kAnyPieceMoved;
#else
  static constexpr TriggerEvent kRefreshTrigger = TriggerEvent::kNone;
#endif

  // 特徴量のうち、値が 1 であるインデックスのリストを取得する
  static void AppendActiveIndices(const Position& pos, Color perspective,
                                  IndexList* active);

  // kAnyPieceMoved は常に reset (= AppendActiveIndices 側) になるので呼ばれない
  static void AppendChangedIndices(const Position& pos, Color perspective,
                                   IndexList* removed, IndexList* added);

#if defined(THREAT_ATTACKER_MAJOR)
  // attacker-major 並び替え (task#59 / report/51 §7.6.2)。
  // index は (as, ac) ブロック内 (from, ord) スラブ × 18 (ds, dc) の attacker-major 配置になる。
  // nn.bin は標準 (pair-major) 配置で学習されているので、FT ロード時に本関数で行を置換する。
  // weights: FT 重み配列 (行 = half_dims 要素)、row_offset: 合成特徴量内の threat 先頭行。
  static void PermuteRows(std::int16_t* weights, std::size_t half_dims, std::size_t row_offset);
#endif
};

} // namespace Eval::NNUE::Features
} // namespace YaneuraOu

#endif  // defined(EVAL_NNUE)

#endif
