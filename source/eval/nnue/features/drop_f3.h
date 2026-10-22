// NNUE 入力特徴量 DropF3: 安全な打ち当たり (docs/superpowers/specs/2026-10-23-drop-f3-feature-design.md)
//
// 視点 c の accumulator に、色 c の持ち駒 p を升 s に打つと相手の駒 (クラス vc) に当たる関係の行を入れる:
//   - 当たる相手が玉なら (p, s, 玉) を立てる (s に相手の利きがあっても立てる = 王手の打ち、F1)
//   - 当たる相手が歩なら立てない
//   - それ以外は s に相手の利きが無い (打った駒がただで取られない) ときだけ立てる (F3)
//   s は空き升で、歩・香は最終段、桂は最終 2 段、歩は二歩を除く (打ち歩詰め・王手回避は見ない)
//   index = p (7: 歩香桂銀金角飛) * 729 + s_n (81、後手視点は Inv) * 9 + vc (9: 香桂銀金類角馬飛龍玉)
//   次元 7 * 81 * 9 = 5,103。(p, s) ごとに vc 昇順で重複なし。
// BulletOu 側 (ShogiHalfKa2ThreatDropF3) と同一 index 仕様。配置は FeatureSet<DropF3, ThreatKa2, HalfKA2> = [KA2][Threat][DropF3]。
//
// ★v0 (2026-10-23): 毎手全列挙 (kAnyPieceMoved、独立 slot)。parity と固定ノード判定用。差分化 (kNone、前局面の集合を
//   StateInfo に保存して対称差分) は固定ノードで効いてから (spec §4)。

#ifndef CLASSIC_NNUE_FEATURES_DROP_F3_H
#define CLASSIC_NNUE_FEATURES_DROP_F3_H

#include "../../../config.h"

#if defined(EVAL_NNUE)

#include "../../../evaluate.h"
#include "features_common.h"

namespace YaneuraOu {
namespace Eval::NNUE::Features {

// 特徴量 DropF3: 安全な打ち当たり (持ち駒, 打つ升, 当たる駒クラス)
class DropF3 {
 public:
  // 特徴量名
  static constexpr const char* kName = "DropF3(SafeDropAttack)";

  // 評価関数ファイルに埋め込むハッシュ値 (SFNN では照合されない。任意の定数)
  static constexpr std::uint32_t kHashValue = 0xD3F30523u;

  // 特徴量の次元数 (7 p × 81 s × 9 vc)
  static constexpr IndexType kDimensions = 7 * 81 * 9;

  // 同時にアクティブになりうる最大特徴数 (実測の最大は両視点合計 70; 超えたら切り捨て)
  static constexpr IndexType kMaxActiveDimensions = 192;

  // v0: ナイーブ全再構築 (毎手 reset)
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
