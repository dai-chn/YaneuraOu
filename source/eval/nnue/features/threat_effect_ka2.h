#ifndef CLASSIC_NNUE_FEATURES_THREAT_EFFECT_KA2_H_INCLUDED
#define CLASSIC_NNUE_FEATURES_THREAT_EFFECT_KA2_H_INCLUDED

// 特徴量 ThreatEffectKa2: ThreatEffect (threat_effect.h) の HalfKA2 ペア用バリアント (task#73 王者移植)
//
// index 計算・次元・trigger・差分更新は ThreatEffect と完全同一。違いはハッシュだけ。
// bullet (BulletOu) 側は composite = HalfKA2_hash ^ "TEF2"(0x54454632)
//   = 0x5f234cb8 ^ 0x54454632 = 0x0B660A8A
// をヘッダに書く。YO の FeatureSet<ThreatEffectKa2, HalfKA2> の合成式
//   composite = ThreatEffectKa2::kHashValue ^ (KA2 << 1) ^ (KA2 >> 31)
// がその値になるよう逆算した定数 (KA2 = 0x5f234cb8, KA2<<1 = 0xBE469970, KA2>>31 = 0):
//   0x0B660A8A ^ 0xBE469970 = 0xB52093FA
// (threat_ka2.h と同じ手順)

#include "../../../config.h"

#if defined(EVAL_NNUE)

#include "threat_effect.h"

namespace YaneuraOu {
namespace Eval::NNUE::Features {

// ThreatEffect の HalfKA2 ペア用バリアント (ハッシュのみ変更、実装は継承)
class ThreatEffectKa2 : public ThreatEffect {
 public:
  // 特徴量名
  static constexpr const char* kName = "ThreatEffectKa2(LongShort)";

  // 評価関数ファイルに埋め込むハッシュ値 (導出はファイル冒頭コメント)
  static constexpr std::uint32_t kHashValue = 0xB52093FAu;
};

} // namespace Eval::NNUE::Features
} // namespace YaneuraOu

#endif  // defined(EVAL_NNUE)

#endif
