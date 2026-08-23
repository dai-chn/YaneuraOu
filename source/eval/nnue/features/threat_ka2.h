#ifndef CLASSIC_NNUE_FEATURES_THREAT_KA2_H_INCLUDED
#define CLASSIC_NNUE_FEATURES_THREAT_KA2_H_INCLUDED

// 特徴量 ThreatKa2: Threat (threat.h) の HalfKA2 ペア用バリアント (task#54)
//
// index 計算・次元・trigger は Threat と完全同一。違いはハッシュだけ。
// bullet (BulletOu) 側は composite = HalfKA2_hash ^ "THRT"(0x54485254) ^ profile_id(full=0)
//   = 0x5f234cb8 ^ 0x54485254 = 0x0b6b1eec
// をヘッダに書く。YO の FeatureSet<ThreatKa2, HalfKA2> の合成式
//   composite = ThreatKa2::kHashValue ^ (KA2 << 1) ^ (KA2 >> 31)
// がその値になるよう逆算した定数 (KA2 = 0x5f234cb8):
//   0x0b6b1eec ^ 0xBE469970 = 0xB52D879C
// (HalfKP 版 threat.h の 0xB3F22C9C と同じ手順。既知値で検算済 2026-08-22)

#include "../../../config.h"

#if defined(EVAL_NNUE)

#include "threat.h"

namespace YaneuraOu {
namespace Eval::NNUE::Features {

// Threat の HalfKA2 ペア用バリアント (ハッシュのみ変更、実装は継承)
class ThreatKa2 : public Threat {
 public:
  // 特徴量名
  static constexpr const char* kName = "ThreatKa2(Full)";

  // 評価関数ファイルに埋め込むハッシュ値 (導出はファイル冒頭コメント)
  static constexpr std::uint32_t kHashValue = 0xB52D879Cu;
};

} // namespace Eval::NNUE::Features
} // namespace YaneuraOu

#endif  // defined(EVAL_NNUE)

#endif
