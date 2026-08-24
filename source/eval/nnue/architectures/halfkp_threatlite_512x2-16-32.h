// NNUE 評価関数で用いる入力特徴量とネットワーク構造の定義
// halfkp_512x2-16-32 に ThreatLite (from-drop 縮約 26,244 次元) を連結した型 (task#59)
#ifndef CLASSIC_NNUE_HALFKP_THREATLITE_512X2_16_32_H_INCLUDED
#define CLASSIC_NNUE_HALFKP_THREATLITE_512X2_16_32_H_INCLUDED

#include "../features/feature_set.h"
#include "../features/half_kp.h"
#include "../features/threat_lite.h"

#include "../layers/input_slice.h"
#include "../layers/affine_transform.h"
#include "../layers/affine_transform_sparse_input.h"
#include "../layers/clipped_relu.h"

namespace YaneuraOu {
namespace Eval::NNUE {

// 評価関数で用いる入力特徴量
// ★テンプレート引数の順序に注意: FeatureSet の CollectActiveIndices は
//   **Tail が先頭オフセット** (Head の index に Tail::kDimensions が足される)。
//   bullet 側は KP が 0..125388、ThreatLite が 125388.. なので、
//   FeatureSet<ThreatLite, HalfKP> と書くことで KP=Tail (オフセット 0)、
//   ThreatLite=Head (+125,388) になり bullet と一致する。
using RawFeatures = Features::FeatureSet<
    Features::ThreatLite,
    Features::HalfKP<Features::Side::kFriend>>;

// 変換後の入力特徴量の次元数
constexpr IndexType kTransformedFeatureDimensions = 512;

namespace Layers {

// ネットワーク構造の定義 (halfkp_512x2-16-32 と同一)
using InputLayer = InputSlice<kTransformedFeatureDimensions * 2>;
using HiddenLayer1 = ClippedReLU<AffineTransformSparseInput<InputLayer, 16>>;
using HiddenLayer2 = ClippedReLU<AffineTransform<HiddenLayer1, 32>>;
using OutputLayer = AffineTransform<HiddenLayer2, 1>;

}  // namespace Layers

using Network = Layers::OutputLayer;

} // namespace Eval::NNUE
} // namespace YaneuraOu

#endif // CLASSIC_NNUE_HALFKP_THREATLITE_512X2_16_32_H_INCLUDED
