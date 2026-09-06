// NNUE 評価関数で用いる入力特徴量とネットワーク構造の定義
// halfkp_512x2-16-32 に ThreatEffect (長/短利き数バケット 26,244 次元) を連結した型 (task#73 v2)
#ifndef CLASSIC_NNUE_HALFKP_THREAT_EFFECT_512X2_16_32_H_INCLUDED
#define CLASSIC_NNUE_HALFKP_THREAT_EFFECT_512X2_16_32_H_INCLUDED

#include "../features/feature_set.h"
#include "../features/half_kp.h"
#include "../features/threat_effect.h"

#include "../layers/input_slice.h"
#include "../layers/affine_transform.h"
#include "../layers/affine_transform_sparse_input.h"
#include "../layers/affine_transform_sparse_input_i16.h"
#include "../layers/clipped_relu.h"

namespace YaneuraOu {
namespace Eval::NNUE {

// 評価関数で用いる入力特徴量 (Tail=HalfKP がオフセット 0、Head=ThreatEffect が +125,388 = bullet と一致)
using RawFeatures = Features::FeatureSet<
    Features::ThreatEffect,
    Features::HalfKP<Features::Side::kFriend>>;

// 変換後の入力特徴量の次元数
constexpr IndexType kTransformedFeatureDimensions = 512;

#ifndef NNUE_L1_SCALE_BITS
#define NNUE_L1_SCALE_BITS 6
#endif

namespace Layers {

using InputLayer = InputSlice<kTransformedFeatureDimensions * 2>;
#if defined(NNUE_L1_INT16)
using HiddenLayer1 = ClippedReLU<AffineTransformSparseInputI16<InputLayer, 16, NNUE_L1_SCALE_BITS>>;
#else
using HiddenLayer1 = ClippedReLU<AffineTransformSparseInput<InputLayer, 16, NNUE_L1_SCALE_BITS>>;
#endif
using HiddenLayer2 = ClippedReLU<AffineTransform<HiddenLayer1, 32>>;
using OutputLayer = AffineTransform<HiddenLayer2, 1>;

}  // namespace Layers

using Network = Layers::OutputLayer;

} // namespace Eval::NNUE
} // namespace YaneuraOu

#endif // CLASSIC_NNUE_HALFKP_THREAT_EFFECT_512X2_16_32_H_INCLUDED
