// Definition of input features and network structure used in NNUE evaluation function
// NNUE評価関数で用いる入力特徴量とネットワーク構造の定義
#ifndef CLASSIC_NNUE_HALFKP_512X2_16_32_H_INCLUDED
#define CLASSIC_NNUE_HALFKP_512X2_16_32_H_INCLUDED

#include "../features/feature_set.h"
#include "../features/half_kp.h"

#include "../layers/input_slice.h"
#include "../layers/affine_transform.h"
#include "../layers/affine_transform_sparse_input.h"
#include "../layers/affine_transform_sparse_input_i16.h"
#include "../layers/clipped_relu.h"

namespace YaneuraOu {
namespace Eval::NNUE {

// Input features used in evaluation function
// 評価関数で用いる入力特徴量
using RawFeatures = Features::FeatureSet<
    Features::HalfKP<Features::Side::kFriend>>;

// Number of input feature dimensions after conversion
// 変換後の入力特徴量の次元数
constexpr IndexType kTransformedFeatureDimensions = 512;

// L1 の重みスケールビット (6=QB64 従来 / 7=QB128, report/51 §7.7)。-DNNUE_L1_SCALE_BITS=7 で切替
#ifndef NNUE_L1_SCALE_BITS
#define NNUE_L1_SCALE_BITS 6
#endif

namespace Layers {

// Define network structure
// ネットワーク構造の定義
using InputLayer = InputSlice<kTransformedFeatureDimensions * 2>;
#if defined(NNUE_L1_INT16)
// L1 を int16 重みで持つ (task#68)。-DNNUE_L1_INT16 -DNNUE_L1_SCALE_BITS=8 (QB=256) で使う
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

#endif // #ifndef NNUE_HALFKP_512X2_16_32_H_INCLUDED
