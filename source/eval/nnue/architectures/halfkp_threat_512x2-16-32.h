// NNUE 評価関数で用いる入力特徴量とネットワーク構造の定義
// halfkp_512x2-16-32 に Threat (利き当たり) 特徴を連結した型 (task#52 Phase-1 / report/51)
#ifndef CLASSIC_NNUE_HALFKP_THREAT_512X2_16_32_H_INCLUDED
#define CLASSIC_NNUE_HALFKP_THREAT_512X2_16_32_H_INCLUDED

#include "../features/feature_set.h"
#include "../features/half_kp.h"
#include "../features/threat.h"

#include "../layers/input_slice.h"
#include "../layers/affine_transform.h"
#include "../layers/affine_transform_sparse_input.h"
#include "../layers/affine_transform_sparse_input_i16.h"
#include "../layers/clipped_relu.h"

namespace YaneuraOu {
namespace Eval::NNUE {

// 評価関数で用いる入力特徴量
// ★テンプレート引数の順序に注意: FeatureSet の CollectActiveIndices は
//   **Tail が先頭オフセット** (Head の index に Tail::kDimensions が足される)。
//   bullet 側は KP が 0..125388、Threat が 125388.. なので、
//   FeatureSet<Threat, HalfKP> と書くことで KP=Tail (オフセット 0)、
//   Threat=Head (+125,388) になり bullet と一致する。
using RawFeatures = Features::FeatureSet<
    Features::Threat,
    Features::HalfKP<Features::Side::kFriend>>;

// 変換後の入力特徴量の次元数
constexpr IndexType kTransformedFeatureDimensions = 512;

// L1 の重みスケールビット (6=QB64 従来 / 7=QB128, report/51 §7.7)。ビルド時 -DNNUE_L1_SCALE_BITS=7 で切替
#ifndef NNUE_L1_SCALE_BITS
#define NNUE_L1_SCALE_BITS 6
#endif

namespace Layers {

// ネットワーク構造の定義 (halfkp_512x2-16-32 と同一)
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

#endif // CLASSIC_NNUE_HALFKP_THREAT_512X2_16_32_H_INCLUDED
