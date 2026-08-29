// Definition of input features and network structure used in NNUE evaluation function
// NNUE評価関数で用いる入力特徴量とネットワーク構造の定義
//
// 我々の本番アーキ halfkp_512x2-16-32 の FT 幅だけを 768 にしたもの
// (学習側 crate は experiments/004e-halfkp-768x2-16-32-factorised、004d との差は
//  `const L1: usize` の 1 行のみ)。
//
// ★これが無かったせいで、幅 768 の判定を **proxy 指標だけ**で行い「NO-GO」と結論していた。
//   AobaNNUE の著者は実対局で 768 が 512/1024 に同一学習時間で勝つと報告しており
//   (report/44 §8.2)、proxy 判定を実測でやり直すためにこのヘッダを追加した。
#ifndef CLASSIC_NNUE_HALFKP_768X2_16_32_H_INCLUDED
#define CLASSIC_NNUE_HALFKP_768X2_16_32_H_INCLUDED

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
constexpr IndexType kTransformedFeatureDimensions = 768;

// L1 の重みスケールビット (6=QB64 従来 / 7=QB128, report/51 §7.7)。-DNNUE_L1_SCALE_BITS=7 で切替
#ifndef NNUE_L1_SCALE_BITS
#define NNUE_L1_SCALE_BITS 6
#endif

namespace Layers {

// Define network structure
// ネットワーク構造の定義 (512x2-16-32 と L1/L2 は同一、FT 幅だけ 512→768)
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

#endif // #ifndef CLASSIC_NNUE_HALFKP_768X2_16_32_H_INCLUDED
