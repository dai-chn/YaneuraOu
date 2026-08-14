// Definition of input features and network structure used in NNUE evaluation function
// NNUE評価関数で用いる入力特徴量とネットワーク構造の定義
//
// AobaNNUE 1.1 (山下宏氏) のアーキテクチャ。同氏の付属文書によれば
// 1024_8_96 / 512_8_64 / 2048_32_32 と比較して、同一学習時間ではこの
// 768x2-16-64 が最良だったとのこと。
// 我々の 512x2-16-32 と L1 幅 (16) が一致しており、FT 幅と L2 幅だけが違う。
// 「同一探索部で評価関数だけ差し替える」比較のために追加した。
#ifndef CLASSIC_NNUE_HALFKP_768X2_16_64_H_INCLUDED
#define CLASSIC_NNUE_HALFKP_768X2_16_64_H_INCLUDED

#include "../features/feature_set.h"
#include "../features/half_kp.h"

#include "../layers/input_slice.h"
#include "../layers/affine_transform.h"
#include "../layers/affine_transform_sparse_input.h"
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

namespace Layers {

// Define network structure
// ネットワーク構造の定義
// AffineTransformSparseInput は AffineTransform と同じ GetHashValue() を返すので、
// 疎入力版を使っても他所で作られた nn.bin をそのまま読める。
using InputLayer = InputSlice<kTransformedFeatureDimensions * 2>;
using HiddenLayer1 = ClippedReLU<AffineTransformSparseInput<InputLayer, 16>>;
using HiddenLayer2 = ClippedReLU<AffineTransform<HiddenLayer1, 64>>;
using OutputLayer = AffineTransform<HiddenLayer2, 1>;

}  // namespace Layers

using Network = Layers::OutputLayer;

} // namespace Eval::NNUE
} // namespace YaneuraOu

#endif // #ifndef CLASSIC_NNUE_HALFKP_768X2_16_64_H_INCLUDED
