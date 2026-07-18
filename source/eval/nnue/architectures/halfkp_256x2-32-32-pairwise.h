// HalfKP 256x2-32-32 + pairwise multiplication (exp013 arm3)
// FT 直後活性化後に視点内 pairwise mult。FT 出力は kHalfDimensions (256, ×2 でない)。
// USE_ELEMENT_WISE_MULTIPLY を有効化する -DNNUE_FT_PAIRWISE がコンパイルフラグで必要。
#ifndef CLASSIC_NNUE_HALFKP_256X2_32_32_PAIRWISE_H_INCLUDED
#define CLASSIC_NNUE_HALFKP_256X2_32_32_PAIRWISE_H_INCLUDED

#if !defined(NNUE_FT_PAIRWISE)
#error "EVAL_NNUE_HALFKP256_PAIRWISE requires -DNNUE_FT_PAIRWISE (set both in EXTRA_CPPFLAGS)"
#endif

#include "../features/feature_set.h"
#include "../features/half_kp.h"

#include "../layers/input_slice.h"
#include "../layers/affine_transform.h"
#include "../layers/affine_transform_sparse_input.h"
#include "../layers/clipped_relu.h"

namespace YaneuraOu {
namespace Eval::NNUE {

using RawFeatures = Features::FeatureSet<
    Features::HalfKP<Features::Side::kFriend>>;

constexpr IndexType kTransformedFeatureDimensions = 256;

namespace Layers {

// pairwise で FT 出力が kHalfDimensions (256) に半減するため InputSlice は ×2 でなく ×1
using InputLayer = InputSlice<kTransformedFeatureDimensions>;
using HiddenLayer1 = ClippedReLU<AffineTransformSparseInput<InputLayer, 32>>;
using HiddenLayer2 = ClippedReLU<AffineTransform<HiddenLayer1, 32>>;
using OutputLayer = AffineTransform<HiddenLayer2, 1>;

}  // namespace Layers

using Network = Layers::OutputLayer;

}  // namespace Eval::NNUE
} // namespace YaneuraOu

#endif // CLASSIC_NNUE_HALFKP_256X2_32_32_PAIRWISE_H_INCLUDED
