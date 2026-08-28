// int16 重み版の疎入力アフィン変換層 (task#68 / report/51 §7.7)
//
// 背景: L1 (1024→16) の重みは中央値 |w| 0.01〜0.03 と小さく、int8 QB=64 の刻み (1/64) では
// 3〜4 割が 0 に丸められて fp32 比で −66cp/std 70 のズレを生んでいた。QB=128 で大半は回収
// できるが、lite のように max|w|≈2.0 で小さい重みも多いネットは int8 では刻みと範囲を両立
// できない。この層は重みを int16 で持ち、QB=256 (WeightScaleBits=8) で刻み 1/256・範囲 ±128
// を確保する。密層は実行時間の 2% なので、演算が 2 倍になっても全体では ~1%。
//
// 演算: 活性 (uint8) を 2 個ずつ int16 に広げて 32bit に詰め、`_mm256_madd_epi16` で
// (a0*w0 + a1*w1) を int32 に積む。非零 32bit ブロック (活性 4 個) の抽出は int8 版と同じ find_nnz。
// 重みは「ブロック (入力 4 個) → 出力 8 個のレジスタ → 対 (入力 2 個)」の順に並べ替えて格納する。
//
// ★シフト上限: 後段 ClippedReLU は int32 を int16 に飽和 pack してからシフトするので、
//   127 << WeightScaleBits が 32767 を超えない WeightScaleBits ≤ 8 (QB ≤ 256) まで。
// ★ハッシュは int8 版と同一 (ファイル形式の区別はできない)。nn.bin 側の l1w が int16 で
//   書かれていることをディレクトリ名等で管理すること (tools/nnue_requant.py --l1-int16)。

#ifndef NNUE_LAYERS_AFFINE_TRANSFORM_SPARSE_INPUT_I16_H_INCLUDED
#define NNUE_LAYERS_AFFINE_TRANSFORM_SPARSE_INPUT_I16_H_INCLUDED

#include "../../../config.h"

#if defined(EVAL_NNUE)

#include "../nnue_common.h"
#include "affine_transform_sparse_input.h"   // find_nnz

namespace YaneuraOu {
namespace Eval::NNUE::Layers {

template <typename PreviousLayer, IndexType OutputDimensions, int WeightScaleBits = 8>
class AffineTransformSparseInputI16 {
   public:
	using InputType  = typename PreviousLayer::OutputType;
	using OutputType = std::int32_t;
	static_assert(std::is_same<InputType, std::uint8_t>::value, "");
	static_assert(WeightScaleBits <= 8, "ClippedReLU の int16 飽和 pack のため 127<<bits <= 32767 が必要");

	static constexpr IndexType kInputDimensions       = PreviousLayer::kOutputDimensions;
	static constexpr IndexType kOutputDimensions      = OutputDimensions;
	static constexpr int kWeightScaleBits             = WeightScaleBits;
	static constexpr IndexType kPaddedInputDimensions = CeilToMultiple<IndexType>(kInputDimensions, kMaxSimdWidth);

	static constexpr std::size_t kSelfBufferSize =
	    CeilToMultiple(kOutputDimensions * sizeof(OutputType), kCacheLineSize);
	static constexpr std::size_t kBufferSize = PreviousLayer::kBufferSize + kSelfBufferSize;

	// 非零判定は 32bit ブロック (活性 4 個) 単位
	static constexpr IndexType kChunkSize = 4;

	// int8 版と同じハッシュ (レイアウトの違いはハッシュに現れない)
	static constexpr std::uint32_t GetHashValue() {
		std::uint32_t hash_value = 0xCC03DAE4u;
		hash_value += kOutputDimensions;
		hash_value ^= PreviousLayer::GetHashValue() >> 1;
		hash_value ^= PreviousLayer::GetHashValue() << 31;
		return hash_value;
	}
	static constexpr std::uint32_t GetHashValue(std::uint32_t prevHash) {
		std::uint32_t hash_value = 0xCC03DAE4u;
		hash_value += kOutputDimensions;
		hash_value ^= prevHash >> 1;
		hash_value ^= prevHash << 31;
		return hash_value;
	}

	static std::string GetStructureString() {
		return "AffineTransformSparseInputI16[" + std::to_string(kOutputDimensions) + "<-" + std::to_string(kInputDimensions) + "](" +
		       PreviousLayer::GetStructureString() + ")";
	}

#if defined(USE_AVX2)
	static constexpr bool kUseSimd = (kOutputDimensions % 8 == 0);
#else
	static constexpr bool kUseSimd = false;
#endif

	// (出力 o, 入力 in) の重みの格納位置
	//   SIMD: ブロック c=in/4 → レジスタ k=o/8 → 対 p=(in%4)/2 → 16 個の int16 [lane=o%8][in%2]
	//   非 SIMD: 行優先 (o * kPaddedInputDimensions + in)
	static constexpr IndexType GetWeightIndex(IndexType o, IndexType in) {
		if constexpr (kUseSimd) {
			constexpr IndexType kNumRegs = kOutputDimensions / 8;
			const IndexType c = in / 4, p = (in % 4) / 2, k = o / 8;
			return ((c * kNumRegs + k) * 2 + p) * 16 + (o % 8) * 2 + (in % 2);
		} else {
			return o * kPaddedInputDimensions + in;
		}
	}

	Tools::Result ReadParameters(std::istream& stream) {
		Tools::Result result = previous_layer_.ReadParameters(stream);
		if (result.is_not_ok()) return result;
		for (std::size_t i = 0; i < kOutputDimensions; ++i)
			biases_[i] = read_little_endian<BiasType>(stream);
		// ファイルは行優先 (o, in) の int16
		for (IndexType o = 0; o < kOutputDimensions; ++o)
			for (IndexType in = 0; in < kPaddedInputDimensions; ++in)
				weights_[GetWeightIndex(o, in)] = read_little_endian<WeightType>(stream);
		return !stream.fail() ? Tools::ResultCode::Ok : Tools::ResultCode::FileReadError;
	}

	bool WriteParameters(std::ostream& stream) const {
		if (!previous_layer_.WriteParameters(stream))
			return false;
		stream.write(reinterpret_cast<const char*>(biases_), kOutputDimensions * sizeof(BiasType));
		for (IndexType o = 0; o < kOutputDimensions; ++o)
			for (IndexType in = 0; in < kPaddedInputDimensions; ++in) {
				const WeightType w = weights_[GetWeightIndex(o, in)];
				stream.write(reinterpret_cast<const char*>(&w), sizeof(WeightType));
			}
		return !stream.fail();
	}

	const OutputType* Propagate(const TransformedFeatureType* transformed_features, char* buffer) const {
		const auto input  = previous_layer_.Propagate(transformed_features, buffer + kSelfBufferSize);
		const auto output = reinterpret_cast<OutputType*>(buffer);

#if defined(USE_AVX2)
		if constexpr (kUseSimd)
		{
			constexpr IndexType kNumChunks = CeilToMultiple<IndexType>(kInputDimensions, 8) / kChunkSize;
			constexpr IndexType kNumRegs   = kOutputDimensions / 8;
			std::uint16_t       nnz[kNumChunks];
			IndexType           count;

			const auto input32 = reinterpret_cast<const std::int32_t*>(input);
			find_nnz<kNumChunks>(input32, nnz, count);

			const __m256i* biasvec = reinterpret_cast<const __m256i*>(biases_);
			__m256i        acc[kNumRegs];
			for (IndexType k = 0; k < kNumRegs; ++k)
				acc[k] = biasvec[k];

			for (IndexType j = 0; j < count; ++j)
			{
				const auto          i = nnz[j];
				const std::uint32_t v = static_cast<std::uint32_t>(input32[i]);
				// 活性 (a0,a1) / (a2,a3) を int16 ペアとして 32bit に詰めて放送
				const __m256i lo = _mm256_set1_epi32(int((v & 0xFFu) | ((v >> 8) & 0xFFu) << 16));
				const __m256i hi = _mm256_set1_epi32(int(((v >> 16) & 0xFFu) | ((v >> 24) & 0xFFu) << 16));
				const auto col = reinterpret_cast<const __m256i*>(&weights_[i * kOutputDimensions * kChunkSize]);
				for (IndexType k = 0; k < kNumRegs; ++k) {
					acc[k] = _mm256_add_epi32(acc[k], _mm256_madd_epi16(lo, col[k * 2 + 0]));
					acc[k] = _mm256_add_epi32(acc[k], _mm256_madd_epi16(hi, col[k * 2 + 1]));
				}
			}

			__m256i* outptr = reinterpret_cast<__m256i*>(output);
			for (IndexType k = 0; k < kNumRegs; ++k)
				outptr[k] = acc[k];
		}
		else
#endif
		{
			for (IndexType o = 0; o < kOutputDimensions; ++o) {
				std::int32_t sum = biases_[o];
				for (IndexType in = 0; in < kInputDimensions; ++in)
					sum += std::int32_t(weights_[GetWeightIndex(o, in)]) * std::int32_t(input[in]);
				output[o] = sum;
			}
		}
		return output;
	}

   private:
	using BiasType   = OutputType;
	using WeightType = std::int16_t;

	PreviousLayer previous_layer_;

	alignas(kCacheLineSize) BiasType biases_[kOutputDimensions];
	alignas(kCacheLineSize) WeightType weights_[kOutputDimensions * kPaddedInputDimensions];
};

} // namespace Eval::NNUE::Layers
} // namespace YaneuraOu

#endif  // defined(EVAL_NNUE)

#endif  // NNUE_LAYERS_AFFINE_TRANSFORM_SPARSE_INPUT_I16_H_INCLUDED
