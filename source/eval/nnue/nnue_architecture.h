// Input features and network structure used in NNUE evaluation function
// NNUE評価関数で用いる入力特徴量とネットワーク構造

#ifndef CLASSIC_NNUE_ARCHITECTURE_H_INCLUDED
#define CLASSIC_NNUE_ARCHITECTURE_H_INCLUDED

#include "../../config.h"

#if defined(EVAL_NNUE)

// Defines the network structure
// 入力特徴量とネットワーク構造が定義されたヘッダをincludeする

#if defined(NNUE_ARCHITECTURE_HEADER)

// 動的に生成されたファイルがある。
#include NNUE_ARCHITECTURE_HEADER

#elif defined(EVAL_NNUE_HALFKP256_SCRELU)

// 標準NNUE halfkp256 の FT 活性化を SCReLU にしたもの (exp013 arm1)
// ビルドは YANEURAOU_ENGINE_NNUE + EXTRA_CPPFLAGS=-DEVAL_NNUE_HALFKP256_SCRELU を要す
// (tools/build_yaneuraou.py の EXTRA_CPPFLAGS_MAP 参照)
#include "architectures/halfkp_256x2-32-32-screlu.h"

#elif defined(EVAL_NNUE_HALFKP256_PAIRWISE)

// 標準NNUE halfkp256 に pairwise multiplication を入れたもの (exp013 arm3)
#include "architectures/halfkp_256x2-32-32-pairwise.h"

#elif defined(EVAL_NNUE_HALFKP256)

// 標準NNUE型。NNUE評価関数のデフォルトは、halfKP256

#include "architectures/halfkp_256x2-32-32.h"

#elif defined(EVAL_NNUE_KP256)

// kp型
#include "architectures/kp_256x2-32-32.h"

#elif defined(EVAL_NNUE_HALFKPE9)

// halfkpe9型
#include "architectures/halfkpe9_256x2-32-32.h"

#elif defined(EVAL_NNUE_HALFKP512_SCRELU)

// halfkp_512x2-16-32 の FT 活性化を SCReLU にしたもの (report/43 A1、exp013 arm1 の 512 移植)
// ビルドは YANEURAOU_ENGINE_NNUE_HALFKP_512X2_16_32 + EXTRA_CPPFLAGS=-DEVAL_NNUE_HALFKP512_SCRELU
#include "architectures/halfkp_512x2-16-32-screlu.h"

#elif defined(YANEURAOU_ENGINE_NNUE_HALFKA_512X2_16_32)

// halfka_512x2-16-32型 (入力近代化: 玉を特徴に含める)
#include "architectures/halfka_512x2-16-32.h"

#elif defined(YANEURAOU_ENGINE_NNUE_HALFKP_512X2_16_32)

// halfkp_512x2-16-32型
#include "architectures/halfkp_512x2-16-32.h"

#elif defined(YANEURAOU_ENGINE_NNUE_HALFKP_THREAT_512X2_16_32)

// halfkp+threat_512x2-16-32型 (task#52 Phase-1: 利き当たり特徴の連結)
#include "architectures/halfkp_threat_512x2-16-32.h"

#elif defined(YANEURAOU_ENGINE_NNUE_HALFKP_THREATLITE_512X2_16_32)

// halfkp+threatlite_512x2-16-32型 (task#59: from-drop 縮約 26,244 次元)
#include "architectures/halfkp_threatlite_512x2-16-32.h"

#elif defined(YANEURAOU_ENGINE_NNUE_HALFKP_512X2_32_32)

// halfkp_512x2-32-32型 (L2=32: L2 がタダか検証用)
#include "architectures/halfkp_512x2-32-32.h"

#elif defined(YANEURAOU_ENGINE_NNUE_HALFKP_512X2_8_64)

// halfkp_512x2-8-64型 (Suisho10 のアーキ)
#include "architectures/halfkp_512x2-8-64.h"

#elif defined(YANEURAOU_ENGINE_NNUE_HALFKP_768X2_8_32)

// halfkp_768x2-8-32型
#include "architectures/halfkp_768x2-8-32.h"

#elif defined(YANEURAOU_ENGINE_NNUE_HALFKP_768X2_16_64)

// halfkp_768x2-16-64型 (AobaNNUE 1.1 のアーキ)
#include "architectures/halfkp_768x2-16-64.h"

#elif defined(YANEURAOU_ENGINE_NNUE_HALFKP_768X2_16_32)

// halfkp_768x2-16-32型 (本番 512x2-16-32 の FT 幅だけ拡げたもの、crate 004e)
#include "architectures/halfkp_768x2-16-32.h"

#elif defined(YANEURAOU_ENGINE_NNUE_HALFKP_1024X2_8_32)

// halfkp_1024x2-8-32型
#include "architectures/halfkp_1024x2-8-32.h"

#elif defined(YANEURAOU_ENGINE_NNUE_HALFKP_1024X2_8_64)

// halfkp_1024x2-8-64型
#include "architectures/halfkp_1024x2-8-64.h"

#elif defined(YANEURAOU_ENGINE_NNUE_SFNNwoP1536)

// SFNN without Psqt 1536型
#include "architectures/sfnnwop-1536.h"

#elif defined(EVAL_NNUE_HALFKP_VM_256X2_32_32)

// halfkpvm_256x2-32-32型
#include "architectures/halfkpvm_256x2-32-32.h"

#else

// どれも定義されていなかったので標準NNUE型にしておく。
#include "architectures/halfkp_256x2-32-32.h"

#endif

namespace YaneuraOu {
namespace Eval::NNUE {

	static_assert(kTransformedFeatureDimensions % kMaxSimdWidth == 0, "");
	static_assert(Network::kOutputDimensions == 1, "");
	static_assert(std::is_same<Network::OutputType, std::int32_t>::value, "");

	// Trigger for full calculation instead of difference calculation
	// 差分計算の代わりに全計算を行うタイミングのリスト
	constexpr auto kRefreshTriggers = RawFeatures::kRefreshTriggers;

} // namespace Eval::NNUE
} // namespace YaneuraOu

#endif  // defined(EVAL_NNUE)

#endif // #ifndef NNUE_ARCHITECTURE_H_INCLUDED
