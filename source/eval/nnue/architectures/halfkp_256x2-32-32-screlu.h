// HalfKP 256x2-32-32 + SCReLU FT activation (exp013 arm1)
// FT 直後の活性化を clamp(acc,0,127)^2 >> 7 に差し替える。レイヤ構成は無印と同一。
#ifndef CLASSIC_NNUE_HALFKP_256X2_32_32_SCRELU_H_INCLUDED
#define CLASSIC_NNUE_HALFKP_256X2_32_32_SCRELU_H_INCLUDED

// nnue_feature_transformer.h (Transform / GetHashValue) がこのフラグで分岐する
#define NNUE_FT_SCRELU

#include "halfkp_256x2-32-32.h"

#endif // CLASSIC_NNUE_HALFKP_256X2_32_32_SCRELU_H_INCLUDED
