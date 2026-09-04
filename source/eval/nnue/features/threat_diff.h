// 駒レベル threat 差分の共有インタフェース (task#59 ②)
//
// Threat (full) の差分更新 (task#37) が計算する「直前手で増減した (attacker, victim) 対」は
// 視点にも index 空間にも依存しない駒レベルの情報なので、ThreatLite (from-drop) も
// そのまま流用できる (lite は count 意味論なので、対の ±1 = lite index の ±1 push が
// 参照カウント無しで正しい)。実装と thread_local キャッシュは threat.cpp に置く。

#ifndef CLASSIC_NNUE_FEATURES_THREAT_DIFF_H
#define CLASSIC_NNUE_FEATURES_THREAT_DIFF_H

#include "../../../config.h"

#if defined(EVAL_NNUE) && defined(KEEP_LAST_MOVE)

#include <cstdint>

namespace YaneuraOu {

class Position;

namespace Eval::NNUE::Features {

// 駒レベルの (attacker, victim) 対。index への写像は各特徴 (Threat / ThreatLite) が行う
struct ThreatPiecePair {
    std::uint8_t attacker_pc;   // Piece
    std::uint8_t attacker_sq;
    std::uint8_t victim_pc;     // Piece
    std::uint8_t victim_sq;
};

// 直前手による駒レベル threat 差分 (視点非依存)。thread_local にキャッシュされ、
// 同一 (StateInfo, key, lastMove) での再呼び出し (BLACK→WHITE の 2 視点) は再計算しない。
struct ThreatPieceDiff {
    const void*   st = nullptr;
    std::uint64_t key = 0;
    std::uint32_t move = 0;
    int n_removed = 0, n_added = 0;
    ThreatPiecePair removed[128];
    ThreatPiecePair added[128];
};

// pos の直前手による差分を返す (キャッシュミス時は収集する)。実装は threat.cpp。
const ThreatPieceDiff& threat_piece_diff(const Position& pos);

} // namespace Eval::NNUE::Features
} // namespace YaneuraOu

#endif  // defined(EVAL_NNUE) && defined(KEEP_LAST_MOVE)

#endif
