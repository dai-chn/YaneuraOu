// NNUE 入力特徴量 ThreatDrop2 (task#73 / report/52 §18.8)
//
// ★bullet-shogi `shogi_halfkp_threat_drop2.rs` と同一 index 仕様。
//   drop2_index = (as*18 + ds*9 + dc) * 81 + to_n
//   列挙経路 (occupancy/effects_from/クラス写像/視点変換) は threat_lite.cpp と同一で、
//   攻撃クラスは index に使わず、玉/無効の除外条件だけに使う。
//
// ★count 意味論: 同一 (as, ds, dc, to) への複数攻撃は同一 index を重複 push する。

#include "../../../config.h"

#if defined(EVAL_NNUE)

#include "threat_drop2.h"
#include "index_list.h"
#include "../../../position.h"
#include "../../../bitboard.h"

#include <cstdint>
#include <cstdio>

namespace YaneuraOu {
namespace Eval::NNUE::Features {

namespace {

// ThreatClass (threat.cpp / threat_lite.cpp と同一の 9 クラス。玉は除外)
// ★順序を変えると bullet と不一致になる — 変更禁止。
inline int drop2_class_of(PieceType pt) {
    switch (pt) {
        case PAWN:       return 0;   // Pawn
        case LANCE:      return 1;   // Lance
        case KNIGHT:     return 2;   // Knight
        case SILVER:     return 3;   // Silver
        case GOLD:
        case PRO_PAWN:
        case PRO_LANCE:
        case PRO_KNIGHT:
        case PRO_SILVER: return 4;   // GoldLike
        case BISHOP:     return 5;   // Bishop
        case ROOK:       return 6;   // Rook
        case HORSE:      return 7;   // Horse
        case DRAGON:     return 8;   // Dragon
        default:         return -1;  // KING ほか
    }
}

// (attacker_color, victim) -> drop2 特徴 index (視点変換込み、HM ミラー無し)
inline uint32_t drop2_index(Color perspective, Color attacker_color,
                            Color target_color, int dc, Square to) {
    const Square to_n = (perspective == BLACK) ? to : Inv(to);
    const int as = (attacker_color != perspective) ? 1 : 0;
    const int ds = (target_color != perspective) ? 1 : 0;
    return uint32_t(as * 18 + ds * 9 + dc) * 81 + uint32_t(to_n);
}

}  // namespace

// 特徴量のインデックスのリストを取得する (★重複 push あり = count 意味論)
void ThreatDrop2::AppendActiveIndices(const Position& pos, Color perspective, IndexList* active) {
    const Bitboard occ = pos.pieces();

    for (int sq_raw = 0; sq_raw < 81; ++sq_raw) {
        const Square from = Square(sq_raw);
        const Piece pc = pos.piece_on(from);
        if (pc == NO_PIECE)
            continue;
        const PieceType pt = type_of(pc);
        if (drop2_class_of(pt) < 0)
            continue;   // 玉 (攻撃側から除外 — index には使わないが除外条件は lite と同一)
        const Color attacker_color = color_of(pc);

        // 実盤面の利き (遮断込み) — threat.cpp / threat_lite.cpp と同一経路
        Bitboard att = effects_from(pc, from, occ) & occ;
        while (att) {
            const Square to = att.pop();
            const Piece tpc = pos.piece_on(to);
            const int dc = drop2_class_of(type_of(tpc));
            if (dc < 0)
                continue;   // 玉は被弾側からも除外
            const Color target_color = color_of(tpc);

            const uint32_t idx = drop2_index(perspective, attacker_color, target_color, dc, to);
            ASSERT_LV3(idx < kDimensions);
            active->push_back(IndexType(idx));
#if defined(THREAT_DROP2_DUMP)
            // bullet 側との index 多重集合照合用 (研究ビルド限定、stderr)
            std::fprintf(stderr, "D2 %c %u\n", perspective == BLACK ? 'B' : 'W', idx);
#endif
        }
    }
#if defined(THREAT_DROP2_DUMP)
    std::fprintf(stderr, "D2 %c END\n", perspective == BLACK ? 'B' : 'W');
#endif
}

// kAnyPieceMoved は常に reset (= AppendActiveIndices) になるので呼ばれない
void ThreatDrop2::AppendChangedIndices(const Position& pos, Color perspective,
                                       IndexList* removed, IndexList* added) {
    (void)pos; (void)perspective; (void)removed; (void)added;
    ASSERT_LV1(false);
}

} // namespace Eval::NNUE::Features
} // namespace YaneuraOu

#endif  // defined(EVAL_NNUE)
