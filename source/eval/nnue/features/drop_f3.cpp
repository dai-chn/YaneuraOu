// NNUE 入力特徴量 DropF3 (安全な打ち当たり) の列挙。仕様は drop_f3.h と spec 2026-10-23 を参照。
// ★BulletOu 側と同一 index 仕様。列挙の順序 (p 昇順 → s 昇順 (回転前の升番号) → vc 昇順) も揃える。

#include "../../../config.h"

#if defined(EVAL_NNUE)

#include "drop_f3.h"
#include "index_list.h"
#include "../../../position.h"
#include "../../../bitboard.h"

#include <cstdint>

namespace YaneuraOu {
namespace Eval::NNUE::Features {

namespace {

// 持ち駒の順 (index の p): 歩 香 桂 銀 金 角 飛。★変更禁止 (bullet と不一致になる)
constexpr PieceType kHandOrder[7] = { PAWN, LANCE, KNIGHT, SILVER, GOLD, BISHOP, ROOK };

// 当たる駒のクラス vc: 香 0, 桂 1, 銀 2, 金類 3 (金・と・成香・成桂・成銀), 角 4, 馬 5, 飛 6, 龍 7, 玉 8。歩は -1 (行を立てない)
inline int target_class(PieceType pt) {
    switch (pt) {
        case LANCE:      return 0;
        case KNIGHT:     return 1;
        case SILVER:     return 2;
        case GOLD:
        case PRO_PAWN:
        case PRO_LANCE:
        case PRO_KNIGHT:
        case PRO_SILVER: return 3;
        case BISHOP:     return 4;
        case HORSE:      return 5;
        case ROOK:       return 6;
        case DRAGON:     return 7;
        case KING:       return 8;
        default:         return -1;   // PAWN
    }
}

}  // namespace

void DropF3::AppendActiveIndices(const Position& pos, Color perspective, IndexList* active) {
    const Color c = perspective;
    const Hand hand = pos.hand_of(c);
    if (hand == HAND_ZERO)
        return;

    const Bitboard occ     = pos.pieces();
    const Bitboard empties = pos.empties();
    const Bitboard enemy   = pos.pieces(~c);

    for (int pi = 0; pi < 7; ++pi) {
        const PieceType pt = kHandOrder[pi];
        if (!hand_exists(hand, pt))
            continue;

        // 静的な打ちの規則: 空き升、歩・香は最終段を除く、桂は最終 2 段を除く、歩は二歩を除く
        Bitboard cand = empties;
        if (pt == PAWN)
            cand &= (c == BLACK) ? pawn_drop_mask<BLACK>(pos.pieces(BLACK, PAWN))
                                 : pawn_drop_mask<WHITE>(pos.pieces(WHITE, PAWN));
        // 🔴 Bitboard::andnot(b) は ~this & b (SSE の andnot の向き) なので cand.andnot(mask) では「空き升でない × 禁止段」になる
        //    (10-24: 合成ネットの nnuedump 照合で香・桂の行が全部消えていた)。禁止段を外すのは cand &= ~mask。
        else if (pt == LANCE)
            cand &= ~rank1_n_bb(c, RANK_1);
        else if (pt == KNIGHT)
            cand &= ~rank1_n_bb(c, RANK_2);

        const Piece pc = make_piece(c, pt);
        while (cand) {
            const Square s = cand.pop();
            // p を s に置いたときの利き先にある相手の駒 (長い利きは今の占有で止まる)
            Bitboard att = effects_from(pc, s, occ) & enemy;
            if (!att)
                continue;
            uint32_t mask = 0;
            int safe = -1;   // 未判定
            while (att) {
                const Square t  = att.pop();
                const int    vc = target_class(type_of(pos.piece_on(t)));
                if (vc < 0)
                    continue;                 // 歩には立てない
                if (vc == 8) {
                    mask |= 1u << 8;          // 王手の打ち (F1): 安全性を問わない
                    continue;
                }
                if (safe < 0)
                    safe = pos.attackers_to(~c, s) ? 0 : 1;   // s に相手の利き (玉を含む) があれば取られる
                if (safe)
                    mask |= 1u << vc;
            }
            if (!mask)
                continue;
            const Square s_n = (c == BLACK) ? s : Inv(s);
            for (int vc = 0; vc < 9; ++vc)
                if (mask & (1u << vc)) {
                    const uint32_t idx = uint32_t(pi) * 729 + uint32_t(s_n) * 9 + uint32_t(vc);
                    ASSERT_LV3(idx < kDimensions);
                    active->push_back(IndexType(idx));
                }
        }
    }
}

// kAnyPieceMoved は常に reset (= AppendActiveIndices) になるので呼ばれない
void DropF3::AppendChangedIndices(const Position& pos, Color perspective,
                                  IndexList* removed, IndexList* added) {
    (void)pos; (void)perspective; (void)removed; (void)added;
    ASSERT_LV1(false);
}

} // namespace Eval::NNUE::Features
} // namespace YaneuraOu

#endif  // defined(EVAL_NNUE)
