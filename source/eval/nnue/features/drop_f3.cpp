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

    const Color    e       = ~c;
    const Bitboard occ     = pos.pieces();
    const Bitboard empties = pos.empties();

    // v0b (10-24): 逆向きの列挙。v0 (持ち駒 × 空き升ごとに利きを出して相手駒を探す、候補 ~135/視点 + 当たりごとに attackers_to)
    // は NPS が王者の 0.50 だった。ここでは相手の駒 (歩以外) t ごとに「p を打つと t に当たる升」= 色を反転した p の t からの利き
    // (経路の空きは対称なので同値) を取り、打ち規則のマスクと安全性 (相手の利き集合を 1 回で作る) で絞る。
    // 利きの計算は (持ち駒の種類数 × 相手の駒数) 回 + 相手の駒数回。

    // 静的な打ちの規則: 空き升、歩は二歩と最終段 (pawn_drop_mask)、香は最終段、桂は最終 2 段を除く。
    // 🔴 Bitboard::andnot(b) は ~this & b (SSE の andnot の向き)。禁止段を外すのは cand &= ~mask
    //    (10-24: cand.andnot(mask) と書いていて香・桂の行が全部消えていた。合成ネットの nnuedump 照合で発見)。
    Bitboard cand[7];
    for (int pi = 0; pi < 7; ++pi) {
        const PieceType pt = kHandOrder[pi];
        if (!hand_exists(hand, pt)) {
            cand[pi] = Bitboard(ZERO);
            continue;
        }
        Bitboard m = empties;
        if (pt == PAWN)
            m &= (c == BLACK) ? pawn_drop_mask<BLACK>(pos.pieces(BLACK, PAWN))
                              : pawn_drop_mask<WHITE>(pos.pieces(WHITE, PAWN));
        else if (pt == LANCE)
            m &= ~rank1_n_bb(c, RANK_1);
        else if (pt == KNIGHT)
            m &= ~rank1_n_bb(c, RANK_2);
        cand[pi] = m;
    }

    // 相手の利き集合 (玉を含む): ここに利きのある升へ打つと取られる
#if defined(DROP_F3_NO_SAFETY)
    // 計測用 (10-24): 安全性判定を外して相手の利き集合の計算コストを分離する。特徴の意味が変わるので判定ビルドには使わない
    const Bitboard safe = ~Bitboard(ZERO);
#else
    Bitboard enemy_att(ZERO);
    for (Bitboard bb = pos.pieces(e); bb;) {
        const Square t = bb.pop();
        enemy_att |= effects_from(pos.piece_on(t), t, occ);
    }
    const Bitboard safe = ~enemy_att;
#endif

    // (p, vc) ごとの升集合。歩には立てない (vc < 0)、玉 (vc 8) は安全性を問わない (王手の打ち = F1)
    Bitboard rows[7][9];
    for (auto& r : rows)
        for (auto& b : r)
            b = Bitboard(ZERO);
    for (Bitboard bb = pos.pieces(e) & ~pos.pieces(e, PAWN); bb;) {
        const Square t  = bb.pop();
        const int    vc = target_class(type_of(pos.piece_on(t)));
        if (vc < 0)
            continue;
        for (int pi = 0; pi < 7; ++pi) {
            if (!cand[pi])
                continue;
            Bitboard rev = effects_from(make_piece(e, kHandOrder[pi]), t, occ) & cand[pi];
            if (vc != 8)
                rev &= safe;
            if (rev)
                rows[pi][vc] |= rev;
        }
    }

#if defined(DROP_F3_NO_EMIT)
    // 計測用 (10-24): 列挙はするが行を出さない (accumulator の行トラフィックを分離)。計算が消されないよう volatile に沈める
    {
        static volatile uint64_t sink;
        uint64_t x = 0;
        for (auto& r : rows)
            for (auto& b : r)
                x ^= b.merge();
        sink = x;
        return;
    }
#endif
    // 出力の順序は v0 と同じ: p 昇順 → s 昇順 (回転前の升番号) → vc 昇順。同じ (p, s, vc) は集合なので 1 回
    for (int pi = 0; pi < 7; ++pi) {
        Bitboard all(ZERO);
        for (int vc = 0; vc < 9; ++vc)
            all |= rows[pi][vc];
        while (all) {
            const Square s   = all.pop();
            const Square s_n = (c == BLACK) ? s : Inv(s);
            for (int vc = 0; vc < 9; ++vc)
                if (rows[pi][vc].test(s)) {
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
