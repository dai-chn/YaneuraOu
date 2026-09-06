// NNUE 入力特徴量 ThreatEffect (task#73 v2 / report/52 §18.10)
//
// ★bullet-shogi `shogi_halfkp_threat_effect.rs` と同一 index 仕様:
//   effect_index = (((as*18 + ds*9 + dc) * 81 + to_n) * 3 + long_bucket) * 3 + short_bucket
// 利き数は列挙で数える (判定用ナイーブ実装)。長/短の分類は bullet の for_each_effect と同一:
//   long = 香/角/飛の射程 + 馬の斜め射程 + 龍の縦横射程、short = それ以外の 1 歩利き (玉を含む)。
// effects_from() は駒種ごとの利き bitboard を返すので、馬/龍は「斜め/縦横」を幾何で分ける。

#include "../../../config.h"

#if defined(EVAL_NNUE)

#include "threat_effect.h"
#include "index_list.h"
#include "../../../position.h"
#include "../../../bitboard.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>

namespace YaneuraOu {
namespace Eval::NNUE::Features {

namespace {

// ThreatClass (threat.cpp と同一の 9 クラス。玉は除外) — 被弾側の分類に使う
inline int effect_class_of(PieceType pt) {
    switch (pt) {
        case PAWN:       return 0;
        case LANCE:      return 1;
        case KNIGHT:     return 2;
        case SILVER:     return 3;
        case GOLD:
        case PRO_PAWN:
        case PRO_LANCE:
        case PRO_KNIGHT:
        case PRO_SILVER: return 4;
        case BISHOP:     return 5;
        case ROOK:       return 6;
        case HORSE:      return 7;
        case DRAGON:     return 8;
        default:         return -1;   // KING ほか
    }
}

inline int bucket(int n) { return n >= 2 ? 2 : n; }

// 攻撃駒 pt の from→to の利きが「長い利き」か
inline bool is_long_effect(PieceType pt, Square from, Square to) {
    const int df = int(file_of(to)) - int(file_of(from));
    const int dr = int(rank_of(to)) - int(rank_of(from));
    switch (pt) {
        case LANCE: case BISHOP: case ROOK: return true;
        case HORSE:  return df == dr || df == -dr;          // 斜め = 角行き
        case DRAGON: return df == 0 || dr == 0;              // 縦横 = 飛行き
        default:     return false;
    }
}

inline uint32_t effect_index(Color perspective, Color attacker_color,
                             Color target_color, int dc, Square to, int lb, int sb) {
    const Square to_n = (perspective == BLACK) ? to : Inv(to);
    const int as = (attacker_color != perspective) ? 1 : 0;
    const int ds = (target_color != perspective) ? 1 : 0;
    return ((uint32_t(as * 18 + ds * 9 + dc) * 81 + uint32_t(to_n)) * 3 + uint32_t(lb)) * 3 + uint32_t(sb);
}

}  // namespace

// 特徴量のインデックスのリストを取得する (被弾駒 × 攻撃側 2 色、重複なし)
void ThreatEffect::AppendActiveIndices(const Position& pos, Color perspective, IndexList* active) {
    const Bitboard occ = pos.pieces();

    // 各升の利き数 [色][升] = (long, short)。玉の利きも数える (board_effect と同定義)
    uint8_t lc[2][SQ_NB] = {}, sc[2][SQ_NB] = {};
    for (int sq_raw = 0; sq_raw < 81; ++sq_raw) {
        const Square from = Square(sq_raw);
        const Piece pc = pos.piece_on(from);
        if (pc == NO_PIECE)
            continue;
        const PieceType pt = type_of(pc);
        const int ci = (color_of(pc) == WHITE) ? 1 : 0;
        Bitboard att = effects_from(pc, from, occ);
        while (att) {
            const Square to = att.pop();
            if (is_long_effect(pt, from, to))
                ++lc[ci][to];
            else
                ++sc[ci][to];
        }
    }

    for (int sq_raw = 0; sq_raw < 81; ++sq_raw) {
        const Square sq = Square(sq_raw);
        const Piece pc = pos.piece_on(sq);
        if (pc == NO_PIECE)
            continue;
        const int dc = effect_class_of(type_of(pc));
        if (dc < 0)
            continue;   // 玉は被弾側から除外
        const Color target_color = color_of(pc);
        for (Color ac : {BLACK, WHITE}) {
            const int ci = (ac == WHITE) ? 1 : 0;
            const uint32_t idx = effect_index(perspective, ac, target_color, dc, sq,
                                              bucket(lc[ci][sq]), bucket(sc[ci][sq]));
            ASSERT_LV3(idx < kDimensions);
            active->push_back(IndexType(idx));
#if defined(THREAT_EFFECT_DUMP)
            std::fprintf(stderr, "TE %c %u\n", perspective == BLACK ? 'B' : 'W', idx);
#endif
        }
    }
#if defined(THREAT_EFFECT_DUMP)
    std::fprintf(stderr, "TE %c END\n", perspective == BLACK ? 'B' : 'W');
#endif
}

// kAnyPieceMoved は常に reset (= AppendActiveIndices) になるので呼ばれない
void ThreatEffect::AppendChangedIndices(const Position& pos, Color perspective,
                                        IndexList* removed, IndexList* added) {
    (void)pos; (void)perspective; (void)removed; (void)added;
    ASSERT_LV1(false);
}

} // namespace Eval::NNUE::Features
} // namespace YaneuraOu

#endif  // defined(EVAL_NNUE)
