// NNUE 入力特徴量 ThreatEffect (task#73 v2 / report/52 §18.10)
//
// ★bullet-shogi `shogi_halfkp_threat_effect.rs` と同一 index 仕様:
//   effect_index = (((as*18 + ds*9 + dc) * 81 + to_n) * 3 + long_bucket) * 3 + short_bucket
//
// 2 系統の実装:
//   (a) THREAT_EFFECT_DIFF (既定): LONG_EFFECT_LIBRARY の board_effect (升ごとの利き数、玉の利き込み) と
//       long_effect (升に届いている長い利きの方向) からバケットを引く。長い利き数 = 方向 bit の popcount、
//       短い利き数 = 利き数 − 長い利き数。差分更新は do_move() で StateInfo に退避した直前局面の利き盤
//       (te_board_effect_prev / te_long_effect_prev) との比較 (HalfKPE9 と同型、ただし 1 手戻りは
//       StateInfo 側に持つので undo/null move の順序に依存しない)。
//   (b) THREAT_NAIVE_REBUILD: 判定用ナイーブ実装。effects_from() で利きを列挙して数え、毎手 reset。
// 長/短の分類は両者で一致させる (bullet の for_each_effect と同一):
//   long = 香/角/飛の射程 + 馬の斜め射程 + 龍の縦横射程 (隣接升を含む)、short = それ以外の 1 歩利き。
//   LONG_EFFECT_LIBRARY の long_effect も「馬・龍の長い利きは角・飛と同じ方向」で計算されるので一致する
//   (long_effect.cpp calc_effect / update_by_*)。

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

inline uint32_t effect_index(Color perspective, Color attacker_color,
                             Color target_color, int dc, Square to, int lb, int sb) {
    const Square to_n = (perspective == BLACK) ? to : Inv(to);
    const int as = (attacker_color != perspective) ? 1 : 0;
    const int ds = (target_color != perspective) ? 1 : 0;
    return ((uint32_t(as * 18 + ds * 9 + dc) * 81 + uint32_t(to_n)) * 3 + uint32_t(lb)) * 3 + uint32_t(sb);
}

#if defined(THREAT_EFFECT_DIFF)

// (長, 短) バケット
struct LS {
    uint8_t lb, sb;
    bool operator!=(const LS& o) const { return lb != o.lb || sb != o.sb; }
};

// 利き盤 (be: 色別の利き数、le: 長い利きの方向) から升 sq に届く色 c の利きのバケットを引く
inline LS ls_of(const LongEffect::ByteBoard* be, const LongEffect::WordBoard& le, Color c, Square sq) {
    const int total = int(be[c].effect(sq));
    const int lng   = int(POPCNT32(uint32_t(le.directions_of(c, sq))));
    const int sh    = total > lng ? total - lng : 0;   // total >= lng が不変条件 (方向 1 つにつき利き 1)
    return { uint8_t(bucket(lng)), uint8_t(bucket(sh)) };
}

#else

// 攻撃駒 pt の from→to の利きが「長い利き」か (ナイーブ実装用)
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

#endif

}  // namespace

#if defined(THREAT_EFFECT_DIFF)

// ---------------------------------------------------------------------------
// (a) 利き盤ベース (列挙なし) + 差分更新
// ---------------------------------------------------------------------------

// 特徴量のインデックスのリストを取得する (被弾駒 × 攻撃側 2 色、重複なし)
void ThreatEffect::AppendActiveIndices(const Position& pos, Color perspective, IndexList* active) {
    const LongEffect::ByteBoard* be = pos.board_effect;
    const LongEffect::WordBoard& le = pos.long_effect;

    Bitboard bb = pos.pieces() & ~pos.pieces(KING);
    while (bb) {
        const Square sq = bb.pop();
        const Piece pc = pos.piece_on(sq);
        const int dc = effect_class_of(type_of(pc));
        ASSERT_LV3(dc >= 0);
        const Color target_color = color_of(pc);
        for (Color ac : {BLACK, WHITE}) {
            const LS b = ls_of(be, le, ac, sq);
            const uint32_t idx = effect_index(perspective, ac, target_color, dc, sq, b.lb, b.sb);
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

// 一手前から値が変化したインデックス。
//   1) 動いた駒の旧配置 (from、旧駒種、直前局面の利き) を removed
//   2) 取られた駒 (to、直前局面の利き) を removed
//   3) 動いた駒の新配置 (to、現局面の利き) を added
//   4) それ以外の駒: 升の (利き数, 長い利き方向) が色ごとに変わった駒だけバケットを引き直し、
//      バケットが変わった (駒, 攻撃色) を removed/added
// 玉は被弾側から除外 (dc < 0) なので 1)〜3) で自然に落ちる。
void ThreatEffect::AppendChangedIndices(const Position& pos, Color perspective,
                                        IndexList* removed, IndexList* added) {
    const StateInfo* st = pos.state();
    const Move m = st->lastMove;
    ASSERT_LV3(m.to_u32() != 0);   // null move は feature_set 側 (dirty_num == 0) で弾かれる

    const LongEffect::ByteBoard* be_now = pos.board_effect;
    const LongEffect::WordBoard& le_now = pos.long_effect;
    const LongEffect::ByteBoard* be_prev = st->te_board_effect_prev;
    const LongEffect::WordBoard& le_prev = st->te_long_effect_prev;

    const Square to = m.to_sq();
    const bool drop = m.is_drop();
    const Piece moved_now = pos.piece_on(to);
    const Color mc = color_of(moved_now);
    const PieceType pt_now = type_of(moved_now);
    const PieceType pt_prev = m.is_promote() ? PieceType(pt_now - PIECE_PROMOTE) : pt_now;
    const Piece captured = st->capturedPiece;   // 無ければ NO_PIECE

    auto emit = [&](IndexList* list, Color ac, Color tc, int dc, Square sq, LS b) {
        const uint32_t idx = effect_index(perspective, ac, tc, dc, sq, b.lb, b.sb);
        ASSERT_LV3(idx < kDimensions);
        list->push_back(IndexType(idx));
    };

    // 1) 動いた駒の旧配置
    if (!drop) {
        const Square from = m.from_sq();
        const int dc = effect_class_of(pt_prev);
        if (dc >= 0)
            for (Color ac : {BLACK, WHITE})
                emit(removed, ac, mc, dc, from, ls_of(be_prev, le_prev, ac, from));
    }
    // 2) 取られた駒
    if (captured != NO_PIECE) {
        const int dc = effect_class_of(type_of(captured));
        if (dc >= 0)
            for (Color ac : {BLACK, WHITE})
                emit(removed, ac, color_of(captured), dc, to, ls_of(be_prev, le_prev, ac, to));
    }
    // 3) 動いた駒の新配置
    {
        const int dc = effect_class_of(pt_now);
        if (dc >= 0)
            for (Color ac : {BLACK, WHITE})
                emit(added, ac, mc, dc, to, ls_of(be_now, le_now, ac, to));
    }
    // 4) その他の駒 (to を除く。from は空升なので pieces() に含まれない)
    Bitboard bb = pos.pieces() & ~pos.pieces(KING);
    bb &= ~Bitboard(to);
    while (bb) {
        const Square sq = bb.pop();
        // 早期棄却: 利き数も長い利き方向も不変ならバケットは変わらない
        const bool chB = be_now[BLACK].effect(sq) != be_prev[BLACK].effect(sq)
                      || le_now.directions_of(BLACK, sq) != le_prev.directions_of(BLACK, sq);
        const bool chW = be_now[WHITE].effect(sq) != be_prev[WHITE].effect(sq)
                      || le_now.directions_of(WHITE, sq) != le_prev.directions_of(WHITE, sq);
        if (!chB && !chW)
            continue;
        const Piece pc = pos.piece_on(sq);
        const int dc = effect_class_of(type_of(pc));
        ASSERT_LV3(dc >= 0);
        const Color tc = color_of(pc);
        for (Color ac : {BLACK, WHITE}) {
            if (!(ac == BLACK ? chB : chW))
                continue;
            const LS o = ls_of(be_prev, le_prev, ac, sq);
            const LS n = ls_of(be_now, le_now, ac, sq);
            if (o != n) {
                emit(removed, ac, tc, dc, sq, o);
                emit(added, ac, tc, dc, sq, n);
            }
        }
    }
}

#else

// ---------------------------------------------------------------------------
// (b) 判定用ナイーブ実装: 列挙で利き数を数え、毎手 reset
// ---------------------------------------------------------------------------

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

#endif  // defined(THREAT_EFFECT_DIFF)

} // namespace Eval::NNUE::Features
} // namespace YaneuraOu

#endif  // defined(EVAL_NNUE)
