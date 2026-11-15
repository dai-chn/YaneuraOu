// NNUE 入力特徴量 Threat (task#52 Phase-1 / report/51)
//
// ★bullet-shogi `shogi_halfkp_threat.rs` / `shogi_halfka_hm_threat.rs` と同一 index 仕様。
//   両実装のズレは静かな学習/推論不一致になる ([[feedback_train_infer_parity_audit]])。
//   最終保証は nn.bin ロード時の hash 照合 + 同一局面での eval 数値一致検証。
//
// 座標系: bullet の Square(file*9+rank) は YO の Square (SQ_11=0) と同一。
// 視点変換: perspective==WHITE なら sq -> Inv(sq)、色 -> 反転。HM ミラー無し。

#include "../../../config.h"

#if defined(EVAL_NNUE)

#include "threat.h"
#include "threat_diff.h"
#include "index_list.h"
#include "../../../position.h"
#include "../../../bitboard.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace YaneuraOu {
namespace Eval::NNUE::Features {

namespace {

// ---------------------------------------------------------------------------
// ThreatClass (bullet 側と同一の 9 クラス。玉は除外)
// ---------------------------------------------------------------------------
enum class TC : int {
    Pawn = 0, Lance = 1, Knight = 2, Silver = 3, GoldLike = 4,
    Bishop = 5, Rook = 6, Horse = 7, Dragon = 8,
};
constexpr int kNumClasses = 9;
constexpr int kNumPatterns = 14;   // Black 9 + White の方向性駒 5

// PieceType -> ThreatClass (-1 = 玉/無効)
inline int threat_class_of(PieceType pt) {
    switch (pt) {
        case PAWN:       return int(TC::Pawn);
        case LANCE:      return int(TC::Lance);
        case KNIGHT:     return int(TC::Knight);
        case SILVER:     return int(TC::Silver);
        case GOLD:
        case PRO_PAWN:
        case PRO_LANCE:
        case PRO_KNIGHT:
        case PRO_SILVER: return int(TC::GoldLike);
        case BISHOP:     return int(TC::Bishop);
        case ROOK:       return int(TC::Rook);
        case HORSE:      return int(TC::Horse);
        case DRAGON:     return int(TC::Dragon);
        default:         return -1;   // KING ほか
    }
}

// 方向性駒 (後手は別 attack pattern)
inline bool is_directional(int tc) { return tc <= int(TC::GoldLike); }

// pattern id (bullet の attack_pattern_id と同一)
inline int attack_pattern_id(int tc, Color oriented_color) {
    return (oriented_color == WHITE && is_directional(tc)) ? kNumClasses + tc : tc;
}

// ---------------------------------------------------------------------------
// 空盤幾何利きの列挙 (bullet の attacks_empty_board と同一。raw 昇順で返す)
// ---------------------------------------------------------------------------
// ★bullet は file/rank を i8 で回して Square::new(f, r) = f*9+r。YO も sq = file*9+rank。
int attacks_empty_board(int tc, Color color, int sq_raw, std::array<uint8_t, 36>& targets) {
    const int file = sq_raw / 9;
    const int rank = sq_raw % 9;
    int count = 0;
    auto push = [&](int f, int r) {
        if (0 <= f && f < 9 && 0 <= r && r < 9)
            targets[count++] = uint8_t(f * 9 + r);
    };
    auto ray = [&](int df, int dr) {
        int f = file + df, r = rank + dr;
        while (0 <= f && f < 9 && 0 <= r && r < 9) {
            targets[count++] = uint8_t(f * 9 + r);
            f += df; r += dr;
        }
    };
    const int fwd = (color == BLACK) ? -1 : 1;

    switch (TC(tc)) {
        case TC::Pawn:   push(file, rank + fwd); break;
        case TC::Lance:  ray(0, fwd); break;
        case TC::Knight: push(file - 1, rank + 2 * fwd); push(file + 1, rank + 2 * fwd); break;
        case TC::Silver:
            for (auto [df, dr] : {std::pair{-1, fwd}, {0, fwd}, {1, fwd}, {-1, -fwd}, {1, -fwd}})
                push(file + df, rank + dr);
            break;
        case TC::GoldLike:
            for (auto [df, dr] : {std::pair{-1, fwd}, {0, fwd}, {1, fwd}, {-1, 0}, {1, 0}, {0, -fwd}})
                push(file + df, rank + dr);
            break;
        case TC::Bishop:
            ray(-1, -1); ray(-1, 1); ray(1, -1); ray(1, 1);
            break;
        case TC::Rook:
            ray(-1, 0); ray(1, 0); ray(0, -1); ray(0, 1);
            break;
        case TC::Horse:
            ray(-1, -1); ray(-1, 1); ray(1, -1); ray(1, 1);
            push(file - 1, rank); push(file + 1, rank); push(file, rank - 1); push(file, rank + 1);
            break;
        case TC::Dragon:
            ray(-1, 0); ray(1, 0); ray(0, -1); ray(0, 1);
            push(file - 1, rank - 1); push(file - 1, rank + 1); push(file + 1, rank - 1); push(file + 1, rank + 1);
            break;
    }

    // raw 昇順 (挿入ソート — bullet と同一)
    for (int i = 1; i < count; ++i) {
        uint8_t key = targets[i];
        int j = i;
        while (j > 0 && targets[j - 1] > key) { targets[j] = targets[j - 1]; --j; }
        targets[j] = key;
    }
    return count;
}

// ---------------------------------------------------------------------------
// テーブル群 (from_offset / attack_order / pair_base) — 初回に一度だけ構築
// ---------------------------------------------------------------------------
struct Tables {
    // pattern ごと: from -> 空盤利き数の前置和
    std::array<std::array<uint32_t, 81>, kNumPatterns> from_offset{};
    // pattern, from, to -> raw 昇順での順位 (攻撃しない組は 0xFF)
    std::array<std::array<std::array<uint8_t, 81>, 81>, kNumPatterns> attack_order{};
    // (attacker_side, ac, defender_side, dc) -> 累積 base (full profile なので除外なし)
    std::array<uint32_t, 2 * 9 * 2 * 9> pair_base{};
    uint32_t threat_dims = 0;
    // クラスごとの片色空盤利き総数 (pair_base の構築に使う)
    std::array<uint32_t, kNumClasses> attacks_per_color{};
    // クラス累積。class_cum[9] = 片側総数 6,020
    std::array<uint32_t, kNumClasses + 1> class_cum{};
    // pattern, from -> 空盤利き数 (視点反転の ord' = cnt - 1 - ord に使う、2026-09-19)
    std::array<std::array<uint8_t, 81>, kNumPatterns> attack_cnt{};

    Tables() {
        // attack_order は「攻撃しない組 = 0xFF」を先に敷いてから構築する
        for (auto& p : attack_order)
            for (auto& f : p)
                f.fill(0xFF);

        std::array<uint8_t, 36> tg{};
        auto fill_pattern = [&](int pat, int tc, Color c) -> uint32_t {
            uint32_t cum = 0;
            for (int sq = 0; sq < 81; ++sq) {
                from_offset[pat][sq] = cum;
                int cnt = attacks_empty_board(tc, c, sq, tg);
                for (int o = 0; o < cnt; ++o)
                    attack_order[pat][sq][tg[o]] = uint8_t(o);
                attack_cnt[pat][sq] = uint8_t(cnt);
                cum += cnt;
            }
            return cum;
        };
        for (int tc = 0; tc < kNumClasses; ++tc) {
            attacks_per_color[tc] = fill_pattern(tc, tc, BLACK);
            if (is_directional(tc)) {
                uint32_t w = fill_pattern(kNumClasses + tc, tc, WHITE);
                ASSERT_LV1(w == attacks_per_color[tc]);   // 対称なので同数のはず
                (void)w;
            }
        }

        class_cum[0] = 0;
        for (int tc = 0; tc < kNumClasses; ++tc)
            class_cum[tc + 1] = class_cum[tc] + attacks_per_color[tc];

        // pair_base (bullet build_pair_base と同一の走査順: as -> ac -> ds -> dc)
        uint32_t cum = 0;
        for (int as = 0; as < 2; ++as)
            for (int ac = 0; ac < kNumClasses; ++ac)
                for (int ds = 0; ds < 2; ++ds)
                    for (int dc = 0; dc < kNumClasses; ++dc) {
                        pair_base[as * 162 + ac * 18 + ds * 9 + dc] = cum;
                        cum += attacks_per_color[ac];
                    }
        threat_dims = cum;
        // 次元の検算 (threat.h の kDimensions と一致しなければ即死)
        ASSERT_LV1(threat_dims == Threat::kDimensions);

        // ★bullet 側 (shogi_halfkp_threat.rs test_tables_checksum) との決定的一致の保証。
        //   pair_base -> from_offset -> attack_order を同じ走査順で FNV-1a。
        //   ここが合えば index 計算は両実装で一致する (残る差は列挙経路のみ)。
        uint64_t h = 0xcbf29ce484222325ULL;
        auto mix = [&](uint64_t v) { h ^= v; h *= 0x100000001b3ULL; };
        for (auto b : pair_base) mix(b);
        for (int pat = 0; pat < kNumPatterns; ++pat)
            for (int sq = 0; sq < 81; ++sq)
                mix(from_offset[pat][sq]);
        for (int pat = 0; pat < kNumPatterns; ++pat)
            for (int f = 0; f < 81; ++f)
                for (int t = 0; t < 81; ++t)
                    mix(attack_order[pat][f][t]);
        if (h != 0x30f7eea2484893cdULL) {
            // ASSERT_LV は tournament ビルドで無効なので、明示的に落とす
            std::fprintf(stderr, "FATAL: Threat tables checksum mismatch: %016llx\n",
                         (unsigned long long)h);
            std::exit(1);
        }

        // 視点反転の関係 (2026-09-19、pair_index_both の前提): 盤を点対称に反転 (sq -> 80 - sq、色反転) すると
        // 空盤利きの raw 昇順が逆順になるので ord' = cnt - 1 - ord、cnt' = cnt。全 (pat, from, to) で検算し、
        // 崩れていたら即死 (テーブルの走査順を変えたときの静かな不一致を防ぐ)。
        for (int tc = 0; tc < kNumClasses; ++tc)
            for (int c = 0; c < 2; ++c) {
                const int pat  = attack_pattern_id(tc, Color(c));
                const int pat2 = attack_pattern_id(tc, ~Color(c));
                for (int f = 0; f < 81; ++f)
                    for (int t = 0; t < 81; ++t) {
                        const uint8_t o = attack_order[pat][f][t];
                        if (o == 0xFF)
                            continue;
                        const int cnt = attack_cnt[pat][f];
                        if (attack_cnt[pat2][80 - f] != cnt || attack_order[pat2][80 - f][80 - t] != uint8_t(cnt - 1 - o)) {
                            std::fprintf(stderr, "FATAL: Threat tables are not point-symmetric (pat %d from %d to %d)\n", pat, f, t);
                            std::exit(1);
                        }
                    }
            }
    }
};

const Tables& tables() {
    static const Tables t;
    return t;
}

// (attacker, victim) 対 -> 特徴 index (視点変換込み、HM ミラー無し)。
// AppendActiveIndices / AppendChangedIndices で共有 — 経路差ゼロの保証。
inline uint32_t pair_index(const Tables& T, Color perspective,
                           Color attacker_color, int ac, Square from,
                           Color target_color, int dc, Square to) {
    const Square from_n = (perspective == BLACK) ? from : Inv(from);
    const Square to_n   = (perspective == BLACK) ? to   : Inv(to);
    const Color oriented = (perspective == BLACK) ? attacker_color : ~attacker_color;
    const int as = (attacker_color != perspective) ? 1 : 0;
    const int ds = (target_color != perspective) ? 1 : 0;
    const int pat = attack_pattern_id(ac, oriented);
    const uint8_t ord = T.attack_order[pat][from_n][to_n];
    ASSERT_LV3(ord != 0xFF);
    return T.pair_base[as * 162 + ac * 18 + ds * 9 + dc] + T.from_offset[pat][from_n] + ord;
}

// (attacker, victim) 対 -> BLACK / WHITE 両視点の index を 1 パスで (2026-09-19、task#82、report/52 §23.3 c)。
// WHITE 視点は盤の点対称反転 (from/to は Inv、色は反転) なので、attack_order を引き直さずに
// ord_w = cnt - 1 - ord_b で出せる (Tables のコンストラクタで全数検算)。差分更新の写像コストの半分 (2 度目の 92 KB 表引き) が消える。
inline void pair_index_both(const Tables& T, Color attacker_color, int ac, Square from,
                            Color target_color, int dc, Square to, uint32_t& idx_b, uint32_t& idx_w) {
    const int pat_b = attack_pattern_id(ac, attacker_color);
    const uint8_t ord_b = T.attack_order[pat_b][from][to];
    ASSERT_LV3(ord_b != 0xFF);
    const int as_b = (attacker_color != BLACK) ? 1 : 0;
    const int ds_b = (target_color != BLACK) ? 1 : 0;
    const int pat_w = attack_pattern_id(ac, ~attacker_color);
    const Square from_w = Inv(from);
    const uint32_t ord_w = uint32_t(T.attack_cnt[pat_b][from]) - 1u - ord_b;
    const int as_w = as_b ^ 1, ds_w = ds_b ^ 1;
    idx_b = T.pair_base[as_b * 162 + ac * 18 + ds_b * 9 + dc] + T.from_offset[pat_b][from] + ord_b;
    idx_w = T.pair_base[as_w * 162 + ac * 18 + ds_w * 9 + dc] + T.from_offset[pat_w][from_w] + ord_w;
}

}  // namespace


// 特徴量のうち、値が 1 であるインデックスのリストを取得する
void Threat::AppendActiveIndices(const Position& pos, Color perspective, IndexList* active) {
    const Tables& T = tables();
    const Bitboard occ = pos.pieces();

    for (int sq_raw = 0; sq_raw < 81; ++sq_raw) {
        const Square from = Square(sq_raw);
        const Piece pc = pos.piece_on(from);
        if (pc == NO_PIECE)
            continue;
        const PieceType pt = type_of(pc);
        const int ac = threat_class_of(pt);
        if (ac < 0)
            continue;   // 玉
        const Color attacker_color = color_of(pc);

        // 実盤面の利き (遮断込み)
        Bitboard att = effects_from(pc, from, occ) & occ;
        while (att) {
            const Square to = att.pop();
            const Piece tpc = pos.piece_on(to);
            const int dc = threat_class_of(type_of(tpc));
            if (dc < 0)
                continue;   // 玉は被弾側からも除外
            const Color target_color = color_of(tpc);

            const uint32_t idx = pair_index(T, perspective, attacker_color, ac, from,
                                            target_color, dc, to);
            ASSERT_LV3(idx < kDimensions);
            active->push_back(IndexType(idx));
        }
    }
}

// 差分更新 (task#37, report/51 §7.2)。
//
// 方針: 「影響を受けた attacker」の被害者集合を prev/now 両占有で丸ごと再列挙し、
// 対称差分を removed/added にする。場合分け (成り/駒打ち/取り/開き当たり/遮断) を
// 集合演算に押し込むことで、ケース漏れバグを構造的に避ける。
//
// 影響 attacker = {動いた駒 (旧/新), 取られた駒} ∪ {from/to に (prev/now いずれかの
// 占有で) 利きを付けている駒}。スライダーの経路が from/to を跨ぐ場合、その駒は必ず
// from/to のマス自体に利きを持つ (途中マスにも利きが乗る) ので、この集合で完全。
// 桂・非スライダーは遮断の影響を受けないので from/to への利きが変わる駒のみで足りる。
//
// prev 側の駒配置の差は「動いた駒 (from に旧型)」「取られた駒 (to)」だけなので、
// 他の駒は現在の盤面 (pos.piece_on) をそのまま使い、from/to のみ読み替える。
#if defined(KEEP_LAST_MOVE)

// 駒レベルの対と視点間キャッシュの型は threat_diff.h の共有型を使う
// (ThreatLite も同じ駒レベル差分を流用する — task#59 ②)。
using ThreatPair = ThreatPiecePair;
using ThreatDiffCache = ThreatPieceDiff;
static thread_local ThreatDiffCache t_diff_cache;

// 2 つの収集器:
//   set   = 影響 attacker の利き先を prev/now 両占有で全列挙して対称差分 (task#37、場合分けなしで安全)
//   exact = 変わる対だけを手の幾何から直接出す (2026-09-18: ソート/マージ/不変対の列挙を全て省く)
// 既定は set。-DTHREAT_EXACT_DIFF で exact。
static void collect_piece_diff_set(const Position& pos, ThreatDiffCache& C);
static void collect_piece_diff_exact(const Position& pos, ThreatDiffCache& C);


// 直前手による駒レベル threat 差分 (視点非依存)。BLACK 呼び出しで計算し、
// 直後の WHITE 呼び出しはキャッシュヒットで再利用。StateInfo のアドレスは
// 使い回されるので (key, lastMove) も照合する。
static ThreatDiffCache& threat_piece_diff_mut(const Position& pos) {
    const StateInfo* st = pos.state();
    const Move m = st->lastMove;
    // dirty_num == 0 (null move) は feature_set 側で弾かれるのでここには来ない
    ASSERT_LV3(m.to_u32() != 0);
    ThreatDiffCache& C = t_diff_cache;
    const bool cache_hit = (C.st == (const void*)st && C.key == (uint64_t)st->key()
                            && C.move == m.to_u32());
    if (!cache_hit) {
        C.st = (const void*)st;
        C.key = (uint64_t)st->key();
        C.move = m.to_u32();
        C.n_removed = C.n_added = 0;
        C.mapped = false;
#if defined(THREAT_EXACT_DIFF)
        collect_piece_diff_exact(pos, C);
#else
        collect_piece_diff_set(pos, C);
#endif
    }
    return C;
}

const ThreatPieceDiff& threat_piece_diff(const Position& pos) {
    return threat_piece_diff_mut(pos);
}

void Threat::AppendChangedIndices(const Position& pos, Color perspective,
                                  IndexList* removed, IndexList* added) {
    const Tables& T = tables();
    ThreatDiffCache& C = threat_piece_diff_mut(pos);
    // ---- 駒レベル差分 -> 両視点の index を 1 パスで (最初の視点の呼び出しで作り、もう一方はコピー) ----
    if (!C.mapped) {
        auto map_all = [&](const ThreatPair* prs, int n, uint32_t (&out)[2][128]) {
            for (int i = 0; i < n; ++i) {
                const ThreatPair& pr = prs[i];
                const Piece apc = Piece(pr.attacker_pc);
                const Piece vpc = Piece(pr.victim_pc);
                pair_index_both(T, color_of(apc), threat_class_of(type_of(apc)), Square(pr.attacker_sq),
                                color_of(vpc), threat_class_of(type_of(vpc)), Square(pr.victim_sq),
                                out[BLACK][i], out[WHITE][i]);
            }
        };
        map_all(C.removed, C.n_removed, C.idx_removed);
        map_all(C.added, C.n_added, C.idx_added);
        C.mapped = true;
    }
    const uint32_t* r = C.idx_removed[perspective];
    for (int i = 0; i < C.n_removed; ++i)
        removed->push_back(IndexType(r[i]));
    const uint32_t* a = C.idx_added[perspective];
    for (int i = 0; i < C.n_added; ++i)
        added->push_back(IndexType(a[i]));
}

// 駒レベルの threat 差分を C.removed/C.added に収集する (視点非依存の 1 回だけの仕事)。set 版。
static void collect_piece_diff_set(const Position& pos, ThreatDiffCache& C) {
    const StateInfo* st = pos.state();
    const Move m = st->lastMove;
    const Square to = m.to_sq();
    const bool drop = m.is_drop();
    const Square from = drop ? SQ_NB : m.from_sq();
    const Piece moved_now = pos.piece_on(to);
    const Piece captured = st->capturedPiece;   // 無ければ NO_PIECE
    const Color mc = color_of(moved_now);
    const PieceType pt_now = type_of(moved_now);
    const PieceType pt_prev = m.is_promote() ? PieceType(pt_now - PIECE_PROMOTE) : pt_now;
    const Piece moved_prev = make_piece(mc, pt_prev);

    const Bitboard occ_now = pos.pieces();
    Bitboard occ_prev = occ_now;
    if (captured == NO_PIECE)
        occ_prev ^= Bitboard(to);      // 取りでなければ prev は to が空
    if (!drop)
        occ_prev |= Bitboard(from);    // prev は from に駒

    // prev 視点での被弾駒の解決: from -> 動いた駒 (旧型)、to -> 取られた駒、他は現盤面
    auto prev_piece_on = [&](Square v) -> Piece {
        if (!drop && v == from) return moved_prev;
        if (v == to) return captured;          // occ_prev に to が立つのは取りの時のみ
        return pos.piece_on(v);
    };

    // 駒レベル対の収集バッファ (影響 attacker ~≤16 × 被害者 ~≤17 で十分な上限)
    constexpr int kCap = 512;
    uint64_t prev_k[kCap]; ThreatPair prev_p[kCap]; int n_prev = 0;
    uint64_t now_k[kCap];  ThreatPair now_p[kCap];  int n_now = 0;

    auto emit_pairs = [&](Piece apc, Square asq, const Bitboard& occ, bool prev_side,
                          uint64_t* keys, ThreatPair* out, int& n) {
        const int ac = threat_class_of(type_of(apc));
        if (ac < 0)
            return;                    // 玉は攻撃側から除外
        Bitboard att = effects_from(apc, asq, occ) & occ;
        while (att) {
            const Square v = att.pop();
            const Piece vpc = prev_side ? prev_piece_on(v) : pos.piece_on(v);
            ASSERT_LV3(vpc != NO_PIECE);
            const int dc = threat_class_of(type_of(vpc));
            if (dc < 0)
                continue;              // 玉は被弾側からも除外
            ASSERT_LV1(n < kCap);
            out[n] = ThreatPair{uint8_t(apc), uint8_t(asq), uint8_t(vpc), uint8_t(v)};
            keys[n] = (uint64_t(apc) << 24) | (uint64_t(asq) << 16)
                    | (uint64_t(vpc) << 8) | uint64_t(v);
            ++n;
        }
    };

    // --- 影響 attacker の列挙 ---
    // (a) 動いた駒: prev = from に旧型 / now = to に新型
    if (!drop)
        emit_pairs(moved_prev, from, occ_prev, /*prev=*/true, prev_k, prev_p, n_prev);
    emit_pairs(moved_now, to, occ_now, /*prev=*/false, now_k, now_p, n_now);
    // (b) 取られた駒: prev のみ (to に居た)
    if (captured != NO_PIECE)
        emit_pairs(captured, to, occ_prev, /*prev=*/true, prev_k, prev_p, n_prev);
    // (c) from/to に利きを付けている他の駒 (現在の盤面上の駒。動いた駒 = to は除外)
    Bitboard aff = pos.attackers_to(to, occ_now) | pos.attackers_to(to, occ_prev);
    if (!drop)
        aff |= pos.attackers_to(from, occ_now) | pos.attackers_to(from, occ_prev);
    aff &= occ_now;
    aff &= ~Bitboard(to);
    while (aff) {
        const Square s = aff.pop();
        const Piece apc = pos.piece_on(s);   // prev でも同じ駒・同じマス
        emit_pairs(apc, s, occ_prev, /*prev=*/true, prev_k, prev_p, n_prev);
        emit_pairs(apc, s, occ_now, /*prev=*/false, now_k, now_p, n_now);
    }

    // --- 対称差分 (キー順に整列して merge) ---
    // 小配列の挿入ソート同等: std::sort で十分軽い
    int ord_prev[kCap], ord_now[kCap];
    for (int i = 0; i < n_prev; ++i) ord_prev[i] = i;
    for (int j = 0; j < n_now; ++j) ord_now[j] = j;
    std::sort(ord_prev, ord_prev + n_prev, [&](int a, int b) { return prev_k[a] < prev_k[b]; });
    std::sort(ord_now, ord_now + n_now, [&](int a, int b) { return now_k[a] < now_k[b]; });
    int i = 0, j = 0;
    while (i < n_prev || j < n_now) {
        if (j >= n_now || (i < n_prev && prev_k[ord_prev[i]] < now_k[ord_now[j]])) {
            ASSERT_LV1(C.n_removed < 128);
            C.removed[C.n_removed++] = prev_p[ord_prev[i++]];
        } else if (i >= n_prev || now_k[ord_now[j]] < prev_k[ord_prev[i]]) {
            ASSERT_LV1(C.n_added < 128);
            C.added[C.n_added++] = now_p[ord_now[j++]];
        } else { ++i; ++j; }           // 両方に居る対 = 不変
    }
}

// exact 版 (2026-09-18): 1 手で変わる対を手の幾何から直接出す。不変対の列挙・ソート・マージを全て省く。
//
// 1 手で変わり得る対は次の 5 群で尽きていて、互いに素 (二重計上なし):
//   A. 動いた駒が攻撃側の対: 旧升 (旧型) の利き先が全部消え、新升 (新型) の利き先が全部生える
//   B. 動いた駒が被弾側の対: from に利いていた駒 (prev) との対が消え、to に利いている駒 (now) との対が生える
//   C. 取られた駒が攻撃側の対: 全部消える (被弾側に動いた駒 (from) を含む)
//   D. 取られた駒が被弾側の対: 全部消える (攻撃側が動いた駒の分は A に含まれるので除く)
//   E. それ以外の駒 (攻撃側 ∉ {動いた駒, 取られた駒}、被弾側 ∉ {from, to}): 利き先が変わるのは
//      from/to を通る方向線を持つ長い利きの駒だけ。その駒の被弾集合を prev/now でビットボード差分する
//      (伸びた先/届かなくなった先は高々数升)。非スライダーの利き先は from/to 以外では変わらない。
// 玉は攻守とも対に入らない (threat_class_of < 0)。打ちは from 無し、成りは旧型/新型で A/B の駒種が変わる。
static void collect_piece_diff_exact(const Position& pos, ThreatDiffCache& C) {
    const StateInfo* st = pos.state();
    const Move m = st->lastMove;
    const Square to = m.to_sq();
    const bool drop = m.is_drop();
    const Square from = drop ? SQ_NB : m.from_sq();
    const Piece moved_now = pos.piece_on(to);
    const Piece captured = st->capturedPiece;   // 無ければ NO_PIECE
    const Color mc = color_of(moved_now);
    const PieceType pt_now = type_of(moved_now);
    const PieceType pt_prev = m.is_promote() ? PieceType(pt_now - PIECE_PROMOTE) : pt_now;
    const Piece moved_prev = make_piece(mc, pt_prev);

    const Bitboard occ_now = pos.pieces();
    Bitboard occ_prev = occ_now;
    if (captured == NO_PIECE)
        occ_prev ^= Bitboard(to);      // 取りでなければ prev は to が空
    if (!drop)
        occ_prev |= Bitboard(from);    // prev は from に駒

    // prev 視点での駒の解決: from -> 動いた駒 (旧型)、to -> 取られた駒、他は現盤面
    auto prev_piece_on = [&](Square v) -> Piece {
        if (!drop && v == from) return moved_prev;
        if (v == to) return captured;
        return pos.piece_on(v);
    };
    auto emit_removed = [&](Piece apc, Square asq, Piece vpc, Square vsq) {
        if (threat_class_of(type_of(apc)) < 0 || threat_class_of(type_of(vpc)) < 0)
            return;
        ASSERT_LV1(C.n_removed < 128);
        C.removed[C.n_removed++] = ThreatPair{uint8_t(apc), uint8_t(asq), uint8_t(vpc), uint8_t(vsq)};
    };
    auto emit_added = [&](Piece apc, Square asq, Piece vpc, Square vsq) {
        if (threat_class_of(type_of(apc)) < 0 || threat_class_of(type_of(vpc)) < 0)
            return;
        ASSERT_LV1(C.n_added < 128);
        C.added[C.n_added++] = ThreatPair{uint8_t(apc), uint8_t(asq), uint8_t(vpc), uint8_t(vsq)};
    };
    const bool mover_is_attacker = threat_class_of(pt_now) >= 0;   // 玉なら攻撃側の対は無い (pt_prev も玉)

    // --- A. 動いた駒が攻撃側 ---
    if (mover_is_attacker) {
        if (!drop) {
            Bitboard v = effects_from(moved_prev, from, occ_prev) & occ_prev;
            while (v) { const Square s = v.pop(); emit_removed(moved_prev, from, prev_piece_on(s), s); }
        }
        Bitboard v = effects_from(moved_now, to, occ_now) & occ_now;
        while (v) { const Square s = v.pop(); emit_added(moved_now, to, pos.piece_on(s), s); }
    }
    // --- C. 取られた駒が攻撃側 (prev のみ。被弾側 s == from は動いた駒の旧型) ---
    if (captured != NO_PIECE && threat_class_of(type_of(captured)) >= 0) {
        Bitboard v = effects_from(captured, to, occ_prev) & occ_prev;
        while (v) { const Square s = v.pop(); emit_removed(captured, to, prev_piece_on(s), s); }
    }
    // from/to への利き (現盤面の駒配置、占有だけ prev/now で読み替え)。動いた駒 (to) は自身の升に利かないが、
    // 旧升 from への利きには現れ得る (飛車が同じ筋を後退した等) ので prev 側では to を除く。
    const Bitboard mover_bb = Bitboard(to);
    Bitboard at_now  = pos.attackers_to(to, occ_now) & ~mover_bb;
    Bitboard at_prev = pos.attackers_to(to, occ_prev) & ~mover_bb;
    Bitboard af_prev, af_now;
    if (!drop) {
        af_prev = pos.attackers_to(from, occ_prev) & ~mover_bb;
        af_now  = pos.attackers_to(from, occ_now) & ~mover_bb;
    }
    // --- B. 動いた駒が被弾側 ---
    if (!drop) {
        Bitboard a = af_prev;
        while (a) { const Square s = a.pop(); emit_removed(pos.piece_on(s), s, moved_prev, from); }
    }
    {
        Bitboard a = at_now;
        while (a) { const Square s = a.pop(); emit_added(pos.piece_on(s), s, moved_now, to); }
    }
    // --- D. 取られた駒が被弾側 (prev のみ。動いた駒 (prev は from) は現盤面に from が無いので含まれない = A が担当) ---
    if (captured != NO_PIECE) {
        Bitboard a = at_prev;
        while (a) { const Square s = a.pop(); emit_removed(pos.piece_on(s), s, captured, to); }
    }
    // --- E. from/to を通る長い利きの駒の被弾集合の差分 (被弾側 from/to は A〜D が担当なので除く) ---
    {
        Bitboard aff = at_now | at_prev;
        if (!drop)
            aff |= af_prev | af_now;
        const Bitboard excl = drop ? mover_bb : (mover_bb | Bitboard(from));
        while (aff) {
            const Square s = aff.pop();
            const Piece apc = pos.piece_on(s);
            if (!has_long_effect(apc))
                continue;                                  // 非スライダーの利き先は from/to 以外で変わらない
            const Bitboard vp = effects_from(apc, s, occ_prev) & occ_prev & ~excl;
            const Bitboard vn = effects_from(apc, s, occ_now) & occ_now & ~excl;
            Bitboard rem = vp & ~vn;
            Bitboard add = vn & ~vp;
            while (rem) { const Square v = rem.pop(); emit_removed(apc, s, pos.piece_on(v), v); }
            while (add) { const Square v = add.pop(); emit_added(apc, s, pos.piece_on(v), v); }
        }
    }
}


#else

// KEEP_LAST_MOVE の無いビルド: kAnyPieceMoved (naive) に落ちるためここは呼ばれない
void Threat::AppendChangedIndices(const Position& /*pos*/, Color /*perspective*/,
                                  IndexList* /*removed*/, IndexList* /*added*/) {
    ASSERT_LV1(false);
}

#endif  // defined(KEEP_LAST_MOVE)

} // namespace Eval::NNUE::Features
} // namespace YaneuraOu

#endif  // defined(EVAL_NNUE)
