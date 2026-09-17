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
    // クラス累積 (attacker-major 配置の base 計算用)。class_cum[9] = 片側総数 6,020
    std::array<uint32_t, kNumClasses + 1> class_cum{};

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
#if defined(THREAT_ATTACKER_MAJOR)
    // attacker-major 配置 (task#59): (as, ac) ブロック内を (from, ord) スラブ × 18 (ds, dc) で並べる。
    // 同一攻撃駒 (ac, from) が触る行が 1 本の連続スラブに収まり、部分木内の再訪でホットに保たれる
    // (pair-major は同じ攻撃駒の行が 18 ブロックへ散る — report/51 §7.6.1)。
    // ★訓練時 (pair-major) と異なる index。nn.bin は PermuteRows がロード時に並び替える。
    return (T.class_cum[kNumClasses] * uint32_t(as) + T.class_cum[ac]
            + T.from_offset[pat][from_n] + ord) * 18u + uint32_t(ds * 9 + dc);
#else
    return T.pair_base[as * 162 + ac * 18 + ds * 9 + dc] + T.from_offset[pat][from_n] + ord;
#endif
}

}  // namespace

#if defined(THREAT_ATTACKER_MAJOR)
// 標準 (pair-major) 配置で学習された nn.bin の threat 行を attacker-major へロード時置換する。
// 純粋な行の relabel なので eval はビット一致のまま (検証はビルド後の eval 照合で行う)。
void Threat::PermuteRows(std::int16_t* weights, std::size_t half_dims, std::size_t row_offset) {
    const Tables& T = tables();
    const uint32_t total = T.class_cum[kNumClasses];
    std::vector<uint32_t> perm(kDimensions);   // old (pair-major) -> new (attacker-major)
    for (int as = 0; as < 2; ++as)
        for (int ac = 0; ac < kNumClasses; ++ac)
            for (int ds = 0; ds < 2; ++ds)
                for (int dc = 0; dc < kNumClasses; ++dc) {
                    const uint32_t ob = T.pair_base[as * 162 + ac * 18 + ds * 9 + dc];
                    const uint32_t nb = (total * uint32_t(as) + T.class_cum[ac]) * 18u
                                      + uint32_t(ds * 9 + dc);
                    const uint32_t n = T.attacks_per_color[ac];
                    for (uint32_t off = 0; off < n; ++off)
                        perm[ob + off] = nb + off * 18u;
                }
    // 全単射の検算 (置換もれ/重複は静かな評価破壊になるので即死させる)
    {
        std::vector<uint8_t> seen(kDimensions, 0);
        for (uint32_t i = 0; i < kDimensions; ++i) {
            if (perm[i] >= kDimensions || seen[perm[i]]) {
                std::fprintf(stderr, "FATAL: THREAT_ATTACKER_MAJOR permutation is not a bijection\n");
                std::exit(1);
            }
            seen[perm[i]] = 1;
        }
    }
    // 行の置換 (一時バッファ ~212MB、ロード時 1 回だけ)
    std::vector<std::int16_t> tmp(std::size_t(kDimensions) * half_dims);
    std::int16_t* rows = weights + row_offset * half_dims;
    for (uint32_t i = 0; i < kDimensions; ++i)
        std::memcpy(&tmp[std::size_t(perm[i]) * half_dims], rows + std::size_t(i) * half_dims,
                    half_dims * sizeof(std::int16_t));
    std::memcpy(rows, tmp.data(), tmp.size() * sizeof(std::int16_t));
}
#endif  // defined(THREAT_ATTACKER_MAJOR)

#if defined(THREAT_DIFF_STATS)
// 診断用カウンタ (研究ビルド限定)。atexit で stderr に出す。
// 2026-09-18: 差分収集 (collect_piece_diff_*) の rdtsc サイクルも集計する (列挙コストの直接計測、report/52 §18.9 の間接推定の検算)。
#include <atomic>
#include <x86intrin.h>
namespace {
std::atomic<uint64_t> g_refresh_calls{0}, g_changed_calls{0}, g_changed_rows{0},
    g_prev_pairs{0}, g_now_pairs{0}, g_collect_calls{0}, g_collect_cycles_set{0}, g_collect_cycles_exact{0};
struct StatsPrinter {
    ~StatsPrinter() {
        std::fprintf(stderr,
            "[threat-diff-stats] refresh=%llu changed=%llu rows/changed=%.2f prev+now_pairs/changed=%.2f"
            " collect=%llu cycles/collect set=%.0f exact=%.0f\n",
            (unsigned long long)g_refresh_calls.load(), (unsigned long long)g_changed_calls.load(),
            g_changed_calls ? double(g_changed_rows) / double(g_changed_calls) : 0.0,
            g_changed_calls ? double(g_prev_pairs + g_now_pairs) / double(g_changed_calls) : 0.0,
            (unsigned long long)g_collect_calls.load(),
            g_collect_calls ? double(g_collect_cycles_set) / double(g_collect_calls) : 0.0,
            g_collect_calls ? double(g_collect_cycles_exact) / double(g_collect_calls) : 0.0);
    }
} g_stats_printer;
}
#define THREAT_STAT(x) x
#else
#define THREAT_STAT(x)
#endif

// 特徴量のうち、値が 1 であるインデックスのリストを取得する
void Threat::AppendActiveIndices(const Position& pos, Color perspective, IndexList* active) {
    THREAT_STAT(g_refresh_calls.fetch_add(1, std::memory_order_relaxed);)
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
//   exact = 変わる対だけを手の幾何から直接出す (2026-09-18、ユーザ案: ソート/マージ/不変対の列挙を全て省く)
// 既定は set。-DTHREAT_EXACT_DIFF で exact、-DTHREAT_DIFF_XCHECK で両方走らせて集合一致を検査 (研究ビルド)。
static void collect_piece_diff_set(const Position& pos, ThreatDiffCache& C);
static void collect_piece_diff_exact(const Position& pos, ThreatDiffCache& C);

#if defined(THREAT_DIFF_XCHECK)
static thread_local ThreatDiffCache t_diff_cache2;
static void xcheck_piece_diff(const Position& pos, const ThreatDiffCache& A, const ThreatDiffCache& B);
#endif

// 直前手による駒レベル threat 差分 (視点非依存)。BLACK 呼び出しで計算し、
// 直後の WHITE 呼び出しはキャッシュヒットで再利用。StateInfo のアドレスは
// 使い回されるので (key, lastMove) も照合する。
const ThreatPieceDiff& threat_piece_diff(const Position& pos) {
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
#if defined(THREAT_DIFF_XCHECK)
        ThreatDiffCache& D = t_diff_cache2;
        D.n_removed = D.n_added = 0;
        THREAT_STAT(uint64_t t0 = __rdtsc();)
        collect_piece_diff_set(pos, C);
        THREAT_STAT(uint64_t t1 = __rdtsc();)
        collect_piece_diff_exact(pos, D);
        THREAT_STAT(uint64_t t2 = __rdtsc();
                    g_collect_calls.fetch_add(1, std::memory_order_relaxed);
                    g_collect_cycles_set.fetch_add(t1 - t0, std::memory_order_relaxed);
                    g_collect_cycles_exact.fetch_add(t2 - t1, std::memory_order_relaxed);)
        xcheck_piece_diff(pos, C, D);
#elif defined(THREAT_EXACT_DIFF)
        THREAT_STAT(uint64_t t0 = __rdtsc();)
        collect_piece_diff_exact(pos, C);
        THREAT_STAT(g_collect_calls.fetch_add(1, std::memory_order_relaxed);
                    g_collect_cycles_exact.fetch_add(__rdtsc() - t0, std::memory_order_relaxed);)
#else
        THREAT_STAT(uint64_t t0 = __rdtsc();)
        collect_piece_diff_set(pos, C);
        THREAT_STAT(g_collect_calls.fetch_add(1, std::memory_order_relaxed);
                    g_collect_cycles_set.fetch_add(__rdtsc() - t0, std::memory_order_relaxed);)
#endif
    }
    return C;
}

void Threat::AppendChangedIndices(const Position& pos, Color perspective,
                                  IndexList* removed, IndexList* added) {
    THREAT_STAT(g_changed_calls.fetch_add(1, std::memory_order_relaxed);)
    const Tables& T = tables();
    const ThreatPieceDiff& C = threat_piece_diff(pos);
    // ---- 駒レベル差分 -> この視点の index ----
    for (int i = 0; i < C.n_removed; ++i) {
        const ThreatPair& pr = C.removed[i];
        const Piece apc = Piece(pr.attacker_pc);
        const Piece vpc = Piece(pr.victim_pc);
        removed->push_back(IndexType(pair_index(T, perspective,
            color_of(apc), threat_class_of(type_of(apc)), Square(pr.attacker_sq),
            color_of(vpc), threat_class_of(type_of(vpc)), Square(pr.victim_sq))));
    }
    for (int i = 0; i < C.n_added; ++i) {
        const ThreatPair& pr = C.added[i];
        const Piece apc = Piece(pr.attacker_pc);
        const Piece vpc = Piece(pr.victim_pc);
        added->push_back(IndexType(pair_index(T, perspective,
            color_of(apc), threat_class_of(type_of(apc)), Square(pr.attacker_sq),
            color_of(vpc), threat_class_of(type_of(vpc)), Square(pr.victim_sq))));
    }
    THREAT_STAT(g_changed_rows.fetch_add(C.n_removed + C.n_added, std::memory_order_relaxed);)
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

    THREAT_STAT(g_prev_pairs.fetch_add(n_prev, std::memory_order_relaxed);
                g_now_pairs.fetch_add(n_now, std::memory_order_relaxed);)
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

#if defined(THREAT_DIFF_XCHECK)
// set 版と exact 版の removed/added を集合として比較。不一致なら局面と手を出して即死 (研究ビルド)。
static void xcheck_piece_diff(const Position& pos, const ThreatDiffCache& A, const ThreatDiffCache& B) {
    auto key = [](const ThreatPair& p) {
        return (uint64_t(p.attacker_pc) << 24) | (uint64_t(p.attacker_sq) << 16)
             | (uint64_t(p.victim_pc) << 8) | uint64_t(p.victim_sq);
    };
    auto sorted = [&](const ThreatPair* v, int n) {
        std::vector<uint64_t> k(n);
        for (int i = 0; i < n; ++i) k[i] = key(v[i]);
        std::sort(k.begin(), k.end());
        return k;
    };
    const auto ar = sorted(A.removed, A.n_removed), br = sorted(B.removed, B.n_removed);
    const auto aa = sorted(A.added, A.n_added),     ba = sorted(B.added, B.n_added);
    if (ar != br || aa != ba) {
        std::fprintf(stderr, "FATAL: threat diff xcheck mismatch: sfen %s lastmove %s set(rem %d add %d) exact(rem %d add %d)\n",
                     pos.sfen().c_str(), to_usi_string(pos.state()->lastMove).c_str(),
                     A.n_removed, A.n_added, B.n_removed, B.n_added);
        auto dump = [&](const char* name, const std::vector<uint64_t>& k) {
            std::fprintf(stderr, "  %s:", name);
            for (auto x : k) std::fprintf(stderr, " %llx", (unsigned long long)x);
            std::fprintf(stderr, "\n");
        };
        dump("set.removed", ar); dump("exact.removed", br); dump("set.added", aa); dump("exact.added", ba);
        std::exit(1);
    }
}
#endif

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
