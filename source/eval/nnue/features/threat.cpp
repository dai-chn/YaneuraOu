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
#include "index_list.h"
#include "../../../position.h"
#include "../../../bitboard.h"

#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

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

            // 視点変換 (HM ミラー無し)
            const Square from_n = (perspective == BLACK) ? from : Inv(from);
            const Square to_n   = (perspective == BLACK) ? to   : Inv(to);
            const Color oriented = (perspective == BLACK) ? attacker_color : ~attacker_color;
            const int as = (attacker_color != perspective) ? 1 : 0;
            const int ds = (target_color != perspective) ? 1 : 0;

            const int pat = attack_pattern_id(ac, oriented);
            const uint8_t ord = T.attack_order[pat][from_n][to_n];
            ASSERT_LV3(ord != 0xFF);
            const uint32_t idx = T.pair_base[as * 162 + ac * 18 + ds * 9 + dc]
                               + T.from_offset[pat][from_n] + ord;
            ASSERT_LV3(idx < kDimensions);
            active->push_back(IndexType(idx));
        }
    }
}

// kAnyPieceMoved は feature_set 側で常に reset 扱いになるため、ここは呼ばれない
void Threat::AppendChangedIndices(const Position& /*pos*/, Color /*perspective*/,
                                  IndexList* /*removed*/, IndexList* /*added*/) {
    ASSERT_LV1(false);
}

} // namespace Eval::NNUE::Features
} // namespace YaneuraOu

#endif  // defined(EVAL_NNUE)
