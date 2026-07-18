// exp009 静止局面フィルタ — N ペアシャードを 1 パスで処理して
// `!in_check && MoveList<CAPTURES>.empty()` を満たす record のみを書き出す。
//
// この USI コマンドは NNUE evaluator も学習側のサポート関数も呼ばないので、
// 元々の EVAL_LEARN ビルドだけでなく、ノーマルビルド (USE_SFEN_PACKER だけ) でも
// 動くように、必要な PackedSfenValue 構造体をここで自前定義する。
//
// 使い方:
//   filter_quiet --in p1 [--in p2 ...] --out q1 [--out q2 ...] [--check-pair on|off]
#include "../config.h"

// USE_SFEN_PACKER は通常ビルドで常時有効 (config.h 433 行)。これが無いと
// Position::set_from_packed_sfen / PackedSfen 構造体が使えないので、ガードは
// USE_SFEN_PACKER ベース。EVAL_LEARN ビルドでも問題なく通る。
#if defined(USE_SFEN_PACKER)

#include <cstdint>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "../position.h"
#include "../movegen.h"
// learn.h は EVAL_LEARN ガードのため通常ビルドでは空。
// PackedSfenValue は EVAL_LEARN とは独立した bin フォーマットなので、ここで
// EVAL_LEARN 版と完全に同じレイアウトの構造体を局所定義する (40 byte = 32 + 2 +
// 2 + 2 + 1 + 1)。フィールド順とサイズが食い違うと既存 PSV ファイルを誤読する
// ので、static_assert で安全網を張る。

namespace YaneuraOu {
namespace Learner {

namespace {

// EVAL_LEARN ビルドの learn.h::PackedSfenValue と同一レイアウト。
// Mirror of: YaneuraOu/source/learn/learn.h::Learner::PackedSfenValue (~line 198).
// EVAL_LEARN is not enabled in our build, so we cannot use the upstream struct
// directly. If the upstream layout changes (field added/reordered), the
// static_assert(sizeof==40) below will fire — update this definition then.
// pragma pack is a defense-in-depth safeguard, not load-bearing for the current
// layout (PackedSfen(32) + s16 + u16 + u16 + s8 + u8 all naturally align to 2).
// If a wider field (s64, double) is added upstream, the natural alignment may
// shift to 4 or 8 and this pragma becomes necessary.
#pragma pack(push, 1)
struct LocalPackedSfenValue {
    PackedSfen sfen;        // 32 byte
    int16_t    score;       // 2  byte
    uint16_t   move;        // 2  byte
    uint16_t   gamePly;     // 2  byte
    int8_t     game_result; // 1  byte
    uint8_t    padding;     // 1  byte
};
#pragma pack(pop)
static_assert(sizeof(LocalPackedSfenValue) == 40,
              "PSV record must be exactly 40 bytes (32+2+2+2+1+1)");
static_assert(sizeof(PackedSfen) == 32, "PackedSfen must be 32 bytes");

constexpr size_t REC_SIZE = sizeof(LocalPackedSfenValue);

enum class ReadResult { Ok, Eof, Truncated };

ReadResult read_record(std::ifstream& f, LocalPackedSfenValue& rec) {
    f.read(reinterpret_cast<char*>(&rec), REC_SIZE);
    auto n = static_cast<size_t>(f.gcount());
    if (n == REC_SIZE) return ReadResult::Ok;
    if (n == 0) return ReadResult::Eof;
    return ReadResult::Truncated;
}

bool is_quiet(Position& pos) {
    if (pos.in_check()) return false;
    // exp009 spec の qsearch 入口判定 = captures-only。CAPTURES_PRO_PLUS は
    // promotions も含む別概念。
    MoveList<CAPTURES> ml(pos);
    return ml.size() == 0;
}

} // namespace

// USI: filter_quiet --in p1 --in p2 ... --out q1 --out q2 ... [--check-pair on|off]
// 必要条件: in と out が同数 (N 個), 1 <= N。
void filter_quiet_cmd(Position& pos, std::istringstream& is) {
    std::vector<std::string> in_paths, out_paths;
    bool check_pair = true;
    std::string token;
    while (is >> token) {
        if (token == "--in") {
            std::string p; is >> p; in_paths.push_back(p);
        } else if (token == "--out") {
            std::string p; is >> p; out_paths.push_back(p);
        } else if (token == "--check-pair") {
            std::string v; is >> v; check_pair = (v == "on" || v == "1" || v == "true");
        } else {
            std::cerr << "filter_quiet: unknown token: " << token << std::endl;
            std::cout << R"({"error":"unknown token","token":")" << token << "\"}" << std::endl;
            return;
        }
    }
    const size_t N = in_paths.size();
    if (N == 0 || N != out_paths.size()) {
        std::cout << R"({"error":"in/out count mismatch","n_in":)" << in_paths.size()
                  << R"(,"n_out":)" << out_paths.size() << "}" << std::endl;
        return;
    }

    std::vector<std::ifstream> ins;
    std::vector<std::ofstream> outs;
    ins.reserve(N); outs.reserve(N);
    for (size_t i = 0; i < N; ++i) {
        ins.emplace_back(in_paths[i], std::ios::binary);
        if (!ins.back()) {
            std::cout << R"({"error":"cannot open in","path":")" << in_paths[i] << "\"}" << std::endl;
            return;
        }
        outs.emplace_back(out_paths[i], std::ios::binary);
        if (!outs.back()) {
            std::cout << R"({"error":"cannot open out","path":")" << out_paths[i] << "\"}" << std::endl;
            return;
        }
    }

    std::vector<LocalPackedSfenValue> recs(N);
    std::vector<ReadResult> rrs(N);
    uint64_t total = 0, kept = 0;

    while (true) {
        for (size_t i = 0; i < N; ++i) {
            rrs[i] = read_record(ins[i], recs[i]);
        }
        // 1) Truncated を最優先で検出 — シャードが破損している。直ちに停止。
        bool truncated = false;
        for (size_t i = 0; i < N; ++i) {
            if (rrs[i] == ReadResult::Truncated) {
                std::cout << R"({"warning":"truncated input record","pair":)" << i
                          << R"(,"record":)" << (total + 1) << "}" << std::endl;
                truncated = true;
                break;
            }
        }
        if (truncated) break;
        // 2) Eof / Ok が混在 → 同じ record boundary で長さがずれている = corruption。
        bool any_eof = false, any_ok = false;
        for (size_t i = 0; i < N; ++i) {
            if (rrs[i] == ReadResult::Eof) any_eof = true;
            else if (rrs[i] == ReadResult::Ok) any_ok = true;
        }
        if (any_eof && any_ok) {
            std::cout << R"({"warning":"eof/ok mismatch across pairs","record":)" << (total + 1) << "}" << std::endl;
            break;
        }
        // 3) 全員 Eof → 正常終了。
        if (any_eof) break;
        ++total;
        if (check_pair) {
            for (size_t i = 1; i < N; ++i) {
                if (!(recs[0].sfen == recs[i].sfen)) {
                    std::cout << R"({"error":"sfen mismatch at record","record":)" << total
                              << R"(,"pair":)" << i << "}" << std::endl;
                    return;
                }
            }
        }
        StateInfo si;
        if (pos.set_from_packed_sfen(recs[0].sfen, &si, false, 0).is_not_ok()) {
            // 復元失敗 record はスキップ (フィルタアウト)
            continue;
        }
        if (!is_quiet(pos)) continue;
        for (size_t i = 0; i < N; ++i) {
            outs[i].write(reinterpret_cast<const char*>(&recs[i]), REC_SIZE);
        }
        for (size_t i = 0; i < N; ++i) {
            if (!outs[i].good()) {
                std::cout << R"({"error":"write failed","pair":)" << i
                          << R"(,"record":)" << total << "}" << std::endl;
                return;  // abort the shard — the output is corrupt
            }
        }
        ++kept;
    }

    std::cout << R"({"total":)" << total
              << R"(,"kept":)" << kept
              << R"(,"quiet_rate":)" << (total ? (double)kept / total : 0.0)
              << "}" << std::endl;
}

} // namespace Learner
} // namespace YaneuraOu

#endif // USE_SFEN_PACKER
