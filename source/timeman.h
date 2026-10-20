#ifndef TIMEMAN_H_INCLUDED
#define TIMEMAN_H_INCLUDED

#include <cmath>
#include <cstdint>

#include "misc.h"

namespace YaneuraOu {

class OptionsMap;
enum Color : uint8_t;

namespace Search {
	struct LimitsType;
}

// The TimeManagement class computes the optimal time to think depending on
// the maximum available time, the game move number, and other parameters.

// TimeManagementクラスは、使用可能な最大時間、対局の手数、
// その他のパラメータに応じて、思考に費やす最適な時間を計算します。

class TimeManagement {
   public:
    // 起動時に呼び出す。
    // このclassが使用するengine optionを追加する。
    void add_options(OptionsMap& options);

    // 今回の思考時間を決定する。
    // optimum(),maximum(),minimum()に反映する。
	// ⚠ optionsに"USI_Ponder"と"Stochastic_Ponder"オプションが含まれていること。
    void init(const Search::LimitsType& limits, Color us, int ply, const OptionsMap& options
#if STOCKFISH
			  , double& originalTimeAdjust
    // 💡 やねうら王では使わないことにする。
#else
			  , int max_moves_to_draw
#endif
    );

    TimePoint optimum() const;
    TimePoint maximum() const;
    TimePoint minimum() const;  // 📌 やねうら王独自追加。

    // 🌈 10-19 (report/59 §15.5): 動的係数の積 m を pivot × (m / pivot)^gamma に (TimeDynamicGamma / TimeDynamicPivot、既定 100 = そのまま)。
    double dynamic_shape(double m) const {
        if (dyn_gamma == 100 || !(m > 0.0))
            return m;
        const double p = dyn_pivot / 100.0;
        return p * std::pow(m / p, dyn_gamma / 100.0);
    }

    // 🌈 #96 時間配分 (report/59): 直前の探索の root 評価値 (内部 Value、手番視点、VALUE_INFINITE = 未知) を init() の前に渡す。
    //     TimeAllocMode > 0 のとき、|e| と手数から残り手数 R(ply,|e|) と将来の重み和 M(ply,|e|) を表 (timeman_tables.h) で見積もる。
    void set_prev_score(Value v) { prev_score = v; }
    // 🌈 #96: |e| >= TimeAllocBigEval のときの硬い下限 [ms] (0 = なし)。探索側は totalTime をこれ以上にする (動的係数で早く止まらない)。
    TimePoint hard_minimum() const { return hard_min; }

    // 🌈 floodgate 10-09 (report/59 §14): ponderhit 後の自分の時計の扱い (PonderhitThinkRatio) と、節約した時間の繰り越し (TimeCarrySpread)。
    //   elapsed_effective() = go からの経過時間 − (ponder 中の時間 × PonderhitThinkRatio / 100)。
    //   0 (既定) なら従来 (ponder 中の時間を全部「使った」とみなし、相手の手を当てると即指しになる)、
    //   100 なら ponderhit から改めて optimum 分だけ自分の時計で考える。optimum / maximum との比較はこの値で行う。
    TimePoint elapsed_effective() const {
        TimePoint credit = (ponderhitTime > startTime) ? (ponderhitTime - startTime) * ponder_think_ratio / 100 : 0;
        return elapsed_time() - credit;
    }
    int ponder_think_ratio_value() const { return ponder_think_ratio; }
    // 探索終了時 (bestmove の直前) に呼ぶ。still_pondering = "stop" で止められた go ponder (外れ) なら繰り越しに数えない。
    void note_search_end(bool still_pondering, bool use_time_management);
    TimePoint carry() const { return carry_ms; }

    // "go"からの経過時間を返す。
    template<typename FUNC>
    TimePoint elapsed(FUNC nodes) const {
#if STOCKFISH
        return useNodesTime ? TimePoint(nodes()) : elapsed_time();
#else
        // 📝 やねうら王ではNodesTimeを使わないのでelapsed()はそのままelapsed_Time()に委譲しておく。
        return elapsed_time();
#endif
    }

    // startTimeからの経過時間。
    // 💡 startTimeは、init()した時にLimitsType::startTimeがコピーされる。そこからの経過時間。
    TimePoint elapsed_time() const { return now() - startTime; };


	// 初期化。
    // 🌈　やねうら王では使わない。Stockfishとの互換性のために用意。
    void clear() { carry_ms = 0; base_optimum = 0; game_initial_time = 0; }   // 🌈 繰り越し (TimeCarrySpread) と対局開始時の持ち時間は対局ごとにリセット
    bool is_short_profile() const { return short_profile; }                 // 🌈 ShortClockMaxMs で短い側の設定が有効か (デバッグ表示用)

#if STOCKFISH

    void advance_nodes_time(std::int64_t nodes);
    // 💡 やねうら王ではNodesTimeを使わない。

#else

    // 📌 以下、やねうら王独自追加。

    // 探索終了の時刻
    /*
	  📓 startTimeからの経過時間がこの値以上になれば探索の終了時間。
		  0ならば終了時刻は未確定。init()で初期化される。

		  つまり、startTime + search_end <= now()になったら停止。
	      この条件式の左辺のstartTimeを右辺に移項させると、

		    search_end <= now() - startTime = elpased()

		  であり、つまり search_end <= elapsed()がその条件。

		  また、search_endはinit()のなかで 0 に初期化されているので、
		  search_endが非0であれば、終了が確定していてその時刻が設定されているという意味になる。

		  よって、
			  search_end && search_end <= elapsed()
		  という条件式が必要で、これはMainManager::check_time()に書いてある。
	*/
    TimePoint search_end;

    // 秒単位で切り上げる。ただし、NetworkDelayの値などを考慮する。
	// ⚠ round_up_to_fullsecondがfalseなら切り上げない
    TimePoint round_up(TimePoint t);

    // 探索を終了させることが確定しているが、秒単位で切り上げて、search_endにそれを設定したい時に呼び出す。
	// 💡 引数のeはelapsedTime。これはなくてもelapsedTime()を呼び出せばいいのだが、
	//     呼び出し元ですでに持っているので、二度elpasedTime()を呼び出すのは嫌だから引数で渡している。
    void set_search_end(TimePoint e);

    // ponderhitTimeからの経過時間。
    // 💡 ponderhitTimeは、init()した時にLimitsType::startTimeがコピーされる。
	//     "ponderhit"したときには、MainManager::set_ponderhit()でその時刻がコピーされる。
	//     よって、"go ponder"での思考でなくともこの変数は使える。
    TimePoint elapsed_time_from_ponderhit() const { return now() - ponderhitTime; };

#endif


#if STOCKFISH
   private:
    TimePoint startTime;      // 💡 探索開始時刻。LimitsType startTimeの値。
#else
    bool      isFinalPush;    // 🌈 秒読みに突入しているので持ち時間を使い切るべきであるフラグ。
							  //     つまりはponderhitTimeから数えてminimumTime分は使って欲しい。
    TimePoint startTime;      // 💡 探索開始時刻。LimitsType startTimeの値。
    TimePoint ponderhitTime;  // 🌈 "ponderhit"した時刻。startTimeからの経過時間ではなく、Timer::now()の生の値。
                              //     "ponderhit"するまではstartTimeと同じ値。
   private:
    TimePoint minimumTime;  // 🌈 やねうら王独自追加。
#endif

    TimePoint optimumTime;
    TimePoint maximumTime;

#if STOCKFISH

	std::int64_t availableNodes = -1;     // When in 'nodes as time' mode
    bool         useNodesTime   = false;  // True if we are in 'nodes as time' mode
    // 💡 やねうら王ではNodesTimeを使わない。

#else
    // 📌 以下、やねうら王独自追加。

    // init()の内部実装。
    void init_(const Search::LimitsType& limits,
               Color                     us,
               int                       ply,
               const OptionsMap&         options,
               int                       max_moves_to_draw);

    // optionsのそれぞれの値
    TimePoint minimum_thinking_time;
    TimePoint network_delay;
    TimePoint remain_time;
    bool      round_up_to_fullsecond;
    Value     prev_score = VALUE_INFINITE;  // 🌈 #96 時間配分: set_prev_score() で渡される直前の root 評価値
    TimePoint hard_min   = 0;               // 🌈 #96 時間配分: |e| 大のときの硬い下限 (init_() で毎回設定)
    int       ponder_think_ratio = 0;   // 🌈 PonderhitThinkRatio
    int       carry_spread       = 0;   // 🌈 TimeCarrySpread (0 = 繰り越し無し)
    int       carry_max_pct      = 50;  // 🌈 TimeCarryMax: carry の上限 (remain_time に対する %)
    TimePoint carry_ms           = 0;   // 🌈 節約して繰り越した時間 [ms] (対局中ずっと保持、clear() でリセット)
    TimePoint base_optimum       = 0;   // 🌈 今回の optimum (繰り越し加算前)。note_search_end() で使う
    TimePoint carry_bonus        = 0;   // 🌈 今回 optimum/maximum に足した carry の量 (デバッグ表示用)
    int       carry_hit_pct      = 0;   // 🌈 TimeCarryHitPct : go ponder で init された手 (当たれば hit) に配る carry の %
    int       carry_miss_pct     = 0;   // 🌈 TimeCarryMissPct: 通常の go (外れ / ponder 無し) の手に配る carry の %。両方 0 なら 1/N (TimeCarrySpread)
    int       ponder_optimum_bonus = 25;      // 🌈 PonderOptimumBonus: USI_Ponder 時に optimum へ足す % (従来は固定 25)
    int       dyn_gamma          = 100;       // 🌈 TimeDynamicGamma: 動的係数の積の指数 [%] (100 = 従来)
    int       dyn_pivot          = 100;       // 🌈 TimeDynamicPivot: その中心 [%] (M の幾何平均 ≈ 1.0)
    TimePoint game_initial_time  = 0;         // 🌈 対局の最初の init_ での自分の持ち時間 (本体 + 秒読み)、clear() で 0
    bool      short_profile      = false;     // 🌈 ShortClockMaxMs により短い側の設定を使っているか
    bool      ponderhit_min_from_hit = false; // 🌈 PonderhitMinFromHit: 最低思考時間を ponderhit から数える (相手の即指しにつられない)
    bool      carry_enabled() const { return carry_spread > 0 || carry_hit_pct > 0 || carry_miss_pct > 0; }

	// TODO : あとで
#if 0
    // 前回のinit()の値。
	// このあとround_up()で用いるので保存しておく。
    Search::LimitsType* lastcall_Limits;
    Color               lastcall_Us;
    int                 lastcall_Ply;
    OptionsMap*         lastcall_Opt;
#endif

#endif
};

}  // namespace YaneuraOu

#endif  // #ifndef TIMEMAN_H_INCLUDED
