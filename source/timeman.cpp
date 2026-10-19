#include "config.h"

#if defined(USE_TIME_MANAGEMENT)

#include "misc.h"
#include "search.h"
#include "thread.h"
#include "usi.h"
#include "timeman.h"
#include "timeman_tables.h"  // 🌈 #96 時間配分の表
#include <cmath>

namespace YaneuraOu {

namespace {

// これぐらい自分が指すと終局すると考えて計画を練る。
// 近年、将棋ソフトは終局までの平均手数が伸びているので160に設定しておく。
const int MoveHorizon = 160;

// 思考時間のrtimeが指定されたときに用いる乱数
PRNG prng;

} // namespace

// 起動時に呼び出す。
// このclassが使用するengine optionを追加する。
void TimeManagement::add_options(OptionsMap& options) {

		// 指し手がGUIに届くまでの時間。
#if defined(YANEURAOU_ENGINE_DEEP)
    // GPUからの結果を待っている時間も込みなのでdefault値を少し上げておく。
    int time_margin = 400;
#else
    int time_margin = 120;
#endif

    // ネットワークの平均遅延時間[ms]
    // この時間だけ早めに指せばだいたい間に合う。
    // 切れ負けの瞬間は、NetworkDelayのほうなので大丈夫。
    options.add("NetworkDelay", Option(time_margin, 0, 10000));

    // ネットワークの最大遅延時間[ms]
    // 切れ負けの瞬間だけはこの時間だけ早めに指す。
    // 1.2秒ほど早く指さないとfloodgateで切れ負けしかねない。
    options.add("NetworkDelay2", Option(time_margin + 1000, 0, 10000));

    // 最小思考時間[ms]
    options.add("MinimumThinkingTime", Option(2000, 1, 100000));

    // 切れ負けのときの思考時間を調整する。序盤重視率。百分率になっている。
    // 例えば200を指定すると本来の最適時間の200%(2倍)思考するようになる。
    // 対人のときに短めに設定して強制的に早指しにすることが出来る。
    options.add("SlowMover", Option(100, 1, 1000));

	// 持ち時間、各秒のギリギリまで使うか。
    options.add("RoundUpToFullSecond", true);

    // 🌈 #96 時間配分 (report/59): 0 = 従来、1 = 残り手数を表 R(ply,|e|) で見積もる、2 = 1 + 勝率感度 w(e)^γ / M(ply,|e|) で配る
    options.add("TimeAllocMode", Option(0, 0, 2));
    options.add("TimeAllocGamma", Option(100, 0, 200));        // γ x100 (1.0 = 対数収益の最適、0 = 均等)
    options.add("TimeAllocBigEval", Option(1000, 0, 100000));  // |e| [cp] がこれ以上なら下限 TimeAllocBigFloor
    options.add("TimeAllocBigFloor", Option(3900, 0, 60000));  // [ms] 床 (1.9 s) の 2 倍
    options.add("TimeAllocScale", Option(100, 10, 400));       // optimum に掛ける百分率 (A/B 用)
    options.add("TimeAllocDebug", false);                      // init() ごとに配分の内部値を info string で出す
    // 🌈 #96: maximum = minimum + remain_estimate × (MaxTimeRatio/10) / MTG。SF/やねうら王の既定 5.0 (= 50)。定跡離脱直後の初手が TT 空で
    //     maximum まで延びる (後手で 40〜50 秒) ので、3.0 などを A/B する。全モード共通。
    options.add("MaxTimeRatio", Option(50, 10, 100));
    // 🌈 floodgate 10-09 (report/59 §14): ponderhit 後に自分の時計で考える割合と、節約分の繰り越し
    options.add("PonderhitThinkRatio", Option(0, 0, 100));   // 0 = 従来 (ponder 中の時間を使用済み扱い)、100 = ponderhit から改めて optimum 分考える
    options.add("TimeCarrySpread", Option(0, 0, 100));       // 0 = 無効。N なら繰り越し carry の 1/N を今回の optimum と maximum に足す
    options.add("TimeCarryMax", Option(50, 0, 100));         // carry の上限 (残り時間に対する %)
    // 🌈 10-14 (report/59 §15.2): carry を hit / 外れで配り分ける、最低思考を ponderhit から数える、USI_Ponder の前倒し率
    options.add("TimeCarryHitPct", Option(0, 0, 100));     // go ponder で init された手 (当たれば hit) に carry の何 % を足すか (0 = TimeCarrySpread の 1/N)
    options.add("TimeCarryMissPct", Option(0, 0, 100));    // 通常の go (外れ / ponder 無し) の手に carry の何 % を足すか
    options.add("PonderhitMinFromHit", false);             // true: 最低思考時間 (MinimumThinkingTime − NetworkDelay) を ponderhit から数える (floor 課金なら無料)
    options.add("PonderOptimumBonus", Option(25, 0, 100)); // USI_Ponder 時に optimum へ足す % (従来の固定 25。carry と二重になるので 0 も試す)
    // 🌈 10-19 (report/59 §15.5): 動的係数の積 M (fallingEval × reduction × bestMoveInstability × highBestMoveEffort) の振れ幅。
    //     M を pivot × (M / pivot)^(gamma/100) に置き換える。並べ方 (どの局面に厚く配るか) は変えず、どれだけ極端に配るかだけを変える。
    //     100 = 従来。pivot は M の幾何平均 (将棋のラベルで 1.02) の近くに置く。
    options.add("TimeDynamicGamma", Option(100, 25, 300));
    options.add("TimeDynamicPivot", Option(100, 10, 1000));
}

void TimeManagement::init(const Search::LimitsType& limits,
                          Color               us,
                          int                 ply,
                          const OptionsMap&   options
#if STOCKFISH
                          ,
                          double& originalTimeAdjust
// 💡 やねうら王では使わないことにする。
#else
                          ,
                          int max_moves_to_draw
#endif

) {

    // 📝 探索開始時刻をコピーしておく。
    //     以降、elapsed_time()は、ここからの経過時間を返す。
    startTime = limits.startTime;

	// TODO あとで
#if 0
    // reinit()が呼び出された時のために呼び出し条件を保存しておく。
    lastcall_Limits = &limits;
    lastcall_Us     = us;
    lastcall_Ply    = ply;
    lastcall_Opt    = const_cast<OptionsMap*>(&options);
#endif

    init_(limits, us, ply, options, max_moves_to_draw);
}

// 今回の思考時間を計算して、optimum(),maximum()が値をきちんと返せるようにする。
// これは探索の開始時に呼び出されて、今回の指し手のための思考時間を計算する。
// limitsで指定された条件に基いてうまく計算する。
// ply : ここまでの手数。平手の初期局面なら1。(0ではない)
void TimeManagement::init_(const Search::LimitsType& limits,
                           Color               us,
                           int                 ply,
                           const OptionsMap&   options, int max_moves_to_draw) {

#if STOCKFISH
	TimePoint npmsec = TimePoint(options["nodestime"]);
	// nodes as timeモード。やねうら王では用いない。

    // If we have no time, we don't need to fully initialize TM.
    // startTime is used by movetime and useNodesTime is used in elapsed calls.
    startTime    = limits.startTime;

	useNodesTime = npmsec != 0;

    if (limits.time[us] == 0)
        return;

	TimePoint npmsec = Options["nodestime"];

	// npmsecがUSI optionで指定されていれば、時間の代わりに、ここで指定されたnode数をベースに思考を行なう。
	// nodes per millisecondの意味。
	// nodes as timeモードで対局しなければならないなら、時間をノード数に変換して、
	// 持ち時間制御の計算式では、この結果の値を用いる。
	if (npmsec)
	{
		// ゲーム開始時に1回だけ
		if (!availableNodes)
			availableNodes = npmsec * limits.time[us];
		
		// ミリ秒をnode数に変換する
		limits.time[us] = TimePoint(availableNodes);
		for (auto c : COLOR)
		{
			limits.inc[c] *= npmsec;
			limits.byoyomi[c] *= npmsec;
		}
		limits.rtime *= npmsec;
		limits.npmsec = npmsec;

		// NetworkDelay , MinimumThinkingTimeなどもすべてnpmsecを掛け算しないといけないな…。
		// 1000で繰り上げる必要もあるしなー。これtime managementと極めて相性が悪いのでは。
	}
#endif

	// ネットワークのDelayを考慮して少し減らすべき。
	// かつ、minimumとmaximumは端数をなくすべき
    network_delay = (TimePoint) options["NetworkDelay"];

	// 秒未満を切り上げるのか
	round_up_to_fullsecond = options["RoundUpToFullSecond"];

	// 探索開始時刻と終了予定時刻。このタイミングで初期化しておく。
	// 終了時刻は0ならば未確定という意味である。
    startTime = ponderhitTime = limits.startTime;
    search_end                = 0;
    hard_min                  = 0;  // 🌈 #96
    ponder_think_ratio        = (int)options["PonderhitThinkRatio"];
    carry_spread              = (int)options["TimeCarrySpread"];
    carry_max_pct             = (int)options["TimeCarryMax"];
    carry_hit_pct             = (int)options["TimeCarryHitPct"];
    carry_miss_pct            = (int)options["TimeCarryMissPct"];
    ponderhit_min_from_hit    = (bool)options["PonderhitMinFromHit"];
    ponder_optimum_bonus      = (int)options["PonderOptimumBonus"];
    dyn_gamma                 = (int)options["TimeDynamicGamma"];
    dyn_pivot                 = (int)options["TimeDynamicPivot"];
    base_optimum              = 0;

	// 今回の最大残り時間(これを超えてはならない)
	// byoyomiとincの指定は残り時間にこの時点で加算して考える。
    remain_time =
      limits.time[us] + limits.byoyomi[us] + limits.inc[us] - (TimePoint) options["NetworkDelay2"];

	// remain_timeを0にすると時間切れのあと自爆するのでとりあえず100はあることにしておく。
	// round_up_to_fullsecond == falseのときは秒未満での戦いなので、1にしておく。
	remain_time = std::max(remain_time, (TimePoint) (round_up_to_fullsecond ? 100 : 1) );

	// 最小思考時間
    minimum_thinking_time = (TimePoint) options["MinimumThinkingTime"];

	// 序盤重視率
	// 　これはこんなパラメーターとして手で調整するべきではなく、探索パラメーターの一種として
	//   別の方法で調整すべき。ただ、対人でソフトに早指ししたいときには意味があるような…。
	int slowMover = (int)options["SlowMover"];

	if (limits.rtime)
	{
		// これが指定されているときは最小思考時間をランダム化する。
		// 連続自己対戦時に同じ進行になるのを回避するためのもの。
		// 終盤で大きく持ち時間を変えると、勝率が5割に寄ってしまうのでそれには注意。

		auto r = limits.rtime;
#if 1
		// 指し手が進むごとに減衰していく曲線にする。
		if (ply)
			r += (int)prng.rand((int)std::min(r * 0.5f, r * 10.0f / (ply)));
#endif

		remain_time = minimumTime = optimumTime = maximumTime = r;
		return;
	}

	// 時間固定モード
	// "go movetime 100"のようにして思考をさせた場合。
	if (limits.movetime)
	{
		remain_time = minimumTime = optimumTime = maximumTime = limits.movetime;
		return;
	}

	// 切れ負けであるか？
	bool time_forfeit = limits.inc[us] == 0 && limits.byoyomi[us] == 0;

	// 1. 切れ負けルールの時は、MoveHorizonを + 40して考える。
	// 2. ゲーム開始直後～40手目ぐらいまでは定跡で進むし、そこまで進まなかったとしても勝負どころはそこではないので
	// 　ゲーム開始直後～40手目付近のMoveHorizonは少し大きめに考える必要がある。逆に40手目以降、MoveHorizonは40ぐらい減らして考えていいと思う。
	// 3. 切れ負けでないなら、100手時点で残り60手ぐらいのつもりで指していいと思う。(これくらいしないと勝負どころすぎてからの持ち時間が余ってしまう..)
	// (現在の大会のフィッシャールールは15分+inctime5秒とか5分+inctime10秒そんな感じなので、160手目ぐらいで持ち時間使い切って問題ない)
	int move_horizon;
	if (time_forfeit)
		move_horizon = MoveHorizon + 40 - std::min(ply , 40);
	else
		// + 20は調整項
		move_horizon = MoveHorizon + 20 - std::min(ply , 80);

	// 残りの自分の手番の回数
	// ⇨　plyは平手の初期局面が1。256手ルールとして、max_game_ply == 256だから、256手目の局面においてply == 256
	// 　その1手前の局面においてply == 255。ply == 255 or 256のときにMTGが1にならないといけない。だから2足しておくのが正解。
    const int MTG = std::min(max_moves_to_draw - ply + 2, move_horizon) / 2;

	if (MTG <= 0)
	{
		// 本来、終局までの最大手数が指定されているわけだから、この条件で呼び出されるはずはないのだが…。
		sync_cout << "info string Error! : MaxMovesToDraw is too small." << sync_endl;
		// 事故防止のために何か設定はしておく。
        minimumTime = optimumTime = maximumTime = 500;
        return;
	}
	if (MTG == 1)
	{
		// この手番で終了なので使いきれば良い。
		minimumTime = optimumTime = maximumTime = remain_time;
		return;
	}

	// 🌈 #96 時間配分 (report/59、docs/superpowers/plans/2026-10-03-time-allocation-dr7.md)
	//   TimeAllocMode 1: 残り手数 MTG を棋譜から推定した表 R(ply,|e|) で置き換える (従来の MoveHorizon は互角で 20〜40%、決着局で 2〜3 倍大きい)
	//   TimeAllocMode 2: 1 に加えて、1 手の重み w = 4P(1-P) (P = σ(|e|/K)) を将来の重み和 M(ply,|e|) で割った比で配る (t ∝ w^γ、γ=TimeAllocGamma/100)
	//   |e| は直前の探索の root 評価値 (set_prev_score、cp)。|e| >= TimeAllocBigEval なら optimum の下限を TimeAllocBigFloor [ms] にする (事故防止)。
	//   表は電竜戦 DR5/DR6 + WCSC34〜36 (+ floodgate 上位) の棋譜から tools/time_alloc_tables.py で生成。
	int       mtg_alloc    = MTG;
	double    ta_share     = 0.0;   // 0 なら従来式 (remain_estimate / mtg_alloc)
	TimePoint ta_big_floor = 0;
	int       ta_scale     = (int)options["TimeAllocScale"];
	int       ta_dbg_e = -1; double ta_dbg_R = 0, ta_dbg_M = 0;
	{
		int ta_mode = (int)options["TimeAllocMode"];
		if (ta_mode > 0 && !time_forfeit)
		{
			using namespace TimeAllocTables;
			int e_cp = 0;
			if (prev_score != VALUE_INFINITE && prev_score != VALUE_NONE)
				e_cp = (std::abs(int(prev_score)) >= int(VALUE_MATE_IN_MAX_PLY)) ? 100000 : std::abs(USIEngine::to_cp(prev_score));
			int pi = 0; while (pi + 1 < N_PLY && ply  >= PLY_EDGES[pi + 1]) ++pi;
			int ei = 0; while (ei + 1 < N_ABS && e_cp >= ABS_EDGES[ei + 1]) ++ei;
			double R = R10[pi][ei] / 10.0;
			double M = M100[pi][ei] / 100.0;
			ta_dbg_e = e_cp; ta_dbg_R = R; ta_dbg_M = M;
			// MaxMovesToDraw までの残り手数は超えない
			int mtg_cap = std::max(1, (max_moves_to_draw - ply + 2) / 2);
			mtg_alloc   = std::clamp((int)std::lround(R), 1, mtg_cap);
			if (ta_mode >= 2)
			{
				double P     = 1.0 / (1.0 + std::exp(-double(e_cp) / K_CP));
				double w     = 4.0 * P * (1.0 - P);
				double gamma = (int)options["TimeAllocGamma"] / 100.0;
				// γ=1: w/M、γ=0: 1/R (均等)、その間は幾何補間
				ta_share = std::pow(std::max(w, 1e-6) / std::max(M, 1e-6), gamma) * std::pow(1.0 / std::max(R, 1.0), 1.0 - gamma);
				ta_share = std::min(ta_share, 1.0);
			}
			if (e_cp >= (int)options["TimeAllocBigEval"])
				ta_big_floor = hard_min = (TimePoint)(int)options["TimeAllocBigFloor"];
		}
	}

	// minimumとoptimumな時間を適当に計算する。

	{
		// 最小思考時間(これが1000より短く設定されることはないはず..)
		// round_up_to_fullsecond == falseのときは秒未満での戦いなので、1にしておく。
		minimumTime = std::max(minimum_thinking_time - network_delay,
							   (TimePoint)(round_up_to_fullsecond ? 1000 : 1));

		// 最適思考時間と、最大思考時間には、まずは上限値を設定しておく。
		optimumTime = maximumTime = remain_time;

		// optimumTime = min ( minimumTime + α     , remain_time)
		// maximumTime = min ( minimumTime + α * 5 , remain_time)
		// みたいな感じで考える

		// 残り手数において残り時間はあとどれくらいあるのか。
		TimePoint remain_estimate = limits.time[us]
			+ limits.inc[us] * mtg_alloc
			// 秒読み時間も残り手数に付随しているものとみなす。
			+ limits.byoyomi[us] * mtg_alloc;

		// 1秒ずつは絶対消費していくねんで！
		remain_estimate -= (mtg_alloc + 1) * 1000;
		remain_estimate = std::max(remain_estimate, TimePoint(0));

		// -- optimumTime
		TimePoint t1 = minimumTime + (ta_share > 0.0 ? (TimePoint)(remain_estimate * ta_share) : remain_estimate / mtg_alloc);
		if (ta_big_floor > 0)
			t1 = std::max(t1, ta_big_floor);  // 🌈 #96 |e| が大きい局面の下限

		// -- maximumTime

		// 5.0fでもうまくマネージメントできる。
		// 📝 上手くタイムマネージメントできない探索部なら3.0fぐらいにするのが無難。
		float max_ratio = (int)options["MaxTimeRatio"] / 10.0f;  // 🌈 #96 (既定 50 = 5.0)

		// 切れ負けルールにおいては、5分を切っていたら、このratioを抑制する。
		if (time_forfeit)
		{
			// 3分     : ratio = 3.0
			// 2分     : ratio = 2.0
			// 1分以下 : ratio = 1.0固定
			max_ratio = std::min(max_ratio, std::max(float(limits.time[us]) / (60 * 1000), 1.0f));
		}
		TimePoint t2 = minimumTime + (int)(remain_estimate * max_ratio / mtg_alloc);

		// ただしmaximumは残り時間の30%以上は使わないものとする。
		// optimumが超える分にはいい。それは残り手数が少ないときとかなので構わない。
		t2 = std::min(t2, (TimePoint)(remain_estimate * 0.3));

		// slowMoverは100分率で与えられていて、optimumTimeの係数として作用するものとする。
		optimumTime = std::min(t1, optimumTime) * slowMover / 100 * ta_scale / 100;
		maximumTime = std::min(t2, maximumTime);

		// Ponderが有効でStochastic_Ponderが無効の場合、
		// ponderhitすると時間が本来の予測より余っていくので思考時間を心持ち多めにとっておく。
        if (options["USI_Ponder"] && !options["Stochastic_Ponder"])
			optimumTime += optimumTime * ponder_optimum_bonus / 100;   // 🌈 PonderOptimumBonus (従来は固定 25%)

	}

	// 🌈 繰り越し (TimeCarrySpread): 前の手までに節約した carry_ms の 1/N を今回の optimum と maximum に足す。
	//     carry は note_search_end() で「optimum − 自分の時計で使った時間」を積んだもの (ponderhit の即指し、単一合法手、安定局面の早止めで溜まる)。
	base_optimum = optimumTime;
	carry_bonus  = 0;
	if (carry_enabled())
	{
		carry_ms = std::min(carry_ms, remain_time * carry_max_pct / 100);
		// 🌈 hit / 外れ別の配分 (TimeCarryHitPct / MissPct) があればそれを、無ければ 1/N (TimeCarrySpread)。
		//     go ponder で init() された手 = 当たれば hit の手、通常の go = 外れ (stop の後) か ponder 無しの手。
		//     外れの手 (想定外の手、ゼロから読む) に厚く配るのが狙い (L115: 従来は外れに 7.7 s、ratio 100 は 4.6 s しか使えなかった)。
		if (carry_hit_pct > 0 || carry_miss_pct > 0)
			carry_bonus = carry_ms * (limits.ponderMode ? carry_hit_pct : carry_miss_pct) / 100;
		else
			carry_bonus = carry_ms / carry_spread;
		optimumTime += carry_bonus;
		maximumTime += carry_bonus;
	}

	// 秒読みモードでかつ、持ち時間がないなら、最小思考時間も最大思考時間もその時間にしたほうが得
    isFinalPush = false;
	if (limits.byoyomi[us])
	{
		// 持ち時間が少ないなら(秒読み時間の1.2倍未満なら)、思考時間を使いきったほうが得
		// これには持ち時間がゼロのケースも含まれる。
        if (limits.time[us] < (int) (limits.byoyomi[us] * 1.2))
        {
            minimumTime = optimumTime = maximumTime = limits.byoyomi[us] + limits.time[us];
			// "ponderhit"の時刻から数えてminimumTime分は使って欲しい。
            isFinalPush                             = true;
        }
	}

	// 残り時間 - network_delay2よりは短くしないと切れ負けになる可能性が出てくる。
	if (options["TimeAllocDebug"])
		sync_cout << "info string TimeAlloc ply " << ply << " e_cp " << ta_dbg_e << " R " << ta_dbg_R << " M " << ta_dbg_M
		          << " share " << ta_share << " mtg " << mtg_alloc << " time " << limits.time[us] << " inc " << limits.inc[us]
		          << " opt " << optimumTime << " max " << maximumTime << " min " << minimumTime << " hardmin " << hard_min
		          << " carry " << carry_ms << " base_opt " << base_optimum << " pthink " << ponder_think_ratio
		          << " cbonus " << carry_bonus << " hitpct " << carry_hit_pct << " misspct " << carry_miss_pct << " pbonus " << ponder_optimum_bonus
		          << " pondermode " << limits.ponderMode << sync_endl;
	minimumTime = std::min(round_up(minimumTime), remain_time);
	optimumTime = std::min(         optimumTime , remain_time);
	maximumTime = std::min(round_up(maximumTime), remain_time);
}

// 1秒単位で繰り上げてdelayを引く。
// ただし、remain_timeよりは小さくなるように制限する。
TimePoint TimeManagement::round_up(TimePoint t0) {
	if (round_up_to_fullsecond)
	{
		// 📓 秒未満を切り上げる時

		// 1000で繰り上げる。Options["MinimalThinkingTime"]が最低値。
		auto t = std::max(((t0 + 999) / 1000) * 1000, minimum_thinking_time);

		// そこから、Options["NetworkDelay"]の値を引く
		t = t - network_delay;

		// これが元の値より小さいなら、もう1秒使わないともったいない。
		if (t < t0)
			t += 1000;

		// remain_timeを上回ってはならない。
		t = std::min(t, remain_time);
		return t;

	} else {
		// 📓 秒未満を切り上げない時

		// Options["MinimalThinkingTime"]が最低値。
		auto t = std::max(t0, minimum_thinking_time);

		// そこから、Options["NetworkDelay"]の値を引く
		t = t - network_delay;

		// remain_timeを上回ってはならない。
		t = std::min(t, remain_time);
		return t;
	}
};

// 探索を終了させることが確定しているが、秒単位で切り上げて、search_endにそれを設定したい時に呼び出す。
void TimeManagement::set_search_end(TimePoint e) {
    /*
		🤔 現在時刻から、ponderhitした時刻から計算して、秒単位で切り上げた時刻まで思考させたい。

		📓
		    1. 使用した時間の計測はponderhitした時刻からの経過時間としてなされるために、
			   ponderhitからの経過時間を秒単位で切り上げしたい。
		       そのためには、round_up(elapsed_from_ponderhit())のようにround upする必要がある。

			2. 一方、"go"した時刻から計算して、tm.minimum()の分は思考することを守らせたい。
			   しかし、ponderhitからの経過時間で切り上げはしたい。

			   しかし、秒読みで秒まで使い切りたいとき(isFinalPush)は、
			   "ponderhit"から数えてtm.minimum()の分は思考しないともったいない。
	*/

    // 1. ponderhitからの経過時間。(go ponder～ponderhitしていない場合は、単にgoからの経過時間)
    TimePoint t1 = e + startTime - ponderhitTime;

	// 2. "go"した時刻からminimum()を足して、ponderhitからの経過時間に変換したもの。
	//    ただし、finalpushであるなら、"ponderhit"した時刻から数える。
    // 🌈 PonderhitMinFromHit: 最低思考時間を ponderhit から数える (相手の即指しにつられて 0.01 秒で指すのを防ぐ。floor 課金なら 0.9 秒は無料)
    TimePoint t2 = (isFinalPush || ponderhit_min_from_hit) ? minimum() : minimum() + startTime - ponderhitTime;

	// t1,t2の大きいほうを秒単位で切り上げて、それをstartTimeからの経過時間に換算したもの。
    // 💡 search_endの値は、startTimeからの経過時間なので。
    search_end = round_up(std::max(t1, t2)) + ponderhitTime - startTime;
}

TimePoint TimeManagement::minimum() const { return minimumTime; }
TimePoint TimeManagement::optimum() const { return optimumTime; }
TimePoint TimeManagement::maximum() const { return maximumTime; }

// 🌈 探索終了時 (bestmove の直前) に呼ぶ: 今回の optimum (繰り越し前) と、自分の時計で実際に使った時間の差を carry_ms に積む。
//     自分の時計の消費 = ponderhit (していなければ go) からの経過時間。使い過ぎた手は carry を減らす (0 が下限)。
void TimeManagement::note_search_end(bool still_pondering, bool use_time_management) {
	if (!carry_enabled() || !use_time_management || still_pondering || base_optimum <= 0)
		return;
	TimePoint own_used = now() - ponderhitTime;
	carry_ms = std::max<TimePoint>(0, carry_ms + base_optimum - own_used);
}


} // namespace YaneuraOu

#endif
