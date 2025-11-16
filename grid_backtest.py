import sys
import datetime
import math
import tushare as ts
import os
import time

# 限速器：每分钟最多50次请求
class RateLimiter:
    def __init__(self, max_requests=50, time_window=60):
        self.max_requests = max_requests
        self.time_window = time_window
        self.requests = []
    
    def wait_if_needed(self):
        now = time.time()
        # 移除超过时间窗口的请求记录
        self.requests = [t for t in self.requests if now - t < self.time_window]
        
        # 如果请求次数达到限制，等待
        if len(self.requests) >= self.max_requests:
            oldest_request = min(self.requests)
            wait_time = self.time_window - (now - oldest_request) + 60  # 多等60秒确保安全
            if wait_time > 0:
                print(f"  达到API限速，等待 {wait_time:.1f} 秒...")
                time.sleep(wait_time)
                # 等待后重新清理请求记录
                now = time.time()
                self.requests = [t for t in self.requests if now - t < self.time_window]
        
        # 记录本次请求时间
        self.requests.append(time.time())

# 全局限速器实例
rate_limiter = RateLimiter(max_requests=50, time_window=60)

def read_token():
    f = open("token", "r")
    token = f.readline().strip()
    f.close()
    return token

def fetch_prices(pro, ts_code, start_date, end_date):
    rate_limiter.wait_if_needed()
    df = pro.daily(ts_code=ts_code, start_date=start_date, end_date=end_date)
    if df.shape[0] == 0:
        return []
    df = df.sort_values(by="trade_date")
    return [(row["trade_date"], float(row["close"])) for _, row in df.iterrows()]

def fetch_bars(ts_code, start_date, end_date, freq):
    if freq == "D":
        pro = ts.pro_api()
        return fetch_prices(pro, ts_code, start_date, end_date)
    rate_limiter.wait_if_needed()
    try:
        df = ts.pro_bar(ts_code=ts_code, start_date=start_date, end_date=end_date, freq=freq, asset="E")
    except Exception:
        return []
    if df is None or df.shape[0] == 0:
        return []
    col_time = "trade_time" if "trade_time" in df.columns else "trade_date"
    df = df.sort_values(by=col_time)
    vals = []
    for _, row in df.iterrows():
        t = row[col_time]
        c = float(row["close"])
        if isinstance(t, str):
            s = t
        else:
            s = str(t)
        vals.append((s, c))
    return vals

def fee_amount(qty, price, fee_rate, min_fee):
    v = qty * price * fee_rate
    return min_fee if v < min_fee else v

def simulate(prices, lower, upper, step, mode, batch_size, fee_rate, min_fee):
    if not prices:
        return {"roi": 0.0, "annualized_roi": 0.0, "trades": [], "equity": [], "capital": 0.0, "total_fee": 0.0, 
                "max_buy_qty": 0, "max_sell_qty": 0, "max_net_qty": 0}
    # 基准价初始化为第一个价格（限制在有效区间内）
    base_price = max(min(prices[0][1], upper), lower)
    cash = 0.0
    pos = 0
    trades = []
    cum = []
    min_cum = 0.0
    
    # 统计最大买入数量、最大卖出数量、最大买卖轧差数量
    total_buy_qty = 0
    total_sell_qty = 0
    max_buy_qty = 0
    max_sell_qty = 0
    max_net_qty = 0  # 最大净持仓（买入-卖出）
    
    for d, p in prices:
        if p < lower or p > upper:
            cum_val = cash + pos * p
            cum.append((d, cum_val))
            if cum_val < min_cum:
                min_cum = cum_val
            continue
        
        # 根据基准价计算触发条件（限价单：以当前市场价格成交）
        # 买入条件：当前价格相对于基准价下跌达到步长
        buy_cond = (p <= base_price * (1 - step)) if mode == "percent" else (p <= base_price - step)
        # 卖出条件：当前价格相对于基准价上涨达到步长
        sell_cond = (p >= base_price * (1 + step)) if mode == "percent" else (p >= base_price + step)
        
        if buy_cond:
            # 限价买入：以当前市场价格p成交
            cost = batch_size * p
            fee = fee_amount(batch_size, p, fee_rate, min_fee)
            cash -= cost + fee
            pos += batch_size
            # 委托全部成交后，基准价更新为成交均价（限价单以当前价格成交）
            base_price = p
            total_buy_qty += batch_size
            if total_buy_qty > max_buy_qty:
                max_buy_qty = total_buy_qty
            # 更新最大买卖轧差数量（当前净持仓）
            net_qty = total_buy_qty - total_sell_qty
            if abs(net_qty) > abs(max_net_qty):
                max_net_qty = net_qty
            trades.append({"date": d, "type": "buy", "price": p, "qty": batch_size, "fee": fee})
        elif sell_cond and pos >= batch_size:
            # 限价卖出：以当前市场价格p成交
            proceeds = batch_size * p
            fee = fee_amount(batch_size, p, fee_rate, min_fee)
            cash += proceeds - fee
            pos -= batch_size
            # 委托全部成交后，基准价更新为成交均价（限价单以当前价格成交）
            base_price = p
            total_sell_qty += batch_size
            if total_sell_qty > max_sell_qty:
                max_sell_qty = total_sell_qty
            # 更新最大买卖轧差数量（当前净持仓）
            net_qty = total_buy_qty - total_sell_qty
            if abs(net_qty) > abs(max_net_qty):
                max_net_qty = net_qty
            trades.append({"date": d, "type": "sell", "price": p, "qty": batch_size, "fee": fee})
        
        cum_val = cash + pos * p
        cum.append((d, cum_val))
        if cum_val < min_cum:
            min_cum = cum_val
    
    final_equity = cash + pos * prices[-1][1]
    capital_req = abs(min_cum) if min_cum < 0 else 0.0
    duration_days = len(prices)
    if capital_req <= 0:
        return {"roi": 0.0, "annualized_roi": 0.0, "trades": trades, "equity": cum, "capital": 0.0, "total_fee": sum(t["fee"] for t in trades),
                "max_buy_qty": max_buy_qty, "max_sell_qty": max_sell_qty, "max_net_qty": max_net_qty}
    roi = final_equity / capital_req - 1.0
    annualized = roi * (365.0 / max(1, duration_days))
    total_fee = sum(t["fee"] for t in trades)
    return {"roi": roi, "annualized_roi": annualized, "trades": trades, "equity": cum, "capital": capital_req, "total_fee": total_fee,
            "max_buy_qty": max_buy_qty, "max_sell_qty": max_sell_qty, "max_net_qty": max_net_qty}

def quantile_bounds(prices, q_low, q_high):
    xs = [p for _, p in prices]
    xs_sorted = sorted(xs)
    n = len(xs_sorted)
    def qpos(q):
        pos = q * (n - 1)
        lo = int(math.floor(pos))
        hi = int(math.ceil(pos))
        if lo == hi:
            return xs_sorted[lo]
        w = pos - lo
        return xs_sorted[lo] * (1 - w) + xs_sorted[hi] * w
    return qpos(q_low), qpos(q_high)

def search_best(prices, batch_sizes, fee_rate, min_fee):
    best = None
    modes = ["percent", "abs"]
    day_opts = [90, 180, 360]
    qpairs = [(0.1, 0.9), (0.15, 0.85), (0.2, 0.8), (0.25, 0.75)]
    perc_steps = [i / 1000.0 for i in range(5, 51, 5)]
    abs_steps = [i / 10.0 for i in range(1, 21)]
    for days in day_opts:
        sub = prices[-days:] if len(prices) > days else prices
        for ql, qh in qpairs:
            low, high = quantile_bounds(sub, ql, qh)
            for m in modes:
                steps = perc_steps if m == "percent" else abs_steps
                for s in steps:
                    for b in batch_sizes:
                        res = simulate(sub, low, high, s, m, b, fee_rate, min_fee)
                        score = res["annualized_roi"]
                        if best is None or score > best["annualized_roi"]:
                            best = {
                                "mode": m,
                                "step": s,
                                "lower": low,
                                "upper": high,
                                "days": len(sub),
                                "roi": res["roi"],
                                "annualized_roi": res["annualized_roi"],
                                "trades": res["trades"],
                                "equity": res["equity"],
                                "capital": res["capital"],
                                "batch_size": b,
                                "total_fee": res["total_fee"],
                                "max_buy_qty": res["max_buy_qty"],
                                "max_sell_qty": res["max_sell_qty"],
                                "max_net_qty": res["max_net_qty"],
                            }
    return best

def try_plot(ts_code, prices, best):
    try:
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates
        from datetime import datetime
    except Exception:
        return None
    def parse_dt(s):
        fs = ["%Y%m%d", "%Y-%m-%d", "%Y%m%d %H:%M:%S", "%Y-%m-%d %H:%M:%S"]
        for f in fs:
            try:
                return datetime.strptime(s, f)
            except Exception:
                pass
        return datetime.strptime(s[:8], "%Y%m%d")
    ds = [parse_dt(d) for d, _ in prices]
    ps = [p for _, p in prices]
    fig1 = plt.figure(figsize=(10, 4))
    plt.plot(ds, ps, label="price")
    plt.axhline(best["lower"], color="green")
    plt.axhline(best["upper"], color="red")
    bx = [t["date"] for t in best["trades"] if t["type"] == "buy"]
    bp = [t["price"] for t in best["trades"] if t["type"] == "buy"]
    sx = [t["date"] for t in best["trades"] if t["type"] == "sell"]
    sp = [t["price"] for t in best["trades"] if t["type"] == "sell"]
    bx_dt = [parse_dt(x) for x in bx]
    sx_dt = [parse_dt(x) for x in sx]
    plt.scatter(bx_dt, bp, color="green")
    plt.scatter(sx_dt, sp, color="red")
    locator = mdates.AutoDateLocator()
    formatter = mdates.ConciseDateFormatter(locator)
    ax = plt.gca()
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(formatter)
    plt.gcf().autofmt_xdate()
    plt.tight_layout()
    p1 = f"{ts_code}_grid_price.png"
    plt.savefig(p1)
    plt.close(fig1)
    fig2 = plt.figure(figsize=(10, 4))
    ed = [parse_dt(d) for d, _ in best["equity"]]
    ev = [v for _, v in best["equity"]]
    plt.plot(ed, ev)
    ax2 = plt.gca()
    ax2.xaxis.set_major_locator(locator)
    ax2.xaxis.set_major_formatter(formatter)
    plt.gcf().autofmt_xdate()
    plt.tight_layout()
    p2 = f"{ts_code}_grid_equity.png"
    plt.savefig(p2)
    plt.close(fig2)
    return (p1, p2)

def main():
    def read_codes(path):
        if not os.path.exists(path):
            return []
        with open(path, "r", encoding="utf-8") as f:
            txt = f.read()
        toks = txt.strip().split()
        return [t for t in toks if len(t) > 0]
    cfg_default = "grid_backtest.cfg"
    codes = []
    if len(sys.argv) >= 2:
        arg = sys.argv[1]
        if arg.endswith(".cfg"):
            codes = read_codes(arg)
        else:
            codes = [arg]
    else:
        codes = read_codes(cfg_default)
    if not codes:
        print("未提供股票代码或配置文件为空, 用法: python grid_backtest.py 600519.SH 或 grid_backtest.py grid_backtest.json")
        return
    token = read_token()
    ts.set_token(token)
    pro = ts.pro_api()
    today = datetime.datetime.today()
    end_date = today.strftime("%Y%m%d")
    start_date = (today - datetime.timedelta(days=800)).strftime("%Y%m%d")
    freq = "D"
    if len(sys.argv) >= 3:
        freq = sys.argv[2]
    fee_rate = 0.0001
    min_fee = 5.0
    batch_sizes = [100, 200, 300, 400, 500, 800, 1000, 1500, 2000, 3000]
    
    # 收集所有回测结果
    results = []
    print(f"开始回测 {len(codes)} 只股票...")
    for idx, ts_code in enumerate(codes, 1):
        print(f"[{idx}/{len(codes)}] 正在处理: {ts_code}")
        prices = fetch_bars(ts_code, start_date, end_date, freq)
        if not prices:
            continue
        best = search_best(prices, batch_sizes, fee_rate, min_fee)
        if best is None:
            continue
        trade_count = len(best["trades"])
        # 只收集交易次数 >= 20 的结果
        if trade_count >= 20:
            suitable = (best["annualized_roi"] > 0.08 and trade_count >= 6 and best["capital"] > 0)
            plots = try_plot(ts_code, prices[-best["days"]:], best)
            results.append({
                "ts_code": ts_code,
                "best": best,
                "trade_count": trade_count,
                "suitable": suitable,
                "plots": plots
            })
    
    # 按交易次数降序排序
    results.sort(key=lambda x: x["trade_count"], reverse=True)
    
    # 输出排序后的结果
    print("\n" + "="*80)
    print(f"回测完成！共找到 {len(results)} 只股票交易次数 >= 20 次")
    print("="*80 + "\n")
    
    for idx, result in enumerate(results, 1):
        ts_code = result["ts_code"]
        best = result["best"]
        trade_count = result["trade_count"]
        suitable = result["suitable"]
        plots = result["plots"]
        
        print(f"【排名 {idx}】股票代码: {ts_code}")
        print("推荐设置类型:", "百分比" if best["mode"] == "percent" else "价格差")
        print("上涨/下跌步长:", ("{:.3f}%".format(best["step"] * 100) if best["mode"] == "percent" else "{:.3f}".format(best["step"])))
        print("有效价格区间:", "{:.3f}".format(best["lower"]), "-", "{:.3f}".format(best["upper"]))
        print("每次交易股数:", best["batch_size"])
        print("回测天数:", best["days"])
        print("交易次数:", trade_count)
        print("资本占用峰值(元):", "{:.2f}".format(best["capital"]))
        print("最终资产(元):", "{:.2f}".format((best["roi"] + 1) * best["capital"]))
        print("收益率:", "{:.3f}".format(best["roi"]))
        print("年化收益率:", "{:.3f}".format(best["annualized_roi"]))
        print("总手续费(元):", "{:.2f}".format(best["total_fee"]))
        print("单次平均手续费(元):", "{:.2f}".format(best["total_fee"] / trade_count if trade_count > 0 else 0.0))
        print("最大买入数量(股):", best["max_buy_qty"])
        print("最大卖出数量(股):", best["max_sell_qty"])
        print("最大买卖轧差数量(股):", best["max_net_qty"])
        print("是否适合网格交易:", "是" if suitable else "否")
        if plots:
            print("价格与交易图:", plots[0])
            print("资产曲线图:", plots[1])
        print("-" * 80)

if __name__ == "__main__":
    main()