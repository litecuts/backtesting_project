
import os, json, math, argparse, datetime as dt
import numpy as np, pandas as pd, matplotlib.pyplot as plt

def try_import_yf():
    try:
        import yfinance as yf
        return yf
    except Exception:
        return None

def download_prices_yf(ticker, start, end):
    yf = try_import_yf()
    if yf is None: return None
    try:
        df = yf.download(ticker, start=start, end=end, progress=False)
        if df is None or df.empty: return None
        df = df.reset_index().rename(columns={
            "Date":"date","Open":"open","High":"high","Low":"low","Close":"close","Adj Close":"adj_close","Volume":"volume"
        })
        if "close" not in df: df["close"]=df["adj_close"]
        return df[["date","open","high","low","close","volume"]]
    except Exception:
        return None

def generate_synth(start="2020-01-01", days=800, s0=100.0, mu=0.1, sigma=0.25, seed=7):
    rng=np.random.default_rng(seed); dtv=1/252
    shocks=rng.normal((mu-0.5*sigma**2)*dtv, sigma*(dtv**0.5), size=days)
    prices=[s0]
    for e in shocks: prices.append(prices[-1]*math.exp(e))
    dates=pd.bdate_range(start=start, periods=days+1)
    df=pd.DataFrame({"date":dates,"close":prices})
    df["open"]=df["close"].shift(1).fillna(df["close"])
    df["high"]=df[["open","close"]].max(axis=1)*(1+0.002)
    df["low"]=df[["open","close"]].min(axis=1)*(1-0.002)
    df["volume"]=(1e6+np.abs(rng.normal(0,2e5,size=len(df)))).astype(int)
    return df

def backtest_sma(df, fast=20, slow=50, fee_bps=5, initial=10000.0):
    df=df.copy(); df["date"]=pd.to_datetime(df["date"]); df=df.sort_values("date").reset_index(drop=True)
    df["fast_sma"]=df["close"].rolling(fast).mean()
    df["slow_sma"]=df["close"].rolling(slow).mean()
    df["signal"]=(df["fast_sma"]>df["slow_sma"]).astype(int)
    df["position"]=df["signal"].shift(1).fillna(0)
    df["trade"]=df["position"].diff().fillna(df["position"])
    df["ret"]=df["close"].pct_change().fillna(0.0)
    equity=[initial]; cash=initial; log=[]
    for i in range(1,len(df)):
        prev=equity[-1]; tr=df.loc[i,"trade"]
        if tr!=0:
            fee=prev*(fee_bps/10000.0); cash-=fee
            log.append({"date":str(df.loc[i,"date"].date()),"action":"BUY" if tr>0 else "SELL","fee":round(fee,2),"equity_before":round(prev,2)})
            prev=cash
        pos=int(df.loc[i-1,"position"]); r=df.loc[i,"ret"]
        cash = prev*(1+r) if pos==1 else prev
        equity.append(cash)
    df["equity"]=equity; df["equity_curve"]=df["equity"]/initial
    roll=df["equity_curve"].cummax(); dd=df["equity_curve"]/roll-1
    daily=df["equity_curve"].pct_change().fillna(0.0)
    vol=daily.std()*(252**0.5); sharpe=(daily.mean()*252)/(vol+1e-12)
    total=df["equity"].iloc[-1]/initial-1
    days=(df["date"].iloc[-1]-df["date"].iloc[0]).days; years=max(days/365.25,1e-9)
    cagr=(1+total)**(1/years)-1
    # trades
    trades=[]; inpos=False; ent=None
    for i in range(len(df)):
        if not inpos and df["trade"].iloc[i]>0: inpos=True; ent=df["equity"].iloc[i]
        elif inpos and df["trade"].iloc[i]<0: trades.append(df["equity"].iloc[i]/ent-1); inpos=False
    win=(np.array(trades)>0).mean() if trades else 0.0
    avg=float(np.mean(trades)) if trades else 0.0
    metrics={"strategy":"SMA_Crossover","fast_sma":fast,"slow_sma":slow,"fee_bps":fee_bps,"initial_cash":initial,
             "total_return_pct":round(total*100,2),"CAGR_pct":round(cagr*100,2),
             "annual_volatility_pct":round(vol*100,2),"sharpe_ratio":round(sharpe,2),
             "max_drawdown_pct":round(float(dd.min())*100,2),"num_trades":int(len(trades)),
             "win_rate_pct":round(float(win)*100,2),"avg_trade_return_pct":round(avg*100,2),"backtest_days":int(days)}
    return df, metrics, pd.DataFrame(log)

def save_outputs(df, metrics, log, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    open(os.path.join(out_dir,"metrics.json"),"w").write(json.dumps(metrics, indent=2))
    log.to_csv(os.path.join(out_dir,"trade_log.csv"), index=False)
    # price+signals
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10,6))
    plt.plot(df["date"], df["close"], label="Close")
    plt.plot(df["date"], df["fast_sma"], label="Fast SMA")
    plt.plot(df["date"], df["slow_sma"], label="Slow SMA")
    buys=df[df["trade"]>0]; sells=df[df["trade"]<0]
    plt.scatter(buys["date"], buys["close"], marker="^", label="Buy")
    plt.scatter(sells["date"], sells["close"], marker="v", label="Sell")
    plt.title("Price with SMA Crossover Signals"); plt.xlabel("Date"); plt.ylabel("Price"); plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(out_dir,"price_signals.png")); plt.close()
    # equity curve
    plt.figure(figsize=(10,6))
    plt.plot(df["date"], df["equity_curve"], label="Equity Curve")
    plt.title("Strategy Equity Curve"); plt.xlabel("Date"); plt.ylabel("Equity (normalized)"); plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(out_dir,"equity_curve.png")); plt.close()

def main():
    import datetime as dt, argparse
    p=argparse.ArgumentParser("Backtest on real Yahoo Finance data (falls back to synthetic).")
    p.add_argument("--ticker", type=str, default="AAPL")
    p.add_argument("--start", type=str, default="2020-01-01")
    p.add_argument("--end", type=str, default=None)
    p.add_argument("--fast", type=int, default=20)
    p.add_argument("--slow", type=int, default=50)
    p.add_argument("--fee_bps", type=float, default=5.0)
    p.add_argument("--initial_cash", type=float, default=10000.0)
    p.add_argument("--out_dir", type=str, default="outputs")
    a=p.parse_args()
    end=a.end or dt.date.today().isoformat()
    df=download_prices_yf(a.ticker, a.start, end)
    if df is None or df.empty:
        print("⚠️ Yahoo Finance download failed. Using synthetic data instead.")
        df=generate_synth(start=a.start)
    df, metrics, log = backtest_sma(df, a.fast, a.slow, a.fee_bps, a.initial_cash)
    save_outputs(df, metrics, log, a.out_dir)
    print(json.dumps(metrics, indent=2))
    print("Outputs saved in:", os.path.abspath(a.out_dir))

if __name__=="__main__":
    main()
