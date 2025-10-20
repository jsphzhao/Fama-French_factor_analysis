
import argparse
import io
import re
import zipfile
import requests
import pandas as pd
import statsmodels.api as sm
import yfinance as yf
from pathlib import Path

FRENCH_3F_URL = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_Factors_CSV.zip"

def load_ff3_monthly() -> pd.DataFrame:
    """
    Robustly download & parse Fama-French 3 Factors (Monthly).
    Handles varying headers/footers by detecting the first YYYYMM row
    and stopping before 'Annual Factors' or the first blank line after the table.
    Returns monthly percent values with a DatetimeIndex at month-end.
    """
    resp = requests.get(
        FRENCH_3F_URL,
        headers={"User-Agent": "Mozilla/5.0"},
        timeout=60
    )
    resp.raise_for_status()

    with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
        name = next(n for n in zf.namelist() if n.lower().endswith(".csv"))
        raw = zf.read(name).decode("latin-1", errors="ignore") 

    lines = raw.splitlines()

    start = None
    for i, line in enumerate(lines):
        if re.match(r"^\s*\d{6}\s*,", line):
            start = i
            break
    if start is None:
        raise ValueError("Could not find data start in Fama-French CSV (no YYYYMM line).")

    end = None
    for i in range(start + 1, len(lines)):
        if lines[i].strip().startswith("Annual Factors"):
            end = i
            break
        if lines[i].strip() == "" and i > start + 5:
            end = i
            break
    if end is None:
        end = len(lines)

    header = "Date,Mkt-RF,SMB,HML,RF"
    csv_text = header + "\n" + "\n".join(lines[start:end])

    df = pd.read_csv(io.StringIO(csv_text))
    df["Date"] = pd.to_datetime(df["Date"].astype(str), format="%Y%m")
    df.set_index("Date", inplace=True)
    df = df[["Mkt-RF", "SMB", "HML", "RF"]].apply(pd.to_numeric, errors="coerce").dropna()
    return df

def load_prices(tickers, start, end):
    data = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)
    if isinstance(data.columns, pd.MultiIndex):
        adj = data["Close"] if "Close" in data.columns.levels[0] else data["Adj Close"]
    else:
        adj = data["Close"] if "Close" in data.columns else data["Adj Close"]
    return adj.dropna(how="all")

def to_returns(prices: pd.DataFrame, freq: str = "M") -> pd.DataFrame:
    if freq.upper() == "M":
        prices = prices.resample("M").last()
    return prices.pct_change() * 100.0

def run(tickers, start, end, freq, outdir="data"):
    base = Path(outdir)
    base.mkdir(parents=True, exist_ok=True)

    i = 1
    while (base / f"output{i}").exists():
        i += 1

    out = base / f"output{i}"
    out.mkdir()
    print(f"[INFO] Saving results to: {out}")


    print("[1/4] Fetching Fama-French 3 factors (monthly)...")
    ff = load_ff3_monthly()
    print(ff.head())

    print(f"[2/4] Downloading prices: {', '.join(tickers)}")
    prices = load_prices(tickers, start, end)
    print(prices.tail())

    print(f"[3/4] Computing {freq}-frequency returns (%)...")
    rets = to_returns(prices, freq=freq).dropna(how="all")
    if freq.upper() == "M":
        rets.index = rets.index.to_period("M").to_timestamp("M")
        ff.index = ff.index.to_period("M").to_timestamp("M")

    rows = []
    for t in (rets.columns if hasattr(rets, "columns") else [rets.name]):
        series = rets[t] if hasattr(rets, "columns") else rets
        df = pd.concat([series.rename("Ri"), ff], axis=1).dropna()
        if df.empty:
            print(f"Skipping {t}: no overlap.")
            continue
        df["Excess"] = df["Ri"] - df["RF"]
        X = sm.add_constant(df[["Mkt-RF","SMB","HML"]])
        y = df["Excess"]
        model = sm.OLS(y, X).fit()

        rows.append({
            "Ticker": t,
            "Alpha(%)": model.params.get("const", float("nan")),
            "Beta_Mkt": model.params.get("Mkt-RF", float("nan")),
            "Beta_SMB": model.params.get("SMB", float("nan")),
            "Beta_HML": model.params.get("HML", float("nan")),
            "R2": model.rsquared,
            "N": int(model.nobs),
        })

        # Save chart
        df["Predicted"] = model.predict(X)
        ax = df[["Excess","Predicted"]].plot(title=f"Actual vs Predicted Excess Returns - {t}")
        fig = ax.get_figure()
        fig.tight_layout()
        fig.savefig(out / f"{t}_actual_vs_predicted.png")
        fig.clf()

        # Save summary
        with open(out / f"{t}_ols_summary.txt","w") as f:
            f.write(model.summary().as_text())

        print(f"{t}: R^2={model.rsquared:.3f}, N={int(model.nobs)}")

    if rows:
        pd.DataFrame(rows).to_csv(out / "factor_loadings.csv", index=False)
        print(f"[4/4] Saved results: {out/'factor_loadings.csv'}")
    else:
        print("No results saved. Check tickers and dates.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fama-French regression for one or more tickers")
    parser.add_argument("--tickers", nargs="+", required=True)
    parser.add_argument("--start", type=str, default="2015-01-01")
    parser.add_argument("--end", type=str, default=None)
    parser.add_argument("--freq", type=str, default="M", choices=["D","M"])
    parser.add_argument("--outdir", type=str, default="data")
    args = parser.parse_args()
    run(args.tickers, args.start, args.end, args.freq, args.outdir)
