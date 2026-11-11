"""Pipeline: latih Baseline, Linear Regression, Random Forest, Gradient Boosting
lalu ekspor hasil ke site/data.js untuk dashboard.

Jalankan:  python pipeline.py [path_xlsx]
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import shutil

SRC = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("dataset_siap_modeling.xlsx")
NEW_SRC = Path("dataset_siap_modeling_baru.xlsx")
OUT = Path(__file__).parent / "site" / "data.js"
SEED = 42

# jumlah_barang (keputusan pemilik) sengaja TIDAK dipakai sebagai fitur (sesuai kamus data)
FEATURES = ["bulan", "hari_dropping", "wil_luar_kota", "jb_lag1", "terjual_lag1",
            "terjual_lag2", "terjual_roll3", "rate_lag1", "habis_lag1"]
MODEL_SHEETS = {
    "Linear Regression": "model_lr",
    "Random Forest": "model_rf",
    "Gradient Boosting": "model_gb",
}
MODEL_FEATURES = {
    "Baseline": [],
    "Linear Regression": ["wil_luar_kota", "jb_lag1", "terjual_lag1", "terjual_lag2", "terjual_roll3", "rate_lag1"],
    "Random Forest": ["hari_dropping", "wil_luar_kota", "jb_lag1", "terjual_lag1", "terjual_lag2", "terjual_roll3", "rate_lag1", "habis_lag1"],
    "Gradient Boosting": ["bulan", "hari_dropping", "wil_luar_kota", "jb_lag1", "terjual_lag1", "terjual_lag2", "terjual_roll3", "rate_lag1", "habis_lag1"],
}
TARGET = "terjual"
SAMPLE_OUTLETS = [
    "ALZA BAI", "IBU YANTI", "PAK ALI BAI", "PT.PASIR", "QUEEN",
    "TOKO SRI AHMAD", "AGENG CELL", "BUDE TITI", "LAUTAN JAYA",
    "NTM MART", "PRATIWI", "TOKO AKI", "TOKO TOLI", "WARUNG MARIAM",
    "ACUN KP", "HANAFI", "IBU SAS", "KAK YANI",
]
MODELS = {
    "Baseline": None,
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(n_estimators=300, min_samples_leaf=3, random_state=SEED, n_jobs=-1),
    "Gradient Boosting": GradientBoostingRegressor(n_estimators=300, learning_rate=0.05, max_depth=3,
                                                   subsample=0.8, random_state=SEED),
}


def r(x, n=3):
    return None if x is None or (isinstance(x, float) and np.isnan(x)) else round(float(x), n)


def metrics(y, p):
    mape_mask = y > 0
    return {
        "MAE": r(mean_absolute_error(y, p)),
        "RMSE": r(np.sqrt(mean_squared_error(y, p))),
        "R2": r(r2_score(y, p), 4),
        "MAPE": r(np.mean(np.abs((y[mape_mask] - p[mape_mask]) / y[mape_mask])) * 100, 2),
    }


def hist(series, bins):
    c, e = np.histogram(series, bins=bins)
    return {"labels": [f"{int(e[i])}-{int(e[i + 1])}" for i in range(len(c))], "counts": c.tolist()}


def main():
    if NEW_SRC.exists():
        try:
            shutil.copyfile(NEW_SRC, SRC)
            NEW_SRC.unlink()
        except PermissionError:
            pass
    active_src = NEW_SRC if NEW_SRC.exists() else SRC
    xl = pd.read_excel(active_src, sheet_name=None)
    df_clean = xl["dataset_model"].copy()
    df_clean["tanggal_dropping"] = pd.to_datetime(df_clean["tanggal_dropping"])

    # Gabungkan kolom bersih (dataset_model) dengan fitur lengkap (model_gb) untuk keperluan EDA & audit
    if "model_gb" in xl:
        df_gb = xl["model_gb"].copy()
        df_gb["tanggal_dropping"] = pd.to_datetime(df_gb["tanggal_dropping"])
        extra_cols = [c for c in df_gb.columns if c not in df_clean.columns]
        df = pd.concat([df_clean, df_gb[extra_cols]], axis=1)
        df["rate"] = df["terjual"] / df["jumlah_barang"]
        df["habis"] = (df["sisa"] == 0).astype(int)
        df["jeda_hari"] = df.groupby("outlet_id")["tanggal_dropping"].diff().dt.days
    else:
        df = df_clean
    kamus = xl["Kamus_Data"]

    # ---------- ringkasan preprocess (BAB III) ----------
    hb_df = xl["HB_terpisah"] if "HB_terpisah" in xl else xl["dihapus"][xl["dihapus"]["nama_outlet"] == "HB"]
    hb_n_rows = len(hb_df)
    n_extra_hb = len(xl["HB_terpisah"]) if "HB_terpisah" in xl else 0

    prep = {
        "mentah": len(df) + n_extra_hb + len(xl["dihapus"]),
        "model_utama": len(df),
        "hb_terpisah": hb_n_rows,
        "dihapus": len(xl["dihapus"]),
        "alasan_dihapus": xl["dihapus"]["alasan"].value_counts().to_dict() if "alasan" in xl["dihapus"] and len(xl["dihapus"]) > 0 else {},
        "outlet_semua": int(xl["peta_outlet_id"].shape[0]),
        "outlet_model": int(df.outlet_id.nunique()),
        "missing": {c: int(v) for c, v in df[FEATURES].isna().sum().items() if v > 0},
        "train_all": int((df.split == "train").sum()),
        "test_all": int((df.split == "test").sum()),
    }
    peta = xl["peta_outlet_id"]
    prep["outlet_dihapus"] = int(peta["status"].str.contains("dihapus").sum())
    prep["outlet_hb"] = int((peta["status"].str.contains("HB") | (peta["nama_outlet"] == "HB")).sum())
    prep["outlet_terpilih"] = int(prep["outlet_model"])

    # ---------- OUTLIER (outlet HB ikut ditampilkan sebagai pembanding) ----------
    per = df.groupby(["outlet_id", "nama_outlet"]).agg(
        jb=("jumlah_barang", "mean"), tj=("terjual", "mean")).reset_index()
    hb_per = hb_df.groupby(["outlet_id", "nama_outlet"]).agg(
        jb=("jumlah_barang", "mean"), tj=("terjual", "mean")).reset_index()
    q1, q3 = per.tj.quantile(0.25), per.tj.quantile(0.75)
    fence = q3 + 1.5 * (q3 - q1)
    allp = pd.concat([per.assign(hb=0), hb_per.assign(hb=1)], ignore_index=True)
    top = allp.sort_values("tj", ascending=False).head(10)
    outlier = {
        "model": [[r(a, 1), r(b, 1)] for a, b in per[["jb", "tj"]].values],
        "hb": [[n, r(a, 1), r(b, 1)] for n, a, b in hb_per[["nama_outlet", "jb", "tj"]].values],
        "stats": {"median": r(per.tj.median(), 1), "q1": r(q1, 1), "q3": r(q3, 1),
                  "fence": r(fence, 1), "max_model": r(per.tj.max(), 1),
                  "hb_mean": r(hb_per.tj.mean(), 1) if len(hb_per) else None,
                  "hb_ratio": r(hb_per.tj.mean() / per.tj.median(), 1) if len(hb_per) else None,
                  "n_model_over_fence": int((per.tj > fence).sum()), "n_model": int(len(per))},
        "top": {"labels": top.nama_outlet.tolist(), "values": [r(v, 1) for v in top.tj],
                "is_hb": top.hb.astype(int).tolist()},
    }

    # ---------- split ----------
    full = df[df.fitur_lengkap == 1]
    tr, te = full[full.split == "train"], full[full.split == "test"]
    prep["train_lengkap"], prep["test_lengkap"] = len(tr), len(te)
    ytr, yte = tr[TARGET].values, te[TARGET].values

    # ---------- EDA (BAB II) ----------
    weekly = df.groupby(df.tanggal_dropping.dt.to_period("W-SUN").dt.start_time).agg(
        terjual=("terjual", "sum"), jumlah_barang=("jumlah_barang", "sum"), n=("outlet_id", "count"))
    wil = df.groupby("wilayah").agg(n=("outlet_id", "count"), outlet=("outlet_id", "nunique"),
                                    terjual=("terjual", "mean"), jb=("jumlah_barang", "mean"),
                                    rate=("rate", "mean"), habis=("habis", "mean"))
    hari = df.groupby("hari_dropping").agg(n=("terjual", "count"), terjual=("terjual", "mean"))
    corr = df[FEATURES + [TARGET]].corr()[TARGET].drop(TARGET).sort_values()
    eda = {
        "describe": {c: {k: r(v) for k, v in df[c].describe().items()} for c in
                     ["terjual", "jumlah_barang", "rate", "sisa", "jeda_hari"]},
        "hist_terjual": hist(df.terjual, [0, 10, 20, 30, 40, 50, 60, 80, 100, 180]),
        "hist_rate": hist(df.rate * 100, [0, 20, 40, 50, 60, 70, 80, 90, 101]),
        "weekly": {"labels": [d.strftime("%d %b") for d in weekly.index],
                   "terjual": weekly.terjual.tolist(), "jumlah_barang": weekly.jumlah_barang.tolist(),
                   "n": weekly.n.tolist()},
        "wilayah": {k: {kk: r(vv) for kk, vv in v.items()} for k, v in wil.to_dict("index").items()},
        "hari": {"labels": ["Sen", "Sel", "Rab", "Kam", "Jum", "Sab", "Min"],
                 "n": [int(hari.n.get(i, 0)) for i in range(7)],
                 "terjual": [r(hari.terjual.get(i, 0), 2) for i in range(7)]},
        "corr": {"labels": corr.index.tolist(), "values": [r(v) for v in corr.values]},
        "scatter_jb": [[int(a), int(b)] for a, b in
                       df[["jumlah_barang", "terjual"]].sample(700, random_state=SEED).values],
        "habis": int(df.habis.sum()),
    }

    # ---------- model per sheet (BAB IV) & evaluasi (BAB V) ----------
    preds, res, importance, fitted = {}, {}, {}, {}
    # Baseline: Rata-rata historis (Mean Statis) per outlet dari data latih (benchmark standar data mining)
    outlet_train_mean = tr.groupby("outlet_id")[TARGET].mean()
    global_train_mean = float(tr[TARGET].mean())
    base_mean_test = te["outlet_id"].map(outlet_train_mean).fillna(global_train_mean).values
    base_mean_train = tr["outlet_id"].map(outlet_train_mean).fillna(global_train_mean).values
    preds["Baseline"] = base_mean_test
    base_roll3 = te["terjual_roll3"].values
    base_lag1 = te["terjual_lag1"].values
    for name, m in MODELS.items():
        if m is None:
            continue
        feats = MODEL_FEATURES[name]
        sheet_name = MODEL_SHEETS[name]
        if sheet_name in xl:
            s_df = xl[sheet_name]
            s_full = s_df[s_df.fitur_lengkap == 1]
            s_tr, s_te = s_full[s_full.split == "train"], s_full[s_full.split == "test"]
            Xtr_m, ytr_m = s_tr[feats], s_tr[TARGET].values
            Xte_m = s_te[feats]
        else:
            Xtr_m, ytr_m, Xte_m = tr[feats], ytr, te[feats]
        m.fit(Xtr_m, ytr_m)
        preds[name] = m.predict(Xte_m)
        fitted[name] = m

    importance["Random Forest"] = dict(zip(MODEL_FEATURES["Random Forest"], map(r, fitted["Random Forest"].feature_importances_)))
    importance["Gradient Boosting"] = dict(zip(MODEL_FEATURES["Gradient Boosting"], map(r, fitted["Gradient Boosting"].feature_importances_)))
    importance["Linear Regression"] = dict(zip(MODEL_FEATURES["Linear Regression"], map(r, fitted["Linear Regression"].coef_)))
    lr_int = r(fitted["Linear Regression"].intercept_)

    for name, p in preds.items():
        m = metrics(yte, p)
        feats = MODEL_FEATURES[name]
        m["MAE_train"] = (r(mean_absolute_error(ytr, base_mean_train)) if name == "Baseline"
                          else r(mean_absolute_error(ytr, fitted[name].predict(tr[feats]))))
        err = yte - p
        m["bias"] = r(err.mean())
        m["within5"] = r((np.abs(err) <= 5).mean() * 100, 1)
        m["within10"] = r((np.abs(err) <= 10).mean() * 100, 1)
        res[name] = m
    res["Baseline (roll-3)"] = metrics(yte, base_roll3)
    res["Baseline (roll-3)"].update(MAE_train=r(mean_absolute_error(ytr, tr["terjual_roll3"])),
                                    bias=r((yte - base_roll3).mean()),
                                    within5=r((np.abs(yte - base_roll3) <= 5).mean() * 100, 1),
                                    within10=r((np.abs(yte - base_roll3) <= 10).mean() * 100, 1))
    res["Baseline (lag-1)"] = metrics(yte, base_lag1)
    res["Baseline (lag-1)"].update(MAE_train=None, bias=r((yte - base_lag1).mean()),
                                   within5=r((np.abs(yte - base_lag1) <= 5).mean() * 100, 1),
                                   within10=r((np.abs(yte - base_lag1) <= 10).mean() * 100, 1))

    best = min((k for k in res if k in MODELS), key=lambda k: res[k]["MAE"])
    for k in res:
        res[k]["improve_vs_baseline"] = r((1 - res[k]["MAE"] / res["Baseline"]["MAE"]) * 100, 2)

    # per wilayah & per minggu test
    te = te.assign(**{f"p_{k}": v for k, v in preds.items()})
    by_wil = {}
    for w, g in te.groupby("wilayah"):
        by_wil[w] = {k: r(mean_absolute_error(g[TARGET], g[f"p_{k}"])) for k in preds}
    by_week = {}
    for wk, g in te.groupby("minggu_ke"):
        by_week[int(wk)] = {k: r(mean_absolute_error(g[TARGET], g[f"p_{k}"])) for k in preds}
    # per rentang volume penjualan
    bins = [0, 20, 30, 45, 1000]
    lab = ["<=20", "21-30", "31-45", ">45"]
    te["vol"] = pd.cut(te[TARGET], bins=bins, labels=lab)
    by_vol = {l: {k: r(mean_absolute_error(g[TARGET], g[f"p_{k}"])) for k in preds}
              for l, g in te.groupby("vol", observed=True)}

    # scatter & residual
    idx = np.random.RandomState(SEED).choice(len(te), 400, replace=False)
    scatter = {k: [[int(yte[i]), r(preds[k][i], 1)] for i in idx] for k in preds}
    rhist = {}
    edges = np.arange(-40, 45, 5)
    for k, p in preds.items():
        c, _ = np.histogram(np.clip(yte - p, -40, 40), bins=edges)
        rhist[k] = c.tolist()
    rlabels = [f"{int(edges[i])}" for i in range(len(edges) - 1)]

    # rata-rata aktual vs prediksi per tanggal
    daily = te.groupby("tanggal_dropping").agg(aktual=(TARGET, "sum"),
                                              **{k: (f"p_{k}", "sum") for k in preds})
    daily_json = {"labels": [d.strftime("%d %b") for d in daily.index], "Aktual": daily.aktual.round(1).tolist()}
    for k in preds:
        daily_json[k] = daily[k].round(1).tolist()

    # ---------- SAMPEL 18 OUTLET DATA ASLI (BAB VI - penerapan & halaman Prediksi) ----------
    # Rekonstruksi fitur lag untuk 2 outlet sampel (IBU YANTI & TOKO AKI) yang berada di sheet `dihapus`
    dih = xl["dihapus"].copy()
    dih["tanggal_dropping"] = pd.to_datetime(dih["tanggal_dropping"])
    extra_frames = []
    for s_name, target_dow in [("IBU YANTI", 4), ("TOKO AKI", 3)]:
        g_dih = dih[(dih["nama_outlet"] == s_name) & (dih["tanggal_dropping"].dt.dayofweek == target_dow)].sort_values("tanggal_dropping").copy()
        if len(g_dih) == 0:
            continue
        g_dih["bulan"] = g_dih["tanggal_dropping"].dt.month
        g_dih["hari_dropping"] = g_dih["tanggal_dropping"].dt.dayofweek
        g_dih["urutan_minggu"] = range(1, len(g_dih) + 1)
        g_dih["jeda_hari"] = g_dih["tanggal_dropping"].diff().dt.days
        g_dih["n_jeda_panjang"] = int((g_dih["jeda_hari"] > 8).sum())
        g_dih["jadwal_mingguan"] = int(g_dih["n_jeda_panjang"].iloc[0] == 0)
        g_dih["rate"] = g_dih["terjual"] / g_dih["jumlah_barang"]
        g_dih["habis"] = (g_dih["sisa"] == 0).astype(int)
        g_dih["wil_luar_kota"] = (g_dih["wilayah"] == "LUAR KOTA").astype(int)
        g_dih["jb_lag1"] = g_dih["jumlah_barang"].shift(1)
        g_dih["terjual_lag1"] = g_dih["terjual"].shift(1)
        g_dih["terjual_lag2"] = g_dih["terjual"].shift(2)
        g_dih["terjual_roll3"] = g_dih["terjual"].shift(1).rolling(3, min_periods=1).mean()
        g_dih["rate_lag1"] = g_dih["rate"].shift(1)
        g_dih["habis_lag1"] = g_dih["habis"].shift(1)
        g_dih["split"] = np.where(g_dih["tanggal_dropping"] < pd.Timestamp("2026-08-01"), "train", "test")
        g_dih["fitur_lengkap"] = (g_dih["urutan_minggu"] >= 3).astype(int)
        extra_frames.append(g_dih[df.columns])

    df_sample_all = pd.concat([df[df["nama_outlet"].isin(SAMPLE_OUTLETS)]] + extra_frames, ignore_index=True)
    sample_tr = df_sample_all[df_sample_all["tanggal_dropping"] < pd.Timestamp("2026-08-01")]
    sample_tr_means = sample_tr.groupby("outlet_id")[TARGET].mean().to_dict()
    te_sample = df_sample_all[df_sample_all["split"] == "test"].copy()
    te_sample["p_Baseline"] = te_sample["outlet_id"].map(sample_tr_means).fillna(global_train_mean).round(1).values
    for name, m in fitted.items():
        te_sample[f"p_{name}"] = m.predict(te_sample[MODEL_FEATURES[name]])

    sample_order = {name: i for i, name in enumerate(SAMPLE_OUTLETS)}
    outlets = {}
    for oid, g in te_sample.sort_values("tanggal_dropping").groupby("outlet_id"):
        outlets[oid] = {
            "nama": g.nama_outlet.iloc[0],
            "wilayah": g.wilayah.iloc[0],
        }
    outlets = dict(sorted(outlets.items(), key=lambda kv: sample_order.get(kv[1]["nama"], 999)))

    # baris uji untuk halaman Prediksi interaktif (18 outlet sampel)
    oidx = {o: i for i, o in enumerate(outlets)}
    rows = [[d.strftime("%Y-%m-%d"), oidx[o], int(jb), int(a)] + [r(x, 1) for x in ps]
            for d, o, jb, a, ps in zip(te_sample.tanggal_dropping, te_sample.outlet_id, te_sample.jumlah_barang, te_sample[TARGET],
                                       zip(*[te_sample[f"p_{k}"].values for k in MODELS]))]

    # riwayat aktual lengkap (Juni - Agustus 2026) untuk 18 outlet sampel
    # Format history: [tanggal, outlet_idx, wil_luar_kota, terjual_aktual, dropping_aktual, sisa_aktual]
    df_valid = df_sample_all[df_sample_all.outlet_id.isin(oidx)].sort_values("tanggal_dropping")
    history = [[d.strftime("%Y-%m-%d"), oidx[o], int(w), int(a), int(jb), int(jb - a)]
               for d, o, w, a, jb in zip(df_valid.tanggal_dropping, df_valid.outlet_id,
                                         df_valid.wil_luar_kota, df_valid[TARGET], df_valid.jumlah_barang)]

    # ---------- PREDIKSI MASA DEPAN (SEPTEMBER - OKTOBER 2026, 8 MINGGU KE DEPAN) ----------
    # Alur sesuai tujuan utama:
    # 1) Model memprediksi `terjual` (barang terjual minggu ke depan)
    # 2) Dari prediksi `terjual` itu dihitung `total_dropping` (kelipatan 10 sesuai rasio serapan outlet) dan `sisa` (dropping - terjual)
    # 3) `total_dropping` dan `terjual` hasil prediksi menjadi input lag minggu berikutnya.
    yall = full[TARGET].values
    future_models = {
        "Linear Regression": LinearRegression().fit(full[MODEL_FEATURES["Linear Regression"]], yall),
        "Random Forest": RandomForestRegressor(n_estimators=300, min_samples_leaf=3, random_state=SEED, n_jobs=-1).fit(full[MODEL_FEATURES["Random Forest"]], yall),
        "Gradient Boosting": GradientBoostingRegressor(n_estimators=300, learning_rate=0.05, max_depth=3,
                                                       subsample=0.8, random_state=SEED).fit(full[MODEL_FEATURES["Gradient Boosting"]], yall),
    }

    outlet_last = {}
    for oid, g in df_valid.groupby("outlet_id"):
        tail = g.tail(3)
        sales = tail[TARGET].astype(float).tolist()
        while len(sales) < 3:
            sales.insert(0, sales[0])
        last_row = tail.iloc[-1]
        mean_rate = float(np.clip(g["rate"].mean(), 0.35, 0.90))
        first_sep = last_row.tanggal_dropping + pd.Timedelta(days=7)
        while first_sep < pd.Timestamp("2026-09-01"):
            first_sep += pd.Timedelta(days=7)
        outlet_last[oid] = {
            "first_sep": first_sep,
            "hari": int(last_row.hari_dropping),
            "luar": int(last_row.wil_luar_kota),
            "mean_rate": r(mean_rate, 3),
            "last_jb": int(last_row.jumlah_barang),
            "jb_state": {m: float(last_row.jumlah_barang) for m in MODELS},
            "sales": {m: list(sales[-3:]) for m in MODELS},
        }

    outlet_meta = [[o, v["nama"], v["wilayah"], outlet_last[o]["mean_rate"], outlet_last[o]["last_jb"]]
                   for o, v in outlets.items()]

    def calc_dropping_from_terjual(pred_val, target_rate):
        # Hitung total dropping (kelipatan 10, minimal >= prediksi terjual + buffer) dari prediksi terjual
        min_drop = np.ceil(max(pred_val, 1.0) * 1.10 / 10.0) * 10.0
        rate_drop = np.round((max(pred_val, 1.0) / max(target_rate, 0.35)) / 10.0) * 10.0
        return float(max(10.0, min_drop, rate_drop))

    future_rows = []
    for step in range(0, 8):
        step_oids = []
        step_dates = []
        X_step = {m: [] for m in MODELS if m != "Baseline"}
        base_preds = []

        for oid in outlets:
            st = outlet_last[oid]
            fdate = st["first_sep"] + pd.Timedelta(days=7 * step)
            step_oids.append(oid)
            step_dates.append(fdate)

            # Baseline: flat horizontal line menggunakan rata-rata historis data latih
            b_pred = r(sample_tr_means.get(oid, global_train_mean), 1)
            base_preds.append(b_pred)

            for m in X_step:
                s_m = st["sales"][m]
                lag1 = s_m[-1]
                lag2 = s_m[-2]
                roll3 = float(np.mean(s_m[-3:]))
                jb1 = float(st["jb_state"][m])
                rate1 = float(np.clip(lag1 / jb1, 0.0, 1.0)) if jb1 > 0 else 0.0
                habis1 = 1 if lag1 >= jb1 else 0
                row_dict = {
                    "bulan": int(fdate.month),
                    "hari_dropping": st["hari"],
                    "wil_luar_kota": st["luar"],
                    "jb_lag1": jb1,
                    "terjual_lag1": lag1,
                    "terjual_lag2": lag2,
                    "terjual_roll3": roll3,
                    "rate_lag1": rate1,
                    "habis_lag1": habis1,
                }
                X_step[m].append([row_dict[f] for f in MODEL_FEATURES[m]])

        m_preds = {"Baseline": base_preds}
        for m, feat_rows in X_step.items():
            X_df = pd.DataFrame(feat_rows, columns=MODEL_FEATURES[m])
            m_preds[m] = np.clip(future_models[m].predict(X_df), 0, None).tolist()

        for i, oid in enumerate(step_oids):
            st = outlet_last[oid]
            for m in MODELS:
                p_val = float(m_preds[m][i])
                st["sales"][m].append(p_val)
                st["jb_state"][m] = calc_dropping_from_terjual(p_val, st["mean_rate"])
            future_rows.append([
                step_dates[i].strftime("%Y-%m-%d"),
                oidx[oid],
                st["last_jb"],
                r(m_preds["Baseline"][i], 1),
                r(m_preds["Linear Regression"][i], 1),
                r(m_preds["Random Forest"][i], 1),
                r(m_preds["Gradient Boosting"][i], 1),
            ])

    future_rows.sort(key=lambda x: (x[0], x[1]))

    # simulator: ekspor model sederhana? -> prediksi dihitung ulang di browser via LR; RF/GB via grid input
    sim_models = {"lr": {"intercept": lr_int, "coef": importance["Linear Regression"]}}

    out = {
        "meta": {"features": FEATURES, "model_features": MODEL_FEATURES, "model_sheets": MODEL_SHEETS,
                 "target": TARGET, "best": best, "seed": SEED,
                 "n_features": len(FEATURES), "split": "Train: 1 Jun-31 Jul 2026 | Test: Agustus 2026 | Forecast: Sep-Okt 2026",
                 "kamus": [[str(c) for c in row[:4]] for row in kamus.values[1:24] if str(row[0]) != "nan"]},
        "prep": prep, "eda": eda, "results": res, "importance": importance, "by_wilayah": by_wil,
        "by_week": by_week, "by_vol": by_vol, "scatter": scatter,
        "resid": {"labels": rlabels, "hist": rhist}, "daily": daily_json,
        "sim": sim_models, "rows": rows, "future": future_rows, "outlet_meta": outlet_meta, "history": history,
        "outlier": outlier,
        "range": [str(te.tanggal_dropping.min().date()), str(te.tanggal_dropping.max().date())],
        "future_range": [future_rows[0][0], future_rows[-1][0]],
        "params": {"Random Forest": "Sheet: model_rf (8 fitur) | n_estimators=300, min_samples_leaf=3",
                   "Gradient Boosting": "Sheet: model_gb (9 fitur) | n_estimators=300, lr=0.05, max_depth=3",
                   "Linear Regression": "Sheet: model_lr (6 fitur) | OLS bebas bias bulan",
                   "Baseline": "Rata-rata historis (Mean Statis) per outlet dari data latih"},
    }
    OUT.write_text("window.DATA = " + json.dumps(out, ensure_ascii=False) + ";", encoding="utf-8")
    print(json.dumps(res, indent=1))
    print("best:", best, "->", OUT)


if __name__ == "__main__":
    main()
