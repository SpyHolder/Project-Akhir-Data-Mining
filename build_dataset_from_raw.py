"""
build_dataset_from_raw.py
=========================
Membangun ulang dataset_siap_modeling_baru.xlsx dari 6 file data mentah baru (633 outlet).
Mencakup ekstraksi data, pembersihan 156 data dihapus (146 luar seleksi + 9 duplikat/ambigu + 1 outlier HB),
feature engineering lag & rolling untuk 477 outlet model, pembagian split train/test,
dan formatting excel profesional.
"""
import openpyxl, os, re, datetime, shutil
from pathlib import Path
import pandas as pd
import numpy as np
from openpyxl import Workbook
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter

BASE_DIR = Path(__file__).resolve().parent
RAW_DIR = BASE_DIR / "data mentah"
OUT_FILE = BASE_DIR / "dataset_siap_modeling_baru.xlsx"
LOCK_FILE = BASE_DIR / "dataset_siap_modeling.xlsx"
REF_FILE = BASE_DIR / "referensi_487_outlet.csv"

# ---------- Ekstraksi Data Mentah ----------
def extract_raw_records():
    files = sorted([f for f in os.listdir(RAW_DIR) if f.endswith('.xlsx')])
    records = []
    
    for f in files:
        wb = openpyxl.load_workbook(RAW_DIR / f, data_only=True)
        wil = 'LUAR KOTA' if 'LUAR' in f.upper() else 'DALAM KOTA'
        month_num = 6 if 'JUNI' in f.upper() else (7 if 'JULI' in f.upper() else 8)
        
        for sn in wb.sheetnames:
            ws = wb[sn]
            h_row = 2
            headers = [str(ws.cell(row=h_row, column=c).value or '').strip().upper() for c in range(1, ws.max_column+1)]
            
            name_col = 2
            for c, h in enumerate(headers, 1):
                if 'NAMA OUTLET' in h:
                    name_col = c
                    break
            
            block_starts = [c for c, h in enumerate(headers, 1) if 'TANGGAL DROPING' in h or 'TANGGAL' in h]
            
            for r in range(h_row+1, ws.max_row+1):
                nama = ws.cell(row=r, column=name_col).value
                if not nama or not str(nama).strip(): continue
                nama_s = re.sub(r'\s+', ' ', str(nama).strip().upper())
                if any(x in nama_s for x in ['TOTAL', 'JUMLAH']): continue
                
                for m_idx, b_col in enumerate(block_starts, 1):
                    tgl = ws.cell(row=r, column=b_col).value
                    if not tgl: continue
                    jb = ws.cell(row=r, column=b_col+2).value
                    tj = ws.cell(row=r, column=b_col+3).value
                    sisa = ws.cell(row=r, column=b_col+4).value
                    
                    records.append({
                        'nama_outlet': nama_s,
                        'wilayah': wil,
                        'bulan': month_num,
                        'minggu_ke': m_idx,
                        'tanggal_dropping': pd.to_datetime(tgl),
                        'jumlah_barang': int(jb) if pd.notna(jb) else 0,
                        'terjual': int(tj) if pd.notna(tj) else 0,
                        'sisa': int(sisa) if pd.notna(sisa) else 0,
                    })
        wb.close()
    
    df = pd.DataFrame(records)
    df['sisa'] = df['jumlah_barang'] - df['terjual']
    return df

# ---------- Feature Engineering & Modeling Dataset ----------
def build_features(df):
    ref = pd.read_csv(REF_FILE)
    
    # Map status & alasan dari referensi 487 outlet
    ref_map = {}
    for _, row in ref.iterrows():
        k = (row['nama_outlet'], row['wilayah'])
        st = str(row['status'])
        al = str(row['alasan']) if pd.notna(row['alasan']) else ''
        if 'HB' in st or row['nama_outlet'] == 'HB':
            ref_map[k] = ('dihapus (outlier HB)', 'Outlier: volume penjualan ekstrem (grosir) ~8x median')
        elif 'duplikat' in st:
            ref_map[k] = ('dihapus (duplikat/ambigu)', al)
        else:
            ref_map[k] = ('dipakai di model', '')

    outlets_sorted = df.groupby(['nama_outlet', 'wilayah']).agg(
        jumlah_baris=('tanggal_dropping', 'count'),
        pertama=('tanggal_dropping', 'min'),
        terakhir=('tanggal_dropping', 'max')
    ).reset_index().sort_values(['nama_outlet', 'wilayah']).reset_index(drop=True)
    
    ref_ids = ref[['outlet_id', 'nama_outlet', 'wilayah']]
    outlets_sorted = outlets_sorted.merge(ref_ids, on=['nama_outlet', 'wilayah'], how='left')
    terpilih = outlets_sorted['outlet_id'].notna()
    next_id = int(ref_ids['outlet_id'].str[1:].astype(int).max()) + 1
    new_ids = [f"O{next_id + i:04d}" for i in range((~terpilih).sum())]
    outlets_sorted.loc[~terpilih, 'outlet_id'] = new_ids
    outlets_sorted = outlets_sorted.sort_values('outlet_id').reset_index(drop=True)

    def _status(r):
        k = (r['nama_outlet'], r['wilayah'])
        if k in ref_map:
            return ref_map[k][0]
        return 'dihapus (di luar seleksi)'

    def _alasan(r):
        k = (r['nama_outlet'], r['wilayah'])
        if k in ref_map:
            return ref_map[k][1]
        return 'Di luar seleksi rujukan (dieliminasi)'

    outlets_sorted['status'] = outlets_sorted.apply(_status, axis=1)
    outlets_sorted['alasan'] = outlets_sorted.apply(_alasan, axis=1)

    peta_outlet_id = outlets_sorted[['outlet_id', 'nama_outlet', 'wilayah', 'jumlah_baris', 'pertama', 'terakhir', 'status']]
    
    # Merge outlet_id dan alasan ke data transaksi
    df = df.merge(outlets_sorted[['outlet_id', 'nama_outlet', 'wilayah', 'status', 'alasan']], on=['nama_outlet', 'wilayah'])
    df = df.sort_values(['outlet_id', 'tanggal_dropping']).reset_index(drop=True)
    
    # Pisahkan data model (477 outlet) dan data dihapus (156 outlet)
    df_del = df[df['status'].str.startswith('dihapus')].copy()
    df_model = df[df['status'] == 'dipakai di model'].copy()
    
    # Feature engineering per outlet untuk model
    dfs_featured = []
    for oid, group in df_model.groupby('outlet_id'):
        g = group.sort_values('tanggal_dropping').copy()
        g['bulan'] = g['tanggal_dropping'].dt.month
        g['hari_dropping'] = g['tanggal_dropping'].dt.dayofweek
        g['wil_luar_kota'] = (g['wilayah'] == 'LUAR KOTA').astype(int)
        
        g['jb_lag1'] = g['jumlah_barang'].shift(1)
        g['terjual_lag1'] = g['terjual'].shift(1)
        g['terjual_lag2'] = g['terjual'].shift(2)
        g['terjual_roll3'] = g['terjual'].shift(1).rolling(3, min_periods=3).mean().round(2)
        
        prev_jb = g['jumlah_barang'].shift(1)
        prev_tj = g['terjual'].shift(1)
        g['rate_lag1'] = np.where(prev_jb > 0, (prev_tj / prev_jb).round(4), 0.0)
        g['rate_lag1'] = np.where(prev_jb.isna(), np.nan, g['rate_lag1'])
        
        g['habis_lag1'] = np.where(g['sisa'].shift(1) == 0, 1, 0)
        g['habis_lag1'] = np.where(g['sisa'].shift(1).isna(), np.nan, g['habis_lag1'])
        
        g['split'] = np.where(g['tanggal_dropping'] < pd.to_datetime('2026-08-01'), 'train', 'test')
        
        lag_cols = ['jb_lag1', 'terjual_lag1', 'terjual_lag2', 'terjual_roll3', 'rate_lag1', 'habis_lag1']
        g['fitur_lengkap'] = g[lag_cols].notna().all(axis=1).astype(int)
        
        dfs_featured.append(g)
    
    full_model = pd.concat(dfs_featured, ignore_index=True)
    
    # Sheet dihapus: 156 outlet (146 luar seleksi + 9 duplikat + 1 outlier HB)
    dihapus_df = df_del[['outlet_id', 'nama_outlet', 'wilayah', 'tanggal_dropping', 'alasan',
                         'minggu_ke', 'jumlah_barang', 'terjual', 'sisa']].reset_index(drop=True)
    
    return full_model, dihapus_df, peta_outlet_id

# ---------- Excel Formatting Styles ----------
C_TITLE = "0F172A"
GROUP_COLOR = {
    "identitas": "334155",   # abu gelap
    "transaksi": "0369A1",   # biru
    "target": "047857",      # hijau
    "fitur": "6D28D9",       # ungu
    "pembagi": "B45309",     # oranye
    "info": "475569",        # abu
}
BAND_FILL = PatternFill("solid", fgColor="F1F5F9")
THIN = Side(style="thin", color="CBD5E1")
BORDER = Border(left=THIN, right=THIN, top=THIN, bottom=THIN)
HEAD_FONT = Font(bold=True, color="FFFFFF", name="Calibri", size=11)

GROUP_OF = {
    **{c: "identitas" for c in ["outlet_id", "nama_outlet", "wilayah", "tanggal_dropping", "minggu_ke", "alasan"]},
    **{c: "transaksi" for c in ["jumlah_barang", "sisa", "rate", "habis"]},
    "terjual": "target",
    **{c: "fitur" for c in ["bulan", "hari_dropping", "wil_luar_kota", "jb_lag1", "terjual_lag1",
                            "terjual_lag2", "terjual_roll3", "rate_lag1", "habis_lag1"]},
    **{c: "pembagi" for c in ["split", "fitur_lengkap"]},
}

DATE_COLS = {"tanggal_dropping", "pertama", "terakhir"}
PCT_COLS = {"rate", "rate_lag1"}
DEC_COLS = {"terjual_roll3"}
INT_COLS = {"jb_lag1", "terjual_lag1", "terjual_lag2", "habis_lag1", "jeda_hari"}

LR = ["wil_luar_kota", "jb_lag1", "terjual_lag1", "terjual_lag2", "terjual_roll3", "rate_lag1"]
RF = ["hari_dropping", "wil_luar_kota", "jb_lag1", "terjual_lag1", "terjual_lag2", "terjual_roll3",
      "rate_lag1", "habis_lag1"]
GB = ["bulan", "hari_dropping", "wil_luar_kota", "jb_lag1", "terjual_lag1", "terjual_lag2",
      "terjual_roll3", "rate_lag1", "habis_lag1"]

def write_df(ws, df, band_by="outlet_id"):
    cols = list(df.columns)
    for j, c in enumerate(cols, 1):
        cell = ws.cell(row=1, column=j, value=c)
        cell.font = HEAD_FONT
        cell.fill = PatternFill("solid", fgColor=GROUP_COLOR[GROUP_OF.get(c, "info")])
        cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
        cell.border = BORDER
    ws.row_dimensions[1].height = 30

    band, last_key = False, None
    for i, row in enumerate(df.itertuples(index=False), 2):
        if band_by in df.columns:
            key = getattr(row, band_by)
            if key != last_key:
                band, last_key = not band, key
        else:
            band = i % 2 == 0
        for j, (c, v) in enumerate(zip(cols, row), 1):
            if pd.isna(v):
                v = None
            elif isinstance(v, pd.Timestamp):
                v = v.to_pydatetime()
            cell = ws.cell(row=i, column=j, value=v)
            cell.border = BORDER
            if band:
                cell.fill = BAND_FILL
            if c in DATE_COLS:
                cell.number_format = "DD-MM-YYYY"
                cell.alignment = Alignment(horizontal="center")
            elif c in PCT_COLS:
                cell.number_format = "0.0%"
            elif c in DEC_COLS:
                cell.number_format = "0.00"
            elif c in INT_COLS:
                cell.number_format = "0"
            elif c in {"outlet_id", "wilayah", "split", "minggu_ke", "bulan", "hari_dropping",
                       "wil_luar_kota", "fitur_lengkap", "habis", "status"}:
                cell.alignment = Alignment(horizontal="center")

    for j, c in enumerate(cols, 1):
        sample = pd.Series([len(str(v)) for v in df[c].head(500)], dtype=int)
        width = max(len(c), int(sample.max()) if len(sample) else 0) + 3
        if c in DATE_COLS:
            width = 13
        ws.column_dimensions[get_column_letter(j)].width = min(max(width, 9), 60)

    ws.freeze_panes = "C2" if "nama_outlet" in cols else "A2"
    ws.auto_filter.ref = f"A1:{get_column_letter(len(cols))}{len(df) + 1}"

def title(ws, r, text, size=14):
    cell = ws.cell(row=r, column=1, value=text)
    cell.font = Font(bold=True, size=size, color=C_TITLE)

def table(ws, r, head, rows, widths=None, color="0F172A"):
    for j, h in enumerate(head, 1):
        c = ws.cell(row=r, column=j, value=h)
        c.font = HEAD_FONT
        c.fill = PatternFill("solid", fgColor=color)
        c.border = BORDER
        c.alignment = Alignment(horizontal="center", vertical="center")
    for i, row in enumerate(rows, r + 1):
        for j, v in enumerate(row, 1):
            c = ws.cell(row=i, column=j, value=v)
            c.border = BORDER
            c.alignment = Alignment(vertical="top", wrap_text=True)
            if i % 2 == 0:
                c.fill = BAND_FILL
    return r + len(rows) + 2

def build_ringkasan(ws, peta, full, dih):
    n_clean = len(full)
    n_del = len(dih)
    lengkap = full[full.fitur_lengkap == 1]
    n_tr = int((lengkap.split == "train").sum())
    n_te = int((lengkap.split == "test").sum())

    n_outlet_total = len(peta)
    n_outlet_clean = full.outlet_id.nunique()
    n_outlet_del = int(peta["status"].str.contains("dihapus").sum())
    n_outlet_tr = full[full.split == "train"].outlet_id.nunique()
    n_outlet_te = full[full.split == "test"].outlet_id.nunique()

    title(ws, 1, "DATASET SIAP MODELING - Prediksi Penjualan Dropping Outlet", 16)
    ws.cell(row=2, column=1, value="Periode Juni - Agustus 2026 | Alur: Data Mentah (6 file Excel) -> Dataset Siap Modeling (file ini)").font = Font(italic=True, color="475569")

    r = 4
    title(ws, r, "1. Alur Pengolahan Data (Berdasarkan Jumlah Outlet)", 12)
    r = table(ws, r + 1, ["Tahap", "Jumlah Outlet", "Keterangan & Riwayat Transaksi"], [
        ["Total Outlet Terdaftar (Data Mentah)", f"{n_outlet_total} outlet", f"Seluruh outlet di 6 file data mentah (total {n_clean + n_del} baris transaksi)"],
        ["-> Outlet Dihapus / Dieliminasi", f"{n_outlet_del} outlet", f"146 di luar seleksi + 9 duplikat/ambigu + 1 outlier HB (lihat sheet 'dihapus', {n_del} baris)"],
        ["-> Total Outlet Masuk Model (dataset_model)", f"{n_outlet_clean} outlet", f"Outlet aktif reguler siap modeling (total {n_clean} baris transaksi bersih)"],
        ["   Data Latih (Train)", f"{n_tr} baris", f"Juni - Juli 2026: {n_outlet_tr} outlet, fitur lengkap siap training"],
        ["   Data Uji (Test)", f"{n_te} baris", f"Agustus 2026: {n_outlet_te} outlet, evaluasi akurasi model"],
    ])

    title(ws, r, "2. Daftar Sheet", 12)
    r = table(ws, r + 1, ["Sheet", "Isi", "Jumlah Kolom"], [
        ["Ringkasan", "Halaman ini: alur data, daftar sheet, keterangan warna", "-"],
        ["Kamus_Data", "Arti setiap kolom dan di sheet mana kolom tersebut dipakai", "-"],
        ["dataset_model", "Hasil data cleaning murni (tanpa fitur tambahan)", 8],
        ["model_lr", f"Fitur untuk Linear Regression ({len(LR)} fitur)", 5 + len(LR) + 3],
        ["model_rf", f"Fitur untuk Random Forest ({len(RF)} fitur)", 5 + len(RF) + 3],
        ["model_gb", f"Fitur untuk Gradient Boosting ({len(GB)} fitur)", 5 + len(GB) + 3],
        ["dihapus", "Baris/outlet anomali, duplikat, dan outlier yang dikeluarkan dari analisis", 9],
        ["peta_outlet_id", "Daftar seluruh outlet terdaftar beserta ID unik dan statusnya", 7],
    ])

    title(ws, r, "3. Keterangan Warna Header Kolom", 12)
    r = table(ws, r + 1, ["Kelompok", "Arti", "Contoh Kolom"], [
        ["Abu gelap", "Identitas (outlet_id, nama, wilayah, tanggal, minggu)", "outlet_id, nama_outlet, tanggal_dropping"],
        ["Biru", "Transaksi aktual hasil dropping", "jumlah_barang, sisa, rate"],
        ["Hijau", "TARGET yang diprediksi oleh model", "terjual"],
        ["Ungu", "Fitur prediktor (lag, rolling, kalender)", "jb_lag1, terjual_lag1, terjual_roll3"],
        ["Oranye", "Pembagi data (split train/test, penanda fitur lengkap)", "split, fitur_lengkap"],
    ])

def build_kamus(ws):
    title(ws, 1, "KAMUS DATA - Arti Setiap Kolom", 16)
    ws.cell(row=2, column=1, value="Rincian setiap kolom di seluruh sheet dataset siap modeling").font = Font(italic=True, color="475569")
    kamus_rows = [
        ["outlet_id", "dibuat", "ID unik outlet = kombinasi nama_outlet + wilayah (O0001 - O0633)", "identitas", "semua sheet"],
        ["nama_outlet", "asli", "Nama toko/outlet dari data mentah (distandarkan huruf besar)", "identitas", "semua sheet"],
        ["wilayah", "asli", "Wilayah dropping: DALAM KOTA atau LUAR KOTA", "identitas", "semua sheet"],
        ["tanggal_dropping", "asli", "Tanggal barang di-drop ke outlet (format DD-MM-YYYY)", "identitas", "semua sheet"],
        ["minggu_ke", "asli", "Minggu ke- dalam bulan pada file mentah (1 s/d 5)", "identitas", "semua sheet"],
        ["jumlah_barang", "asli", "Jumlah barang yang di-drop (keputusan pemilik) - TIDAK jadi fitur", "transaksi", "dataset_model, dihapus"],
        ["terjual", "asli", "Jumlah barang terjual pada minggu dropping tersebut", "TARGET", "semua sheet model & dataset_model"],
        ["sisa", "asli", "Sisa barang yang tidak terjual (jumlah_barang - terjual)", "transaksi", "dataset_model, dihapus"],
        ["bulan", "turunan", "Bulan dropping: 6 (Juni), 7 (Juli), 8 (Agustus)", "fitur", "model_gb"],
        ["hari_dropping", "turunan", "Hari dropping: 0=Senin, 1=Selasa ... 5=Sabtu", "fitur", "model_rf, model_gb"],
        ["wil_luar_kota", "turunan", "Biner: 1 = LUAR KOTA, 0 = DALAM KOTA", "fitur", "model_lr, model_rf, model_gb"],
        ["jb_lag1", "turunan", "jumlah_barang pada dropping sebelumnya", "fitur", "model_lr, model_rf, model_gb"],
        ["terjual_lag1", "turunan", "terjual pada dropping sebelumnya", "fitur", "model_lr, model_rf, model_gb"],
        ["terjual_lag2", "turunan", "terjual pada dua dropping sebelumnya", "fitur", "model_lr, model_rf, model_gb"],
        ["terjual_roll3", "turunan", "Rata-rata terjual 3 dropping sebelumnya (juga dasar Baseline)", "fitur", "model_lr, model_rf, model_gb"],
        ["rate_lag1", "turunan", "Rasio terjual / jumlah_barang pada dropping sebelumnya", "fitur", "model_lr, model_rf, model_gb"],
        ["habis_lag1", "turunan", "Biner: 1 jika dropping sebelumnya habis (sisa = 0), 0 jika ada sisa", "fitur", "model_rf, model_gb"],
        ["split", "turunan", "train = sebelum 1 Agustus 2026, test = Agustus 2026", "pembagi data", "model_lr, model_rf, model_gb"],
        ["fitur_lengkap", "turunan", "1 jika semua fitur lag terisi. Hanya baris = 1 yang dipakai training & evaluasi", "pembagi data", "model_lr, model_rf, model_gb"],
    ]
    table(ws, 4, ["Nama Kolom", "Asal", "Keterangan", "Kelompok", "Ada di Sheet"], kamus_rows)

def main():
    print("1. Mengekstrak data mentah...")
    raw_df = extract_raw_records()
    print(f"   Ekstraksi selesai: {len(raw_df)} baris transaksi.")
    
    print("2. Membangun fitur lag, rolling, dan pemisahan 156 data dihapus...")
    full, dih, peta = build_features(raw_df)
    print(f"   Model data: {len(full)} baris, {full.outlet_id.nunique()} outlet.")
    print(f"   Dihapus data: {len(dih)} baris, {dih.outlet_id.nunique()} outlet.")
    
    ident = ["outlet_id", "nama_outlet", "wilayah", "tanggal_dropping", "minggu_ke"]
    tail = ["terjual", "split", "fitur_lengkap"]
    sheets = {
        "dataset_model": full[ident + ["jumlah_barang", "terjual", "sisa"]],
        "model_lr": full[ident + LR + tail],
        "model_rf": full[ident + RF + tail],
        "model_gb": full[ident + GB + tail],
    }
    
    print("3. Membangun workbook Excel siap modeling...")
    wb = Workbook()
    ws = wb.active
    ws.title = "Ringkasan"
    build_ringkasan(ws, peta, full, dih)
    build_kamus(wb.create_sheet("Kamus_Data"))
    for name, df in sheets.items():
        write_df(wb.create_sheet(name), df)
    write_df(wb.create_sheet("dihapus"), dih)
    write_df(wb.create_sheet("peta_outlet_id"), peta, band_by="__none__")
    
    tabs = {"Ringkasan": "0F172A", "Kamus_Data": "0F172A", "dataset_model": "0369A1",
            "model_lr": "6D28D9", "model_rf": "6D28D9", "model_gb": "6D28D9",
            "dihapus": "B91C1C", "peta_outlet_id": "475569"}
    for n, col in tabs.items():
        wb[n].sheet_properties.tabColor = col
    
    wb.save(OUT_FILE)
    print(f"   Tersimpan ke: {OUT_FILE.name}")
    try:
        shutil.copyfile(OUT_FILE, LOCK_FILE)
        print(f"   Berhasil disalin ke: {LOCK_FILE.name}")
    except PermissionError:
        print(f"   {LOCK_FILE.name} terkunci di Excel, perubahan aktif di {OUT_FILE.name}")

if __name__ == "__main__":
    main()
