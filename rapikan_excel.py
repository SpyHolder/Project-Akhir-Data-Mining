"""Rapikan tampilan dataset_siap_modeling.xlsx.

- Urutan sheet: Ringkasan -> Kamus_Data -> dataset_model -> model_lr/rf/gb -> HB_terpisah -> dihapus -> peta_outlet_id
- Header berwarna sesuai kelompok kolom (identitas, transaksi, fitur, target, pembagi data)
- Freeze panes, autofilter, lebar kolom otomatis, format tanggal & angka
- Baris diberi warna selang-seling per outlet supaya riwayat satu outlet mudah dibaca
- Kamus_Data disusun ulang menjadi tabel rapi sesuai struktur sheet yang baru

Jalankan:  python rapikan_excel.py
(Tutup file Excel-nya dulu. Jika masih terbuka, hasil disimpan ke dataset_siap_modeling_baru.xlsx)
"""
from pathlib import Path

import pandas as pd
from openpyxl import Workbook
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter

BASE = Path(__file__).parent
SRC = BASE / "dataset_siap_modeling.xlsx"
NEW = BASE / "dataset_siap_modeling_baru.xlsx"

# ---------- palet warna ----------
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
    **{c: "identitas" for c in ["outlet_id", "nama_outlet", "wilayah", "tanggal_dropping", "minggu_ke"]},
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


def load():
    path = NEW if NEW.exists() else SRC
    xl = pd.read_excel(path, sheet_name=None)
    base = xl["model_gb"] if "model_gb" in xl else xl["dataset_model"]
    clean = xl["dataset_model"]
    # gabungkan kolom bersih + seluruh fitur (sumber untuk membuat ulang sheet algoritma)
    full = pd.concat([clean, base[[c for c in base.columns if c not in clean.columns]]], axis=1)
    full["tanggal_dropping"] = pd.to_datetime(full["tanggal_dropping"])
    full = full.sort_values(["outlet_id", "tanggal_dropping"]).reset_index(drop=True)
    for c in ["terjual_roll3"]:
        full[c] = full[c].round(2)
    full["rate_lag1"] = full["rate_lag1"].round(4)
    return xl, full


def write_df(ws, df, band_by="outlet_id"):
    cols = list(df.columns)
    # header
    for j, c in enumerate(cols, 1):
        cell = ws.cell(row=1, column=j, value=c)
        cell.font = HEAD_FONT
        cell.fill = PatternFill("solid", fgColor=GROUP_COLOR[GROUP_OF.get(c, "info")])
        cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
        cell.border = BORDER
    ws.row_dimensions[1].height = 30

    # isi
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

    # lebar kolom otomatis
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


def build_ringkasan(ws, xl, full):
    peta = xl["peta_outlet_id"]
    n_clean = len(full)
    n_hb = len(xl["HB_terpisah"])
    n_del = len(xl["dihapus"])
    lengkap = full[full.fitur_lengkap == 1]
    n_tr = int((lengkap.split == "train").sum())
    n_te = int((lengkap.split == "test").sum())

    n_outlet_total = len(peta)
    n_outlet_clean = full.outlet_id.nunique()
    n_outlet_del = int(peta["status"].str.contains("dihapus").sum())
    n_outlet_hb = int(peta["status"].str.contains("HB").sum())
    n_outlet_tr = full[full.split == "train"].outlet_id.nunique()
    n_outlet_te = full[full.split == "test"].outlet_id.nunique()

    title(ws, 1, "DATASET SIAP MODELING - Prediksi Penjualan Dropping Outlet", 16)
    ws.cell(row=2, column=1, value="Periode Juni - Agustus 2026 | Alur: Data Mentah (6 file Excel) -> Dataset Siap Modeling (file ini)").font = Font(italic=True, color="475569")

    r = 4
    title(ws, r, "1. Alur Pengolahan Data (Berdasarkan Jumlah Outlet)", 12)
    r = table(ws, r + 1, ["Tahap", "Jumlah Outlet", "Keterangan & Riwayat Transaksi"], [
        ["Total Outlet Terdaftar (Data Mentah)", f"{n_outlet_total} outlet", f"Seluruh outlet di 6 file data mentah (total {n_clean + n_hb + n_del} baris transaksi)"],
        ["-> Outlet dihapus (anomali)", f"{n_outlet_del} outlet", f"Duplikat / nama rancu antar wilayah (lihat sheet 'dihapus', {n_del} baris)"],
        ["-> Outlet HB dipisah (outlier)", f"{n_outlet_hb} outlet", f"Outlet grosir dengan volume ekstrem dipisah (lihat sheet 'HB_terpisah', {n_hb} baris)"],
        ["-> Outlet Masuk Model (dataset_model)", f"{n_outlet_clean} outlet", f"Outlet aktif reguler siap modeling (total {n_clean} baris transaksi bersih)"],
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
        ["HB_terpisah", "Riwayat Outlet HB yang dipisahkan (outlier)", len(xl["HB_terpisah"].columns)],
        ["dihapus", "Baris yang dibuang saat cleaning beserta alasannya", len(xl["dihapus"].columns)],
        ["peta_outlet_id", "Daftar seluruh outlet, ID unik, dan statusnya", len(xl["peta_outlet_id"].columns)],
    ])
    ws.cell(row=r - 1, column=1, value="Baseline tidak punya sheet sendiri: prediksinya menggunakan rata-rata historis (Mean Statis) per outlet.").font = Font(italic=True, color="475569")
    r += 1

    title(ws, r, "3. Keterangan Warna Header", 12)
    r += 1
    legend = [
        ("identitas", "Identitas baris (outlet, wilayah, tanggal)"),
        ("transaksi", "Data transaksi asli (jumlah barang, sisa)"),
        ("target", "Target yang diprediksi (terjual)"),
        ("fitur", "Fitur prediktor hasil feature engineering"),
        ("pembagi", "Pembagi data (split train/test, penanda fitur lengkap)"),
    ]
    for g, desc in legend:
        c = ws.cell(row=r, column=1, value=g.upper())
        c.fill = PatternFill("solid", fgColor=GROUP_COLOR[g])
        c.font = HEAD_FONT
        c.border = BORDER
        ws.cell(row=r, column=2, value=desc).border = BORDER
        r += 1
    ws.cell(row=r + 1, column=1, value="Baris berwarna selang-seling per outlet: satu blok warna = riwayat satu outlet.").font = Font(italic=True, color="475569")

    ws.column_dimensions["A"].width = 36
    ws.column_dimensions["B"].width = 70
    ws.column_dimensions["C"].width = 70
    ws.sheet_view.showGridLines = False


KAMUS = [
    # kolom, jenis, keterangan, peran, sheet
    ["outlet_id", "dibuat", "ID unik outlet = kombinasi nama_outlet + wilayah (O0001, O0002, ...)", "identitas", "semua"],
    ["nama_outlet", "asli", "Nama outlet dari data mentah (sudah distandarkan huruf besar)", "identitas", "semua"],
    ["wilayah", "asli", "DALAM KOTA / LUAR KOTA", "identitas", "semua"],
    ["tanggal_dropping", "asli", "Tanggal barang di-drop ke outlet", "identitas", "semua"],
    ["minggu_ke", "asli", "Minggu ke- dalam bulan pada file mentah", "identitas", "semua"],
    ["jumlah_barang", "asli", "Jumlah barang yang di-drop minggu itu (keputusan pemilik) - TIDAK dijadikan fitur", "transaksi", "dataset_model"],
    ["terjual", "asli", "Jumlah barang terjual minggu itu", "TARGET", "semua"],
    ["sisa", "asli", "jumlah_barang - terjual", "transaksi", "dataset_model"],
    ["bulan", "turunan", "Bulan dropping (6, 7, 8)", "fitur", "model_gb"],
    ["hari_dropping", "turunan", "Hari dropping: 0=Senin ... 6=Minggu", "fitur", "model_rf, model_gb"],
    ["wil_luar_kota", "turunan", "1 = LUAR KOTA, 0 = DALAM KOTA", "fitur", "model_lr, model_rf, model_gb"],
    ["jb_lag1", "turunan", "jumlah_barang pada dropping sebelumnya", "fitur", "model_lr, model_rf, model_gb"],
    ["terjual_lag1", "turunan", "terjual pada dropping sebelumnya", "fitur", "model_lr, model_rf, model_gb"],
    ["terjual_lag2", "turunan", "terjual dua dropping sebelumnya", "fitur", "model_lr, model_rf, model_gb"],
    ["terjual_roll3", "turunan", "Rata-rata terjual 3 dropping sebelumnya (juga dipakai sebagai Baseline)", "fitur", "model_lr, model_rf, model_gb"],
    ["rate_lag1", "turunan", "Rasio terjual / jumlah_barang pada dropping sebelumnya", "fitur", "model_lr, model_rf, model_gb"],
    ["habis_lag1", "turunan", "1 jika dropping sebelumnya habis terjual (sisa = 0)", "fitur", "model_rf, model_gb"],
    ["split", "turunan", "train = sebelum 1 Agustus 2026, test = Agustus 2026", "pembagi data", "model_lr, model_rf, model_gb"],
    ["fitur_lengkap", "turunan", "1 jika semua fitur lag sudah terisi (dropping ke-3 dst). Training & evaluasi hanya memakai baris = 1", "filter", "model_lr, model_rf, model_gb"],
]


def build_kamus(ws):
    title(ws, 1, "KAMUS DATA", 14)
    ws.cell(row=2, column=1, value="Arti setiap kolom dan sheet tempat kolom itu dipakai").font = Font(italic=True, color="475569")
    head = ["kolom", "jenis", "keterangan", "peran", "dipakai di sheet"]
    r0 = 4
    for j, h in enumerate(head, 1):
        c = ws.cell(row=r0, column=j, value=h)
        c.font = HEAD_FONT
        c.fill = PatternFill("solid", fgColor=C_TITLE)
        c.border = BORDER
        c.alignment = Alignment(horizontal="center")
    role_group = {"identitas": "identitas", "transaksi": "transaksi", "TARGET": "target",
                  "fitur": "fitur", "pembagi data": "pembagi", "filter": "pembagi"}
    for i, row in enumerate(KAMUS, r0 + 1):
        for j, v in enumerate(row, 1):
            c = ws.cell(row=i, column=j, value=v)
            c.border = BORDER
            c.alignment = Alignment(vertical="top", wrap_text=True)
        cc = ws.cell(row=i, column=1)
        cc.font = Font(bold=True, color="FFFFFF")
        cc.fill = PatternFill("solid", fgColor=GROUP_COLOR[role_group[row[3]]])
    for col, w in zip("ABCDE", [18, 10, 70, 14, 32]):
        ws.column_dimensions[col].width = w
    ws.freeze_panes = f"A{r0 + 1}"
    ws.sheet_view.showGridLines = False


def main():
    xl, full = load()
    ident = ["outlet_id", "nama_outlet", "wilayah", "tanggal_dropping", "minggu_ke"]
    tail = ["terjual", "split", "fitur_lengkap"]
    sheets = {
        "dataset_model": full[ident + ["jumlah_barang", "terjual", "sisa"]],
        "model_lr": full[ident + LR + tail],
        "model_rf": full[ident + RF + tail],
        "model_gb": full[ident + GB + tail],
    }
    hb = xl["HB_terpisah"].copy()
    dih = xl["dihapus"].copy()
    peta = xl["peta_outlet_id"].copy()
    for d in (hb, dih):
        d["tanggal_dropping"] = pd.to_datetime(d["tanggal_dropping"])
    dih = dih.sort_values(["outlet_id", "tanggal_dropping"])
    for c in ("pertama", "terakhir"):
        peta[c] = pd.to_datetime(peta[c])

    wb = Workbook()
    ws = wb.active
    ws.title = "Ringkasan"
    build_ringkasan(ws, xl, full)
    build_kamus(wb.create_sheet("Kamus_Data"))
    for name, df in sheets.items():
        write_df(wb.create_sheet(name), df)
    write_df(wb.create_sheet("HB_terpisah"), hb)
    write_df(wb.create_sheet("dihapus"), dih)
    write_df(wb.create_sheet("peta_outlet_id"), peta, band_by="__none__")

    # warna tab sheet
    tabs = {"Ringkasan": "0F172A", "Kamus_Data": "0F172A", "dataset_model": "0369A1",
            "model_lr": "6D28D9", "model_rf": "6D28D9", "model_gb": "6D28D9",
            "HB_terpisah": "B45309", "dihapus": "B91C1C", "peta_outlet_id": "475569"}
    for n, col in tabs.items():
        wb[n].sheet_properties.tabColor = col

    try:
        wb.save(SRC)
        if NEW.exists():
            NEW.unlink()
        print("Tersimpan:", SRC.name)
    except PermissionError:
        wb.save(NEW)
        print("File asli sedang terbuka di Excel -> tersimpan sebagai:", NEW.name)


if __name__ == "__main__":
    main()
