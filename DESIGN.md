# DESIGN.md: Sistem Telemetri Prediksi Dropping Outlet

## Design Read
Reading this as: **aplikasi analitik dan telemetri peramalan distribusi barang (dropping outlet)** untuk **analis data dan pengambil keputusan operasional**, dalam gaya **Industrial Telemetry / High-Precision Instrument** (futuristic karena presisi instrumen, sudut terpangkas/chamfered, tipografi teknis, dan angka tabular; bukan karena dekorasi neon atau gradien pelangi), dial **ENERGY 2 / RHYTHM 2 / MOTION 1**.

## Tiga Dial (Liveliness Toolkit)
- **ENERGY: 2 (Balanced)**: Kontras hierarki yang tegas antara angka metrik utama, grafik interaktif, dan tabel rincian tanpa efek cahaya berlebihan.
- **RHYTHM: 2 (Consistent with deliberate breaks)**: Setiap halaman dibuka dengan pita ringkasan/keputusan utama yang asimetris atau terfokus (bukan sekadar deretan kartu seragam), diikuti komposisi grafik yang menyesuaikan bobot konten (misalnya grafik rentang waktu utama memanjang penuh, sedangkan diagnostik residual berdampingan).
- **MOTION: 1 (Hover & focus states only)**: Tanpa animasi berulang tanpa henti (tidak ada titik berkedip terus-menerus). Transisi hanya pada interaksi kontrol dan menghormati `prefers-reduced-motion: reduce`.

## Palet Warna (R-01, R-21, R-25, R-29, R-34)
Maksimal 2 warna inti netral + 1 warna aksen utama (`Cyan Telemetri`), ditambah penanda seri grafik yang teruji kontrasnya di kedua tema (Dark & Light mode berfungsi penuh):

### Mode Gelap (Default: Ruang Kendali Telemetri)
- **Kanvas Utama (`--bg`)**: `#090D16` (Obsidian pekat)
- **Permukaan Panel (`--surface`)**: `#111827` (Slate instrumen padat, tanpa efek blur masif)
- **Permukaan Sekunder (`--surface-2`)**: `#1E293B`
- **Teks Utama (`--text`)**: `#F1F5F9` (Kontras 17.74:1 di atas `--bg`, 16.19:1 di atas `--surface` -> LULUS WCAG AAA)
- **Teks Sekunder (`--muted`)**: `#94A3B8` (Kontras 6.92:1 di atas `--surface` -> LULUS WCAG AA 4.5:1)
- **Garis Tepi (`--border`)**: `#334155` (Kontras > 3:1 untuk batas komponen aktif)
- **Aksen Tunggal (`--accent`)**: `#38BDF8` (Kontras 8.28:1 di atas `--surface` -> LULUS WCAG AA)

### Mode Terang (Pilihan Siang Hari / Laporan Cetak)
- **Kanvas Utama (`--bg`)**: `#F8FAFC`
- **Permukaan Panel (`--surface`)**: `#FFFFFF`
- **Permukaan Sekunder (`--surface-2`)**: `#F1F5F9`
- **Teks Utama (`--text`)**: `#0F172A` (Kontras 17.43:1 di atas `--surface` -> LULUS WCAG AAA)
- **Teks Sekunder (`--muted`)**: `#475569` (Kontras 7.58:1 di atas `--surface` -> LULUS WCAG AA)
- **Garis Tepi (`--border`)**: `#CBD5E1` (Batas panel) & `#64748B` (Batas kontrol interaktif >= 3:1)
- **Aksen Tunggal (`--accent`)**: `#0369A1` (Kontras 5.93:1 di atas `#FFFFFF` -> LULUS WCAG AA)

## Tipografi (R-06)
- **Heading & Identitas (`Chakra Petch`)**: Huruf geometris dengan sudut teknis (chamfered) yang memberi karakter instrumen futuristik tanpa mengorbankan keterbacaan seperti huruf display sci-fi ekstrem.
- **Teks Antarmuka (`Plus Jakarta Sans`)**: Dirancang untuk keterbacaan tinggi pada tabel padat dan label bahasa Indonesia.
- **Angka & Metrik (`IBM Plex Mono`, `tabular-nums`)**: Memastikan nilai MAE, RMSE, R², dan angka penjualan sejajar vertikal saat dibandingkan di tabel maupun kartu metrik.

## Motif Identitas & Alasan Keputusan (R-31)
1. **Warna (2 netral + 1 aksen cyan)**: Memusatkan perhatian mata hanya pada elemen aktif, batas awal prediksi, dan seri aktual tanpa polusi gradien ungu-biru.
2. **Sudut Panel Chamfered (`clip-path` / aksen siku presisi pada panel fokus)**: Menjadi motif identitas berulang yang membedakan antarmuka telemetri ini dari template dashboard generik.
3. **Tanpa Glassmorphism / Glow Masif (R-10, R-13)**: Semua kartu menggunakan latar belakang padat agar grafik dan angka memiliki kontras maksimum; elevasi hanya diberikan pada bilah navigasi lekat (sticky header).
4. **Judul Grafik Berupa Pertanyaan Analitis (`antislop-ui`)**: Setiap grafik mencantumkan pertanyaan atau keputusan yang dijawabnya, bukan sekadar label satu kata.
5. **Grafik Terpadu Sebelum vs Sesudah Prediksi**: Pada halaman Prediksi, data historis sebelum tanggal awal prediksi digabungkan langsung dengan data aktual dan hasil 4 model setelah tanggal awal prediksi dalam satu kanvas (baik agregat maupun per outlet) dengan garis batas vertikal yang jelas.
