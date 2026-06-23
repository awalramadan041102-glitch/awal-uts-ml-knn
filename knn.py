import math
import tkinter as tk

# =====================================================================
# 1. PENJELASAN MATEMATIKA & ENGINE KNN (TANPA LIBRARY)
# =====================================================================
def hitung_jarak(titik1, titik2):
    """Menghitung Euclidean Distance antara dua titik"""
    total = 0
    for i in range(len(titik1)):
        total += (titik1[i] - titik2[i]) ** 2
    return math.sqrt(total)

def prediksi_knn(data_latih, data_uji_fitur, k=3):
    """Memprediksi kelas dengan voting mayoritas dari K tetangga terdekat"""
    jarak_list = []
    
    for baris in data_latih:
        fitur_latih = baris[:-1]
        label_latih = baris[-1]
        jarak = hitung_jarak(data_uji_fitur, fitur_latih)
        jarak_list.append((jarak, label_latih))
    
    # Urutkan jarak dari yang terkecil (Ascending)
    jarak_list.sort(key=lambda x: x[0])
    
    # Ambil K tetangga terdekat
    tetangga = jarak_list[:k]
    
    # Voting Mayoritas
    counts = {}
    for t in tetangga:
        label = t[1]
        counts[label] = counts.get(label, 0) + 1
        
    label_terpilih = max(counts, key=counts.get)
    return label_terpilih


# =====================================================================
# 2. PARSER KHUSUS DATASET HEART DISEASE UCI (PENGGANTI PANDAS)
# =====================================================================
def muat_data_heart_disease(nama_file):
    """Membaca data format spasi & multiline spesifik dataset Heart Disease UCI"""
    dataset = []
    semua_angka = []
    
    with open(nama_file, 'r') as file:
        for baris in file:
            bagian = baris.split()
            for item in bagian:
                if item.isalpha() or 'name' in item:
                    if len(semua_angka) >= 14:
                        umur = semua_angka[2]       # Fitur X
                        tt_darah = semua_angka[9]   # Fitur Y
                        num_label = semua_angka[13] # Label asli 0-4
                        
                        if umur != -9 and tt_darah != -9 and num_label != -9:
                            label_biner = 0 if num_label == 0 else 1
                            dataset.append([umur, tt_darah, label_biner])
                            
                    semua_angka = []
                else:
                    try:
                        semua_angka.append(float(item))
                    except ValueError:
                        continue
                        
    return dataset


# =====================================================================
# 3. VISUALISASI DECISION BOUNDARY INTERAKTIF (MENGGUNAKAN TKINTER)
# =====================================================================
class AplikasiKNNInteraktif(tk.Tk):
    def __init__(self, dataset, k_awal=3):
        super().__init__()
        self.title("UTS Machine Learning - Eksperimen Nilai K KNN")
        self.geometry("700x700")
        self.resizable(False, False)
        
        self.dataset = dataset
        self.k_value = k_awal
        
        self.canvas_lebar = 600
        self.canvas_tinggi = 500
        
        # Ekstrak nilai min/max untuk scaling koordinat
        self.x_vals = [b[0] for b in dataset]
        self.y_vals = [b[1] for b in dataset]
        self.min_x, self.max_x = min(self.x_vals), max(self.x_vals)
        self.min_y, self.max_y = min(self.y_vals), max(self.y_vals)
        
        self.buat_widget()
        self.render_grafik()

    def ubah_ke_pixel(self, x, y):
        pad = 50  
        range_x = (self.max_x - self.min_x) if (self.max_x - self.min_x) != 0 else 1
        range_y = (self.max_y - self.min_y) if (self.max_y - self.min_y) != 0 else 1
        px = pad + ((x - self.min_x) / range_x) * (self.canvas_lebar - 2 * pad)
        py = self.canvas_tinggi - (pad + ((y - self.min_y) / range_y) * (self.canvas_tinggi - 2 * pad))
        return px, py

    def ubah_ke_fitur(self, px, py):
        pad = 50
        range_x = (self.max_x - self.min_x) if (self.max_x - self.min_x) != 0 else 1
        range_y = (self.max_y - self.min_y) if (self.max_y - self.min_y) != 0 else 1
        x = self.min_x + ((px - pad) / (self.canvas_lebar - 2 * pad)) * range_x
        inv_py = self.canvas_tinggi - py
        y = self.min_y + ((inv_py - pad) / (self.canvas_tinggi - 2 * pad)) * range_y
        return x, y

    def buat_widget(self):
        # Panel Atas: Judul Dinamis yang berubah nilainya sesuai K
        self.label_judul = tk.Label(self, text=f"Decision Boundary KNN dengan Nilai K = {self.k_value}", font=("Arial", 12, "bold"))
        self.label_judul.pack(pady=5)
        
        # Canvas Utama
        self.canvas = tk.Canvas(self, width=self.canvas_lebar, height=self.canvas_tinggi, bg="white", highlightthickness=1, highlightbackground="black")
        self.canvas.pack()
        
        # Legenda Warna
        legenda = tk.Label(self, text="Legenda: Merah Muda/Merah = Sehat (0)  |  Biru Muda/Biru = Sakit Jantung (1)", font=("Arial", 9, "italic"))
        legenda.pack(pady=5)
        
        # Panel Kontrol Bawah: Mengubah nilai K secara interaktif saat presentasi
        frame_kontrol = tk.Frame(self)
        frame_kontrol.pack(pady=10)
        
        lbl_input = tk.Label(frame_kontrol, text="Masukkan Nilai K baru (Angka Ganjil): ", font=("Arial", 10))
        lbl_input.pack(side=tk.LEFT)
        
        # Kotak input angka K
        self.input_k = tk.Entry(frame_kontrol, width=5, font=("Arial", 10), justify='center')
        self.input_k.insert(0, str(self.k_value))
        self.input_k.pack(side=tk.LEFT, padx=5)  # <-- Sudah diperbaiki dari px menjadi padx
        
        # Tombol Update Gambar
        btn_update = tk.Button(frame_kontrol, text="Update Grafik", command=self.proses_perubahan_k, bg="#4CAF50", fg="white", font=("Arial", 10, "bold"))
        btn_update.pack(side=tk.LEFT, padx=5)  # <-- Sudah diperbaiki dari px menjadi padx

    def render_grafik(self):
        """Menghapus gambar lama dan merender ulang background serta titik data"""
        self.canvas.delete("all") # Bersihkan kanvas lama
        self.label_judul.config(text=f"Decision Boundary KNN dengan Nilai K = {self.k_value}")
        
        # 1. Gambar Ulang Boundary
        langkah = 12  # Resolusi scanning grid
        for px in range(0, self.canvas_lebar, langkah):
            for py in range(0, self.canvas_tinggi, langkah):
                x_fitur, y_fitur = self.ubah_ke_fitur(px, py)
                prediksi = prediksi_knn(self.dataset, [x_fitur, y_fitur], k=self.k_value)
                
                warna_bg = "#ffcccc" if prediksi == 0 else "#ccffff"
                self.canvas.create_rectangle(px, py, px + langkah, py + langkah, fill=warna_bg, outline="")
                
        # 2. Gambar Ulang Titik Data Asli
        r = 4  
        for baris in self.dataset:
            x, y, label = baris[0], baris[1], baris[2]
            px, py = self.ubah_ke_pixel(x, y)
            warna_titik = "red" if label == 0 else "blue"
            self.canvas.create_oval(px - r, py - r, px + r, py + r, fill=warna_titik, outline="black", width=1)

    def proses_perubahan_k(self):
        """Mengambil nilai baru dari kotak input Entry dan memvalidasinya sebelum render"""
        try:
            nilai_baru = int(self.input_k.get())
            if nilai_baru > 0 and nilai_baru <= len(self.dataset):
                self.k_value = nilai_baru
                self.render_grafik() # Jalankan proses kalkulasi ulang KNN secara realtime
            else:
                print("Nilai K harus lebih besar dari 0.")
        except ValueError:
            print("Masukkan angka integer ganjil yang valid.")


# =====================================================================
# 4. RUN PROGRAM UTAMA
# =====================================================================
if __name__ == "__main__":
    NAMA_FILE = "data.csv" 
    
    try:
        print(f">> Sedang memproses data dari {NAMA_FILE}...")
        data_points = muat_data_heart_disease(NAMA_FILE)
        print(f">> Berhasil memuat {len(data_points)} data pasien!")
        
        if len(data_points) > 0:
            app = AplikasiKNNInteraktif(dataset=data_points, k_awal=3)
            app.mainloop()
        else:
            print("Peringatan: Format data tidak cocok.")
            
    except FileNotFoundError:
        print(f"Error: File '{NAMA_FILE}' tidak ditemukan!")