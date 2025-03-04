import cv2  # type: ignore
import time
import torch # type: ignore
import threading
from ultralytics import YOLO  # type: ignore

# Load model YOLOv8 hasil training
model_path = r"D:\yov8\bebas\best.engine"  # Ubah sesuai lokasi model
model = YOLO(model_path)

# Gunakan GPU jika tersedia
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model.to(device)
print(f"Model berjalan di: {device.upper()}")

# Kelas untuk menangani video secara threading
class VideoStream:
    def __init__(self, src=0):
        self.cap = cv2.VideoCapture(src, cv2.CAP_DSHOW)  # Backend DirectShow untuk Windows
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)  # Set resolusi kamera
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 30)  # Set FPS kamera lebih tinggi
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))  # Format MJPG lebih cepat
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Kurangi buffer kamera agar tidak delay

        self.ret, self.frame = self.cap.read()
        self.stopped = False
        self.thread = threading.Thread(target=self.update, args=())
        self.thread.daemon = True
        self.thread.start()

    def update(self):
        """Loop untuk membaca frame secara terus-menerus."""
        while not self.stopped:
            for _ in range(5):  # Buang 5 frame lama untuk mengurangi delay
                self.cap.read()
            self.ret, self.frame = self.cap.read()

    def read(self):
        """Mengembalikan frame terbaru."""
        return self.ret, self.frame

    def stop(self):
        """Berhenti membaca frame."""
        self.stopped = True
        self.cap.release()

# Inisialisasi stream video
stream = VideoStream()

# Variabel untuk menghitung FPS
prev_time = time.time() - 1  # Beri jeda awal agar tidak nol

# Loop utama
while True:
    ret, frame = stream.read() # type: ignore
    if not ret:
        print("Gagal membaca frame dari kamera.")
        break

    # Hitung FPS dengan validasi untuk mencegah pembagian nol
    curr_time = time.time()
    delta_time = curr_time - prev_time
    fps = 1 / delta_time if delta_time > 0 else 0
    prev_time = curr_time  # Update waktu sebelumnya

    # Deteksi objek dengan tracking agar lebih cepat
    results = model.track(frame, persist=True, conf=0.7, device=device)

    # Salin frame untuk anotasi
    annotated_frame = frame.copy()

    # Loop melalui hasil deteksi dan tambahkan bounding box + label
    for result in results:
        if hasattr(result, 'boxes') and result.boxes is not None:
            for box, cls, conf in zip(result.boxes.xyxy, result.boxes.cls, result.boxes.conf):
                x1, y1, x2, y2 = map(int, box)  # Koordinat bounding box
                class_id = int(cls)  # ID kelas
                confidence = float(conf)  # Confidence score

                # Ambil nama kelas berdasarkan model hasil training
                class_name = model.names.get(class_id, "Unknown")

                # Format label
                label = f"{class_name} ({confidence * 100:.2f}%)"
                coord_text = f"({x1}, {y1}) - ({x2}, {y2})"

                # Gambar bounding box dan label pada frame
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                # Tambahkan teks label dan koordinat
                cv2.putText(annotated_frame, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 3, cv2.LINE_AA)  # Outline putih
                cv2.putText(annotated_frame, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)  # Teks hitam

                cv2.putText(annotated_frame, coord_text, (x1, y2 + 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 3, cv2.LINE_AA)  # Outline putih
                cv2.putText(annotated_frame, coord_text, (x1, y2 + 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)  # Teks hitam

    # Tambahkan FPS di tampilan
    fps_text = f"FPS: {fps:.2f}"
    cv2.putText(annotated_frame, fps_text, (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 3, cv2.LINE_AA)  # Outline putih
    cv2.putText(annotated_frame, fps_text, (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2, cv2.LINE_AA)  # Teks hitam

    # Tampilkan hasil deteksi
    cv2.imshow('Deteksi Sampah dengan Model Custom', annotated_frame)

    # Tekan 'q' untuk keluar
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break
    

# Hentikan video stream dan tutup semua jendela
stream.stop() # type: ignore
cv2.destroyAllWindows()
cv2.waitKey(1)
