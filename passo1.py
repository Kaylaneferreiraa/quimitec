import cv2
import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk

# Variáveis globais
cap = None
frame_atual = None

# >>> MUDA AQUI se precisar: 0 = webcam notebook, 1 = webcam externa
CAMERA_INDEX = 0  

def ligar_camera():
    global cap
    cap = cv2.VideoCapture(CAMERA_INDEX)
    mostrar_video()

def mostrar_video():
    global cap, frame_atual
    if cap is not None and cap.isOpened():
        ret, frame = cap.read()
        if ret:
            frame_atual = frame
            # Converte para formato do Tkinter
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(img)
            imgtk = ImageTk.PhotoImage(image=img)
            lbl_video.imgtk = imgtk
            lbl_video.configure(image=imgtk)
    lbl_video.after(10, mostrar_video)

def tirar_foto():
    global frame_atual
    if frame_atual is not None:
        cv2.imwrite("foto_capturada.jpg", frame_atual)
        messagebox.showinfo("Sucesso", "Foto salva como foto_capturada.jpg")
    else:
        messagebox.showwarning("Aviso", "Nenhuma imagem capturada!")

def interpretar():
    messagebox.showinfo("Interpretar", "Função de interpretação será implementada depois!")

# Interface Tkinter
janela = tk.Tk()
janela.title("Captura de Foto pela Webcam")

lbl_video = tk.Label(janela)
lbl_video.pack()

btn_ligar = tk.Button(janela, text="Ligar a câmera", command=ligar_camera)
btn_ligar.pack(pady=5)

btn_foto = tk.Button(janela, text="Tirar a foto", command=tirar_foto)
btn_foto.pack(pady=5)

btn_interpretar = tk.Button(janela, text="Interpretar", command=interpretar)
btn_interpretar.pack(pady=5)

janela.mainloop()

# Libera a câmera ao fechar
if cap is not None:
    cap.release()
cv2.destroyAllWindows()