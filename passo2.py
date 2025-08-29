import os
import cv2
import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk
import torch
from baixarmodelo import MolScribe
from huggingface_hub import hf_hub_download


ckpt_path = hf_hub_download('yujieq/MolScribe', 'swin_base_char_aux_1m.pth')
model = MolScribe(ckpt_path, device=torch.device('cpu'))  # troque para 'cuda' se tiver GPU

# Variáveis globais
cap = None
frame_atual = None
SMILES = None


def ligar_camera():
    global cap
    CAMERA_INDEX = 1  # mude se precisar (0 = notebook, 1 = externa, etc.)
    cap = cv2.VideoCapture(CAMERA_INDEX)
    mostrar_video()

def mostrar_video():
    global cap, frame_atual
    if cap is not None and cap.isOpened():
        ret, frame = cap.read()
        if ret:
            frame_atual = frame
            # Converter para exibir no Tkinter
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(img)
            imgtk = ImageTk.PhotoImage(image=img)
            lbl_video.imgtk = imgtk
            lbl_video.configure(image=imgtk)
    lbl_video.after(10, mostrar_video)

def tirar_foto():
    global frame_atual
    if frame_atual is not None:
        caminho = os.path.join(os.getcwd(), "foto_capturada.jpg")
        cv2.imwrite(caminho, frame_atual)
        messagebox.showinfo("Foto Capturada", f"Foto salva em:\n{caminho}")
    else:
        messagebox.showwarning("Aviso", "Nenhuma imagem capturada!")

def interpretar():
    global SMILES, frame_atual
    if frame_atual is None:
        messagebox.showwarning("Aviso", "Primeiro capture uma foto!")
        return

    # Salvar frame atual temporário
    path_temp = "tmp.png"
    cv2.imwrite(path_temp, frame_atual)

    # Rodar o MolScribe
    resultado = model.predict_image_file(
        path_temp, 
        return_atoms_bonds=False, 
        return_confidence=False
    )

    SMILES = resultado.get('smiles', None)

    if SMILES:
        messagebox.showinfo("SMILES Gerado", SMILES)
    else:
        messagebox.showwarning("Erro", "Não foi possível extrair SMILES da imagem.")

# ========================
# Interface Tkinter
# ========================
janela = tk.Tk()
janela.title("Webcam → MolScribe")

lbl_video = tk.Label(janela)
lbl_video.pack()

btn_ligar = tk.Button(janela, text="Ligar a câmera", command=ligar_camera)
btn_ligar.pack(pady=5)

btn_foto = tk.Button(janela, text="Tirar a foto", command=tirar_foto)
btn_foto.pack(pady=5)

btn_interpretar = tk.Button(janela, text="Interpretar (SMILES)", command=interpretar)
btn_interpretar.pack(pady=5)

janela.mainloop()

# ========================
# Finalização
# ========================
if cap is not None:
    cap.release()
cv2.destroyAllWindows()
