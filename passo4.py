import os
import cv2
import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk
import torch
from baixarmodelo import MolScribe
from huggingface_hub import hf_hub_download
from rdkit import Chem
from rdkit.Chem import Descriptors

# =========================
# Setup do MolScribe
# =========================
ckpt_path = hf_hub_download('yujieq/MolScribe', 'swin_base_char_aux_1m.pth')
model = MolScribe(ckpt_path, device=torch.device('cpu'))  # use 'cuda' se tiver GPU

# =========================
# Variáveis globais
# =========================
cap = None
frame_atual = None
SMILES = None
CAMERA_INDEX = 0  # 0 = webcam do notebook, 1 = externa

# =========================
# Funções da câmera
# =========================
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

# =========================
# Função de interpretação SMILES + Nome da molécula
# =========================
def interpretar():
    global SMILES, frame_atual
    if frame_atual is None:
        messagebox.showwarning("Aviso", "Primeiro capture uma foto!")
        return

    # Salvar frame temporário
    path_temp = "tmp.png"
    cv2.imwrite(path_temp, frame_atual)

    # Gerar SMILES
    resultado = model.predict_image_file(
        path_temp,
        return_atoms_bonds=False,
        return_confidence=False
    )

    SMILES = resultado.get('smiles', None)

    if SMILES:
        # Tentar identificar a molécula pelo SMILES
        mol = Chem.MolFromSmiles(SMILES)
        if mol:
            # Obter fórmula molecular
            formula = Chem.rdMolDescriptors.CalcMolFormula(mol)
            messagebox.showinfo("Resultado", f"SMILES: {SMILES}\nFórmula molecular: {formula}")
        else:
            messagebox.showinfo("SMILES Gerado", f"SMILES: {SMILES}\nNão foi possível identificar a molécula.")
    else:
        messagebox.showwarning("Erro", "Não foi possível extrair SMILES da imagem.")

# =========================
# Interface Tkinter
# =========================
janela = tk.Tk()
janela.title("Webcam → MolScribe")

lbl_video = tk.Label(janela)
lbl_video.pack()

btn_ligar = tk.Button(janela, text="Ligar a câmera", command=ligar_camera)
btn_ligar.pack(pady=5)

btn_foto = tk.Button(janela, text="Tirar a foto", command=tirar_foto)
btn_foto.pack(pady=5)

btn_interpretar = tk.Button(janela, text="Interpretar (SMILES + Nome)", command=interpretar)
btn_interpretar.pack(pady=5)

janela.mainloop()

# =========================
# Finalização
# =========================
if cap is not None:
    cap.release()
cv2.destroyAllWindows()
