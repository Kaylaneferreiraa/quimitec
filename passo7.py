import os
import cv2
import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk
import torch
from molscribe import MolScribe
from rdkit import Chem
from rdkit.Chem import Draw

# ======================
# Configurações
# ======================
MODEL_LOCAL_PATH = r"C:\Users\Aluno\quimitec\modelos\swin_base_char_aux_1m.pth"  # ajuste se necessário
CAMERA_INDEX = 0  # 0 = notebook, 1 = externa

# Variáveis globais
cap = None
frame_atual = None
SMILES = None
mol_image = None

# ======================
# Carregar modelo MolScribe
# ======================
model = MolScribe(model_path=MODEL_LOCAL_PATH, device=torch.device("cpu"))

# ======================
# Funções
# ======================
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
            # Reduz o tamanho da imagem para caber melhor na tela
            frame_resized = cv2.resize(frame, (320, 240))
            img = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(img)
            imgtk = ImageTk.PhotoImage(image=img)
            lbl_foto.imgtk = imgtk
            lbl_foto.configure(image=imgtk)
    lbl_foto.after(10, mostrar_video)

def tirar_foto():
    global frame_atual
    if frame_atual is not None:
        caminho = os.path.join(os.getcwd(), "foto_capturada.jpg")
        cv2.imwrite(caminho, frame_atual)
        messagebox.showinfo("Foto Capturada", f"Foto salva em:\n{caminho}")
    else:
        messagebox.showwarning("Aviso", "Nenhuma imagem capturada!")

def interpretar():
    global SMILES, mol_image, frame_atual
    if frame_atual is None:
        messagebox.showwarning("Aviso", "Primeiro capture uma foto!")
        return

    # Salvar frame temporário
    path_temp = "tmp.png"
    cv2.imwrite(path_temp, frame_atual)

    # Gerar SMILES com MolScribe
    resultado = model.predict_image_file(path_temp, return_atoms_bonds=False, return_confidence=False)
    SMILES = resultado.get('smiles', None)

    if SMILES:
        lbl_smiles.config(text=f"SMILES: {SMILES}")

        # Gerar imagem da molécula com RDKit
        mol = Chem.MolFromSmiles(SMILES)
        if mol is not None:
            img = Draw.MolToImage(mol, size=(300, 300))
            mol_image = ImageTk.PhotoImage(img)
            lbl_mol.configure(image=mol_image)
            lbl_mol.image = mol_image
        else:
            messagebox.showwarning("Erro", "Não foi possível gerar imagem da molécula!")
    else:
        messagebox.showwarning("Erro", "Não foi possível extrair SMILES da imagem!")

# ======================
# Interface Tkinter
# ======================
janela = tk.Tk()
janela.title("Webcam → MolScribe + RDKit")

# Label para mostrar a foto da câmera
lbl_foto = tk.Label(janela)
lbl_foto.pack(pady=5)

# Botões
btn_ligar = tk.Button(janela, text="Ligar a câmera", command=ligar_camera)
btn_ligar.pack(pady=5)

btn_foto = tk.Button(janela, text="Tirar a foto", command=tirar_foto)
btn_foto.pack(pady=5)

btn_interpretar = tk.Button(janela, text="Interpretar (SMILES + Imagem)", command=interpretar)
btn_interpretar.pack(pady=5)

# Label para imagem da molécula
lbl_mol = tk.Label(janela)
lbl_mol.pack(pady=5)

# Label para mostrar SMILES
lbl_smiles = tk.Label(janela, text="SMILES: ")
lbl_smiles.pack(pady=5)

# ======================
# Finalização
# ======================
janela.mainloop()
if cap is not None:
    cap.release()
cv2.destroyAllWindows()
