import os
import cv2
import tkinter as tk
from tkinter import messagebox, simpledialog
from PIL import Image, ImageTk
from rdkit import Chem
from rdkit.Chem import Draw

# Variáveis globais
cap = None
frame_atual = None
CAMERA_INDEX = 0
mol_image = None

# Tamanho da visualização da câmera
CAMERA_WIDTH = 320
CAMERA_HEIGHT = 240

# ---------- Funções ----------

def ligar_camera():
    global cap
    cap = cv2.VideoCapture(CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
    mostrar_video()

def mostrar_video():
    global cap, frame_atual
    if cap is not None and cap.isOpened():
        ret, frame = cap.read()
        if ret:
            frame_atual = frame
            # Redimensionar frame para caber na tela
            frame_resized = cv2.resize(frame, (CAMERA_WIDTH, CAMERA_HEIGHT))
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

def gerar_molecula():
    global mol_image
    if frame_atual is None:
        messagebox.showwarning("Aviso", "Primeiro capture uma foto!")
        return

    # Receber o SMILES manualmente
    smiles = simpledialog.askstring("Entrada de SMILES", "Digite o SMILES da molécula:")
    if not smiles:
        messagebox.showwarning("Aviso", "Nenhum SMILES fornecido!")
        return

    # Criar objeto molécula e imagem com RDKit
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        messagebox.showerror("Erro", "SMILES inválido!")
        return

    img = Draw.MolToImage(mol, size=(300, 300))
    mol_image = ImageTk.PhotoImage(img)
    
    lbl_mol.configure(image=mol_image)
    lbl_mol.image = mol_image
    lbl_smiles.config(text=f"SMILES: {smiles}")

# ---------- Interface Tkinter ----------

janela = tk.Tk()
janela.title("Webcam → Molécula (RDKit)")

# Foto capturada
lbl_foto = tk.Label(janela)
lbl_foto.pack(pady=5)

# Botões
btn_ligar = tk.Button(janela, text="Ligar a câmera", command=ligar_camera)
btn_ligar.pack(pady=5)

btn_foto = tk.Button(janela, text="Tirar foto", command=tirar_foto)
btn_foto.pack(pady=5)

btn_gerar = tk.Button(janela, text="Gerar molécula", command=gerar_molecula)
btn_gerar.pack(pady=5)

# Imagem da molécula
lbl_mol = tk.Label(janela)
lbl_mol.pack(pady=5)

# SMILES
lbl_smiles = tk.Label(janela, text="SMILES: ")
lbl_smiles.pack(pady=5)

janela.mainloop()

# ---------- Finalização ----------
if cap is not None:
    cap.release()
cv2.destroyAllWindows()
