import os
import cv2
import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk
import torch
from molscribe import MolScribe
from huggingface_hub import hf_hub_download
from rdkit import Chem
from rdkit.Chem import Draw

# ====== Carregar modelo MolScribe ======
ckpt = hf_hub_download('yujieq/MolScribe', 'swin_base_char_aux_1m.pth')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = MolScribe(ckpt, device=device)

# ====== Carregar banco de SMILES ======
DATABASE_FILE = "banco_smiles.txt"

def carregar_banco():
    banco = {}
    if not os.path.exists(DATABASE_FILE):
        messagebox.showerror("Erro", f"Arquivo {DATABASE_FILE} não encontrado!")
        return banco
    with open(DATABASE_FILE, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                try:
                    nome, smiles = line.strip().split(";")
                    smiles = smiles.strip()
                    nome = nome.strip()
                    if smiles in banco:
                        banco[smiles].append(nome)
                    else:
                        banco[smiles] = [nome]
                except ValueError:
                    continue
    return banco

banco_smiles = carregar_banco()

# ====== Variáveis globais ======
cap = None
frame_atual = None

# ====== Funções ======
def ligar_camera():
    global cap
    cap = cv2.VideoCapture(0)
    atualizar_frame()

def atualizar_frame():
    global frame_atual
    if cap is not None and cap.isOpened():
        ret, frame = cap.read()
        if ret:
            frame_atual = frame
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(img)
            img = img.resize((400, 300))  # tamanho menor
            imgtk = ImageTk.PhotoImage(img)
            lbl_video.imgtk = imgtk
            lbl_video.configure(image=imgtk)
    lbl_video.after(10, atualizar_frame)

def interpretar():
    if frame_atual is None:
        messagebox.showwarning("Aviso", "Nenhuma imagem capturada!")
        return

    temp_path = "tmp_capture.png"
    cv2.imwrite(temp_path, frame_atual)

    # Usar MolScribe para gerar SMILES
    resultado = model.predict_image_file(
        temp_path,
        return_atoms_bonds=False,
        return_confidence=False
    )

    smiles = resultado.get('smiles', None)
    if not smiles:
        messagebox.showerror("Erro", "Não foi possível extrair SMILES.")
        return

    lbl_smiles.config(text=f"SMILES: {smiles}")

    nomes = banco_smiles.get(smiles, ["NOME NÃO ENCONTRADO"])
    lbl_nome.config(text="Nome(s): " + ", ".join(nomes))

    mol = Chem.MolFromSmiles(smiles)
    if mol:
        img = Draw.MolToImage(mol, size=(300, 300))
        imgtk = ImageTk.PhotoImage(img)
        lbl_mol.imgtk = imgtk
        lbl_mol.configure(image=imgtk)
    else:
        messagebox.showerror("Erro", "SMILES inválido para RDKit.")

# ====== Interface ======
root = tk.Tk()
root.title("MolScribe + RDKit")

lbl_video = tk.Label(root)
lbl_video.pack()

btn_ligar = tk.Button(root, text="Ligar câmera", command=ligar_camera)
btn_ligar.pack(pady=5)

btn_interpretar = tk.Button(root, text="Interpretar e gerar", command=interpretar)
btn_interpretar.pack(pady=5)

lbl_smiles = tk.Label(root, text="SMILES: ")
lbl_smiles.pack(pady=5)

lbl_nome = tk.Label(root, text="Nome(s): ")
lbl_nome.pack(pady=5)

lbl_mol = tk.Label(root)
lbl_mol.pack(pady=5)

root.mainloop()

if cap:
    cap.release()
cv2.destroyAllWindows()
