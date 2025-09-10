import streamlit as st
import tempfile
from PIL import Image, ImageEnhance
from molscribe import MolScribe
from huggingface_hub import hf_hub_download
from rdkit import Chem
from rdkit.Chem import Draw
import torch
import os

@st.cache_resource
def carregar_modelo():
    ckpt = hf_hub_download('yujieq/MolScribe', 'swin_base_char_aux_1m.pth')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return MolScribe(ckpt, device=device)

model = carregar_modelo()

DATABASE_FILE = "banco_smiles.txt"

def carregar_banco():
    banco = {}
    if not os.path.exists(DATABASE_FILE):
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

st.set_page_config(page_title="MolScribe + RDKit", layout="wide")

st.title("🧪Quimitec")
st.write("Reconhecimento de moléculas a partir de imagens para **SMILES** e nomes.")

st.divider()

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📸 Captura da molécula")
    img_capturada = st.camera_input("Use a câmera para tirar a foto:")

    if img_capturada is not None:
        st.success("✅ Foto capturada com sucesso!")

        st.markdown("### 🎨 Ajuste de contraste")
        img_pil = Image.open(img_capturada).resize((400, 300))
        enhancer = ImageEnhance.Contrast(img_pil)
        img_contrast = enhancer.enhance(2.0)
        st.image(img_contrast, caption="Imagem ajustada")

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmpfile:
            img_contrast.save(tmpfile.name)
            tmp_filename = tmpfile.name

        if st.button("🔍 Interpretar molécula"):
            with st.spinner("Processando imagem..."):
                try:
                    resultado = model.predict_image_file(
                        tmp_filename,
                        return_atoms_bonds=False,
                        return_confidence=False
                    )
                    st.session_state.resultado = resultado
                except Exception as e:
                    st.error(f"⚠️ Ops, erro: {e}")

with col2:
    st.subheader("📊 Resultado da análise")

    if "resultado" in st.session_state and st.session_state.resultado:
        smiles = st.session_state.resultado.get('smiles', None)

        if not smiles:
            st.error("⚠️ Não consegui extrair SMILES. Tente outra imagem.")
        else:
            st.success("✅ Interpretação concluída!")
            st.markdown(f"**SMILES:** `{smiles}`")

            nomes = banco_smiles.get(smiles, ["NOME NÃO ENCONTRADO"])
            st.info("**Nome:** " + ", ".join(nomes))

            mol = Chem.MolFromSmiles(smiles)
            if mol:
                img_mol = Draw.MolToImage(mol, size=(350, 350))
                st.image(img_mol, caption="Molécula interpretada")
    else:
        st.info("📌 Capture e interprete uma molécula para ver o resultado aqui.")
