import streamlit as st
import tempfile
from PIL import Image, ImageOps
from molscribe import MolScribe
from huggingface_hub import hf_hub_download
from rdkit import Chem
from rdkit.Chem import Draw
import torch
import os
import streamlit.components.v1 as components

def falar_texto(texto):
    js_code = f"""
    <script>
    (function() {{
        var msg = new SpeechSynthesisUtterance("{texto}");
        msg.lang = "pt-BR";
        msg.rate = 1.0;
        window.speechSynthesis.cancel();
        window.speechSynthesis.speak(msg);
    }})();
    </script>
    """
    components.html(js_code, height=0, width=0)

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

st.title("Quimitec")
st.write("Reconhecimento de moléculas a partir de imagens.")
st.divider()

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Captura da molécula")
    img_capturada = st.camera_input("Use a câmera para tirar a foto:")

    if img_capturada is not None:
        st.success("Foto capturada!")

        st.markdown("### Ajuste de contraste")
        img_pil = Image.open(img_capturada).resize((400, 300))

        img_gray = img_pil.convert("L")
        block_size = 15
        offset = 10
        img_bw = Image.new("1", img_gray.size)
        pixels = img_gray.load()
        pixels_bw = img_bw.load()
        width, height = img_gray.size

        for y in range(height):
            for x in range(width):
                x0 = max(x - block_size // 2, 0)
                y0 = max(y - block_size // 2, 0)
                x1 = min(x + block_size // 2, width - 1)
                y1 = min(y + block_size // 2, height - 1)

                total = 0
                count = 0
                for j in range(y0, y1 + 1):
                    for i in range(x0, x1 + 1):
                        total += pixels[i, j]
                        count += 1
                media_local = total // count
                pixels_bw[x, y] = 0 if pixels[x, y] < (media_local - offset) else 1

        img_final = img_bw.convert("RGB")
        st.image(img_final, caption="Imagem ajustada")

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmpfile:
            img_final.save(tmpfile.name)
            tmp_filename = tmpfile.name

        if st.button("Interpretar imagem"):
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
    st.subheader("Resultado da análise")

    if "resultado" in st.session_state and st.session_state.resultado:
        smiles = st.session_state.resultado.get('smiles', None)

        if not smiles:
            st.error("Não foi possível extrair SMILES. Tente outra imagem.")
        else:
            st.success("Interpretação concluída!")
            st.markdown(f"**SMILES:** `{smiles}`")

            nomes = banco_smiles.get(smiles, ["NOME NÃO ENCONTRADO"])
            st.info("**Nome:** " + ", ".join(nomes))

            texto_para_falar = f"A molécula identificada é {', '.join(nomes)}."
            falar_texto(texto_para_falar)

            mol = Chem.MolFromSmiles(smiles)
            if mol:
                img_mol = Draw.MolToImage(mol, size=(350, 350))
                st.image(img_mol, caption="Molécula interpretada")
