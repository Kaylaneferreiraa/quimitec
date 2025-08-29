from huggingface_hub import hf_hub_download

# Baixa o modelo e salva na pasta 'modelos' do projeto
model_path = hf_hub_download(
    repo_id="yujieq/MolScribe",      # repositório do modelo
    filename="swin_base_char_aux_1m.pth",  # arquivo do modelo
    cache_dir="./modelos"            # pasta local onde será salvo
)

print("Modelo baixado para:", model_path)
