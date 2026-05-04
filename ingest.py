# ingest.py — rode UMA única vez para popular o MongoDB
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from db_manager import HateSpeechDB

# Caminho do CSV
BASE_DIR = Path(__file__).resolve().parent
caminho = BASE_DIR / "Bases_de_dados" / "ToLD-BR.csv"

# Carrega o CSV
print("Lendo CSV...")
df = pd.read_csv(caminho)
print(f"Total de registros: {len(df)}")

# Divide em treino e teste (igual ao app.py)
train, test = train_test_split(df, test_size=0.3, random_state=42)

# Conecta ao MongoDB e insere
db = HateSpeechDB()

# Limpa a coleção antes de reinserir (evita duplicatas)
db.collection.drop()
print("Coleção limpa.")

# Insere
db.insert_from_dataframe(train, source="ToLD-BR", split="train")
db.insert_from_dataframe(test,  source="ToLD-BR", split="test")

print(f"\nTotal de documentos no MongoDB: {db.total_documentos()}")
print("Ingestão concluída!")