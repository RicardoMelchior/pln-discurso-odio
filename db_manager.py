# db_manager.py
import os
import pandas as pd
from pymongo import MongoClient
from dotenv import load_dotenv
from pathlib import Path

# Garante que o .env é encontrado independente de onde o script é chamado
env_path = Path(__file__).resolve().parent / ".env"
load_dotenv(dotenv_path=env_path)

class HateSpeechDB:
    def __init__(self):
        uri = os.getenv("MONGO_URI")
        if not uri:
            raise ValueError("MONGO_URI não encontrada! Verifique o arquivo .env")
        self.client = MongoClient(uri)
        self.db = self.client["pln_hate_speech"]
        self.collection = self.db["samples"]

    def insert_from_dataframe(self, df: pd.DataFrame, source: str, split: str):
        """Insere registros de um DataFrame no MongoDB."""
        records = df.to_dict(orient="records")
        for r in records:
            r["source"] = source
            r["split"] = split
        self.collection.insert_many(records)
        print(f"[MongoDB] {len(records)} documentos inseridos — fonte: '{source}' / split: '{split}'")

    def get_split(self, split: str, source: str = None) -> pd.DataFrame:
        """Retorna os dados de treino ou teste como DataFrame."""
        query = {"split": split}
        if source:
            query["source"] = source
        cursor = self.collection.find(query, {"_id": 0})
        return pd.DataFrame(list(cursor))

    def total_documentos(self) -> int:
        return self.collection.count_documents({})