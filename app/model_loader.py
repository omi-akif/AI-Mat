import torch
import bentoml
import mlflow
from app.config import settings

class TwoTowerInference:
    def __init__(self):
        print("Loading Two Tower model...")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        try:
            self.service = bentoml.mlflow.load_model(f"{settings.MODEL_NAME_TWO_TOWER}:latest")
            self.model = self.service._model_impl.get_raw_model()
            self.model.to(self.device)
            self.model.eval()
        except Exception as e:
            print(f"Error loading Two Tower model: {e}")
            raise e
            
        print("Two Tower model loaded successfully.")

    def get_candidate_embedding(self, feature_vector):
        # Convert list to tensor if needed
        if not isinstance(feature_vector, torch.Tensor):
            feature_vector = torch.tensor(feature_vector, dtype=torch.float32)

        # Add batch dimension (required by model)
        if feature_vector.dim() == 1:
            feature_vector = feature_vector.unsqueeze(0)

        # Move to same device as model
        feature_vector = feature_vector.to(self.device)

        with torch.no_grad():
            embedding = self.model.get_candidate_embeddings(feature_vector)
            embedding = embedding.detach().cpu()

        return embedding.squeeze(0).tolist()

    def get_job_embedding(self, feature_vector):
        # Convert list to tensor if needed
        if not isinstance(feature_vector, torch.Tensor):
            feature_vector = torch.tensor(feature_vector, dtype=torch.float32)

        # Add batch dimension (required by model)
        if feature_vector.dim() == 1:
            feature_vector = feature_vector.unsqueeze(0)

        # Move to same device as model
        feature_vector = feature_vector.to(self.device)

        with torch.no_grad():
            embedding = self.model.get_job_embeddings(feature_vector)
            embedding = embedding.detach().cpu()

        return embedding.squeeze(0).tolist()

two_tower_inference = TwoTowerInference()
