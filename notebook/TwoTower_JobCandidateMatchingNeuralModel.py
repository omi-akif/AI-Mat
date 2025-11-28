import marimo

__generated_with = "0.17.7"
app = marimo.App(width="full")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # AI Job-Candidate Matching using Two-Tower Architecture
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The two-tower model architecture consists of two seperate feed forward (MLP) neural networks that are able to take feature vectors and create low-dimensional dense embeddings. One is used for one type of entity and another is used for another entity. In our case, one neural network feeds on job features and another neural network feeds on candidate features.

    The specialty of this architecture is that it provides more accurate representation of the features of jobs and candidates which can be compared to provide how similar they are through the embeddings that are gotten as output in the last layer. Since Job and Candidate are two different entities and not directly comparable, two-neural networks are created. Through supervised training, the model should be able to perform more accurate similarity and can be used for any type of recommendation system or matching systems.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We will first call all the necessary modules to load data and do some data wrangling and data extraction
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Loading Data and Libraries
    """)
    return


@app.cell
def _():
    # # Loading Python Libraries
    # # import os
    # # import ast
    # # import requests
    import pandas as pd
    import bentoml
    import mlflow
    # import torch
    # from torch.utils.data import DataLoader, Dataset
    # # import json
    # # import warnings

    # # warnings.filterwarnings('ignore')
    import ast
    import numpy as np
    import marimo as mo
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, Dataset
    import pytorch_lightning as L
    from pytorch_lightning.loggers import MLFlowLogger
    import mlflow
    import optuna
    from optuna.integration import MLflowCallback
    import random
    import petname
    from sklearn.model_selection import train_test_split
    return (
        DataLoader,
        Dataset,
        F,
        L,
        MLFlowLogger,
        ast,
        bentoml,
        mlflow,
        mo,
        nn,
        np,
        optuna,
        pd,
        petname,
        random,
        torch,
        train_test_split,
    )


@app.cell
def _(pd):
    # Loading Candidate and Job data in dataframes

    # candidate_df = pd.read_csv('cleaned_datasets/candidate_data_df_clean.csv')
    # job_df = pd.read_csv('cleaned_datasets/job_data_df_clean.csv')

    annotation_df = pd.read_csv('annotation_datasets/two_tower_annotations_B00_1000.csv')

    two_tower_matching_df = pd.read_csv('helping_datasets/job_candidate_annotation_data_two_tower_v2.csv')

    candidate_feature_vector_df = pd.read_csv('processed_dataset/candidate_feature_vectors.csv')
    job_feature_vector_df = pd.read_csv('processed_dataset/job_feature_vectors.csv')

    candidate_df = pd.read_csv('cleaned_datasets/candidate_data_df_clean.csv')
    job_df = pd.read_csv('cleaned_datasets/job_data_df_clean.csv')
    return annotation_df, candidate_feature_vector_df, job_feature_vector_df


@app.cell
def _(mlflow, torch):
    torch.set_float32_matmul_precision('high')
    mlflow.end_run()

    exp_name = 'Two Tower Model Experimentation'
    mlflow.set_tracking_uri("sqlite:///mlflow_database/mlflow.db")

    # Get existing experiment by name
    exp = mlflow.get_experiment_by_name(exp_name)
    if exp is None:
        # Create it if it doesn't exist
        exp_id = mlflow.create_experiment(exp_name)
    else:
        exp_id = exp.experiment_id

    print(f"Experiment ID: {exp_id}")
    mlflow.set_experiment(experiment_id=exp_id)
    return (exp_id,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The data will very likely have NaN values which needs to be replaced with something meaningful
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Saving Data for Annotation
    """)
    return


@app.cell
def _(annotation_df, candidate_feature_vector_df, job_feature_vector_df):
    result_df_ = (annotation_df[['id', 'post_id', 'Score']]
        .merge(candidate_feature_vector_df, on='id', how='left')
        .merge(job_feature_vector_df, on='post_id', how='left')
        .rename(columns={'Score': 'label'})
        [['id', 'candidate_feature_vector', 'post_id', 'job_feature_vector', 'label']]
    )
    return (result_df_,)


@app.cell
def _(ast, candidate_feature_vector_df, job_feature_vector_df, result_df_):
    def fix_col(col):
        return col.apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)

    candidate_feature_vector_df['candidate_feature_vector'] = fix_col(candidate_feature_vector_df['candidate_feature_vector'])
    job_feature_vector_df['job_feature_vector'] = fix_col(job_feature_vector_df['job_feature_vector'])

    result_df_['candidate_feature_vector'] = fix_col(result_df_['candidate_feature_vector'])
    result_df_['job_feature_vector'] = fix_col(result_df_['job_feature_vector'])


    def flatten_vector(v):
        flat = []
        for x in v:
            if isinstance(x, list):   # nested list → extend
                flat.extend(x)
            else:                     # normal float/int
                flat.append(float(x))
        return flat


    candidate_feature_vector_df['candidate_feature_vector'] = candidate_feature_vector_df['candidate_feature_vector'].apply(flatten_vector)
    job_feature_vector_df['job_feature_vector'] = job_feature_vector_df['job_feature_vector'].apply(flatten_vector)

    result_df_['candidate_feature_vector'] = result_df_['candidate_feature_vector'].apply(flatten_vector)
    result_df_['job_feature_vector'] = result_df_['job_feature_vector'].apply(flatten_vector)
    return


@app.cell
def _(result_df_):
    result_df_['label'] = result_df_['label'].replace(-1, 1)
    return


@app.cell
def _(Dataset, torch):

    # ==================== Dataset ====================
    class CandidateJobDataset(Dataset):
        def __init__(self, candidate_features, job_features, labels):
            """
            candidate_features: tensor of candidate feature vectors (n_candidates x feature_dim)
            job_features: tensor of job feature vectors (n_jobs x feature_dim)
            labels: list of (candidate_index, job_index, label) tuples
            """
            self.candidate_features = candidate_features
            self.job_features = job_features
            self.labels = labels

        def __len__(self):
            return len(self.labels)

        def __getitem__(self, idx):
            candidate_index, job_index, label = self.labels[idx]
            return (
                self.candidate_features[candidate_index],
                self.job_features[job_index],
                torch.tensor(label, dtype=torch.float32)
            )

    # candidate_features = torch.tensor(
    #     result_df_['candidate_feature_vector'].tolist(),
    #     dtype=torch.float32
    # )

    # job_features = torch.tensor(
    #     result_df_['job_feature_vector'].tolist(),
    #     dtype=torch.float32
    # )

    # labels = torch.tensor(
    #     result_df_['label'].tolist(),
    #     dtype=torch.float32
    # )


    # job_candidate_dataset = CandidateJobDataset(candidate_features, job_features, labels)
    # job_candidate_dataloader = DataLoader(job_candidate_dataset, batch_size=64, shuffle=True)
    return (CandidateJobDataset,)


@app.cell
def _():
    # # ==================== Dataset ====================
    # class JobCandidateDataset(Dataset):
    #     def __init__(self, candidate_features, job_features, labels):
    #         """
    #         df: pandas DataFrame with candidate_features, job_features, and label columns
    #         candidate_col: name of column containing candidate feature vectors
    #         job_col: name of column containing job feature vectors
    #         label_col: name of column containing labels
    #         """
    #         self.candidate_features = candidate_features
    #         self.job_features = job_features
    #         self.labels = labels

    #     def __len__(self):
    #         return len(self.labels)

    #     def __getitem__(self, idx):
    #         candidate_index, job_index, label = self.labels[idx]
    #         return self.candidate_features[candidate_index], self.job_features[job_index], torch.tensor(label, dtype=torch.float32)


    #         row = self.df.iloc[idx]
    #         candidate_features = torch.tensor(np.array(row[self.candidate_col]), dtype=torch.float32)
    #         job_features = torch.tensor(np.array(row[self.job_col]), dtype=torch.float32)
    #         label = torch.tensor(row[self.label_col], dtype=torch.float32)
    #         return candidate_features, job_features, label
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Neural Network Development
    """)
    return


@app.cell
def _(F, L, nn, torch):
    # ==================== Model (Lightning Module) ====================
    class TwoTowerModel(L.LightningModule):
        def __init__(self, candidate_input_dim, job_input_dim, hidden_dim, lr=1e-3):
            super().__init__()
            self.candidate_tower = nn.Sequential(
                nn.Linear(candidate_input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(p=0.5),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(p=0.5),
                nn.Linear(hidden_dim, hidden_dim),
            )
            self.job_tower = nn.Sequential(
                nn.Linear(job_input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(p=0.5),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(p=0.5),
                nn.Linear(hidden_dim, hidden_dim),
            )
            self.lr = lr
            self.criterion = nn.CosineEmbeddingLoss()
            self.save_hyperparameters()

        def get_candidate_embeddings(self, candidate_features):
            return F.normalize(self.candidate_tower(candidate_features), dim=1)

        def get_job_embeddings(self, job_features):
            return F.normalize(self.job_tower(job_features), dim=1)

        def forward(self, candidate_features, job_features):
            candidate_embedding = self.get_candidate_embeddings(candidate_features)
            job_embedding = self.get_job_embeddings(job_features)
            return candidate_embedding, job_embedding

        def training_step(self, batch, batch_idx):
            candidate_features, job_features, labels = batch
            candidate_emb, job_emb = self(candidate_features, job_features)
            loss = self.criterion(candidate_emb, job_emb, labels)
            self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
            return loss

        def validation_step(self, batch, batch_idx):
            candidate_features, job_features, labels = batch
            candidate_emb, job_emb = self(candidate_features, job_features)
            loss = self.criterion(candidate_emb, job_emb, labels)
            self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
            return loss

        def test_step(self, batch, batch_idx):
            candidate_features, job_features, labels = batch
            candidate_emb, job_emb = self(candidate_features, job_features)
            loss = self.criterion(candidate_emb, job_emb, labels)
            self.log('test_loss', loss)
            return loss

        def configure_optimizers(self):
            return torch.optim.Adam(self.parameters(), lr=self.lr)
    return (TwoTowerModel,)


@app.cell
def _(DataLoader, L, MLFlowLogger, TwoTowerModel, mlflow, petname, random):
    # ==================== Optuna Objective ====================
    def objective(trial, train_dataset, val_dataset, test_dataset, device='gpu'):
        """Optuna objective with nested MLflow run."""
        run_name = f"trial_{trial.number}"
        with mlflow.start_run(nested=True, run_name=run_name) as child_run:
            # Hyperparameters
            learning_rate = trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True)
            batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])
            hidden_dim = trial.suggest_categorical("hidden_dim", [64, 128, 256])
            patience = trial.suggest_int("patience", 5, 10)
            epochs = trial.suggest_int("epochs", 10, 30)

            params = {
                "learning_rate": learning_rate,
                "batch_size": batch_size,
                "hidden_dim": hidden_dim,
                "patience": patience,
                "epochs": epochs,
            }
            mlflow.log_params(params)

            # Data loaders
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

            # Lightning Model
            two_tower_model = TwoTowerModel(
                candidate_input_dim=train_dataset.candidate_features.shape[1],
                job_input_dim=train_dataset.job_features.shape[1],
                hidden_dim=hidden_dim,
                lr=learning_rate
            )

            run_id = mlflow.active_run().info.run_id

            # Trainer
            trainer = L.Trainer(
                max_epochs=epochs,
                accelerator=device,
                devices=1,
                enable_progress_bar=False,
                enable_checkpointing=False,
                logger=MLFlowLogger(
                    run_name=run_name,
                    experiment_name="Two-Tower Candidate-Job Matching",
                    tracking_uri="sqlite:///mlflow_database/mlflow.db",
                    log_model=False,
                    tags={'model': 'two_tower'},
                    run_id=run_id
                ),
                callbacks=[
                    L.callbacks.EarlyStopping('val_loss', patience=patience, mode='min')
                ]
            )

            # Train
            trainer.fit(two_tower_model, train_loader, val_loader)
            val_loss = trainer.callback_metrics['val_loss'].item()
            mlflow.log_metrics({"val_loss": val_loss})

            # Generate model name and log
            model_code = petname.generate(words=2, separator="-") + "-" + str(random.randint(0, 999))
            model_name = f"two_tower-{model_code}"

            mlflow.pytorch.log_model(
                pytorch_model=two_tower_model,
                name=model_name
            )

            mlflow.log_param("error", f"{val_loss:.4f}")
            mlflow.set_tag("Model Name", model_name)
            trial.set_user_attr("run_id", child_run.info.run_id)
            trial.set_user_attr("model_name", model_name)

            return val_loss
    return (objective,)


@app.cell
def _(exp_id, mlflow, objective, optuna):
    # ==================== Main Training Loop ====================
    def main(train_dataset, val_dataset, test_dataset, device='gpu'):
        """Main optimization loop using Optuna and MLflow."""

        # Close any leftover runs
        while mlflow.active_run() is not None:
            print("Closing leftover run:", mlflow.active_run().info.run_id)
            mlflow.end_run()

        # Set up MLflow
        # mlflow.set_tracking_uri("sqlite:///mlflow_database/mlflow.db")
        # exp = mlflow.get_experiment_by_name("Two-Tower Candidate-Job Matching")
        # if exp is None:
        #     exp_id = mlflow.create_experiment("Two-Tower Candidate-Job Matching")
        # else:
        #     exp_id = exp.experiment_id

        parent_run_name = "Two Tower Study"

        # Main optimization loop
        with mlflow.start_run(run_name=parent_run_name, experiment_id=exp_id) as parent_run:
            n_trials = 10
            mlflow.log_param("n_trials", n_trials)

            study = optuna.create_study(direction="minimize")
            study.optimize(
                lambda trial: objective(trial, train_dataset, val_dataset, test_dataset, device=device),
                n_trials=n_trials
            )

            # Log best trial results
            best_trial = study.best_trial
            mlflow.log_params(best_trial.params)
            mlflow.log_metrics({"best_val_loss": study.best_value})

            # Register ONLY the best model
            if best_run_id := best_trial.user_attrs.get("run_id"):
                best_model_name = best_trial.user_attrs.get("model_name")
                mlflow.register_model(
                    model_uri=f"runs:/{best_run_id}/{best_model_name}",
                    name='two_tower_candidate_job'
                )
                mlflow.log_param("best_child_run_id", best_run_id)

            print(f"Best trial: {best_trial.number}")
            print(f"Best value (val_loss): {study.best_value}")
            print(f"Best params: {best_trial.params}")
    return (main,)


@app.cell
def _(CandidateJobDataset, main, np, result_df_, torch, train_test_split):
    candidate_features = torch.tensor(
        result_df_['candidate_feature_vector'].tolist(),
        dtype=torch.float32
    )
    job_features = torch.tensor(
        result_df_['job_feature_vector'].tolist(),
        dtype=torch.float32
    )
    labels_tensor = torch.tensor(
        result_df_['label'].tolist(),
        dtype=torch.float32
    )

    # Create labels as (candidate_idx, job_idx, label) tuples
    labels = [(i, i, labels_tensor[i].item()) for i in range(len(labels_tensor))]

    # Split into train/val/test using sklearn
    label_values = [l[2] for l in labels]
    indices = np.arange(len(labels))
    train_idx, temp_idx = train_test_split(
        indices, 
        test_size=0.3, 
        random_state=42, 
        stratify=label_values
    )
    val_idx, test_idx = train_test_split(
        temp_idx, 
        test_size=0.5, 
        random_state=42, 
        stratify=[label_values[i] for i in temp_idx]
    )

    # Get split labels
    train_labels = [labels[i] for i in train_idx]
    val_labels = [labels[i] for i in val_idx]
    test_labels = [labels[i] for i in test_idx]

    print(f"Train size: {len(train_labels)}, Val size: {len(val_labels)}, Test size: {len(test_labels)}")

    # Create datasets
    train_dataset = CandidateJobDataset(candidate_features, job_features, train_labels)
    val_dataset = CandidateJobDataset(candidate_features, job_features, val_labels)
    test_dataset = CandidateJobDataset(candidate_features, job_features, test_labels)


    # Run optimization
    main(train_dataset, val_dataset, test_dataset, device='gpu')

    return


@app.cell
def _(bentoml, mlflow):
    # Load Two Toweer model

    model_name_two_tower = 'two_tower_candidate_job'
    mlflow_client = mlflow.tracking.MlflowClient()

    version_two_tower = mlflow_client.search_model_versions(f"name='{model_name_two_tower}'")

    latest_two_tower = sorted(version_two_tower, key=lambda v: int(v.version))[-1]

    bentoml.mlflow.import_model(
        name=model_name_two_tower,
        model_uri=latest_two_tower.source,
    )
    return mlflow_client, model_name_two_tower


@app.cell
def _(bentoml, mlflow_client, model_name_two_tower):
    two_tower_load = bentoml.mlflow.load_model("two_tower_candidate_job:latest")
    two_tower_model = two_tower_load._model_impl.get_raw_model() 
    device_two_tower = next(two_tower_model.parameters()).device
    def get_model_info(model_name):
        versions = mlflow_client.search_model_versions(f"name='{model_name}'")
        latest = sorted(versions, key=lambda v: int(v.version))[-1]
        return {
            "name": latest.name,
            "version": latest.version,
            "stage": latest.current_stage,
            "source": latest.source,
            "run_id": latest.run_id,
        }

    two_tower_info = get_model_info(model_name_two_tower)  # Fixed: use model_name_two_tower instead of device_two_tower

    print(two_tower_info)
    return (two_tower_model,)


@app.cell
def _(torch):
    def get_candidate_embedding(two_tower_model, candidate_feature_vector):
        """Get Candidate Embedding"""
        # Convert list to tensor if needed
        if not isinstance(candidate_feature_vector, torch.Tensor):
            candidate_feature_vector = torch.tensor(candidate_feature_vector, dtype=torch.float32)
    
        # Add batch dimension (required by model)
        if candidate_feature_vector.dim() == 1:
            candidate_feature_vector = candidate_feature_vector.unsqueeze(0)
    
        # Move to same device as model
        device = next(two_tower_model.parameters()).device
        candidate_feature_vector = candidate_feature_vector.to(device)
    
        with torch.no_grad():
            embedding = two_tower_model.get_candidate_embeddings(candidate_feature_vector)
            embedding = embedding.detach().cpu()
    
        result = embedding.squeeze(0).tolist()
    
        # Clean up
        del candidate_feature_vector
        del embedding
        torch.cuda.empty_cache()
    
        return result

    def get_job_embedding(two_tower_model, job_feature_vector):
        """Get Job Embedding"""
        # Convert list to tensor if needed
        if not isinstance(job_feature_vector, torch.Tensor):
            job_feature_vector = torch.tensor(job_feature_vector, dtype=torch.float32)
    
        # Add batch dimension (required by model)
        if job_feature_vector.dim() == 1:
            job_feature_vector = job_feature_vector.unsqueeze(0)
    
        # Move to same device as model
        device = next(two_tower_model.parameters()).device
        job_feature_vector = job_feature_vector.to(device)
    
        with torch.no_grad():
            embedding = two_tower_model.get_job_embeddings(job_feature_vector)
            embedding = embedding.detach().cpu()
    
        result = embedding.squeeze(0).tolist()
    
        # Clean up
        del job_feature_vector
        del embedding
        torch.cuda.empty_cache()
    
        return result
    return get_candidate_embedding, get_job_embedding


@app.cell
def _(get_job_embedding, job_feature_vector_df, two_tower_model):
    job_feature_vector_df['job_embedding'] = job_feature_vector_df['job_feature_vector'].apply(lambda x : get_job_embedding(two_tower_model, x))
    return


@app.cell
def _(candidate_feature_vector_df, get_candidate_embedding, two_tower_model):
    candidate_feature_vector_df['candidate_embedding'] = candidate_feature_vector_df['candidate_feature_vector'].apply(lambda x : get_candidate_embedding(two_tower_model, x))
    return


@app.cell
def _(candidate_feature_vector_df, job_feature_vector_df):
    job_feature_vector_df.to_csv('processed_dataset/job_embedding_data.csv')
    candidate_feature_vector_df.to_csv('processed_dataset/candidate_embedding_data.csv')
    return


if __name__ == "__main__":
    app.run()
