import marimo

__generated_with = "0.17.7"
app = marimo.App(width="full")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Deep Learning Embedding Model (DLEM)

    This notebook implements the Deep Learning Embedding Model (DLEM) for job-candidate matching.
    The model uses a combination of Convolutional Neural Networks (CNNs) and Attention mechanisms to learn embeddings for both job descriptions and candidate resumes.

    The architecture consists of:
    1.  **Word2Vec Embedding**: Converts text to dense vectors.
    2.  **Stacked Convolutional Blocks**: Extracts local features with varying kernel sizes.
    3.  **Attention Layer**: Focuses on important parts of the text.
    4.  **Fully Connected Layer**: Produces the final embedding.
    5.  **Classifier**: Predicts the relevance score based on the concatenated embeddings of job and candidate.
    """)
    return


@app.cell
def _():
    import marimo as mo
    import petname
    import random
    import gc
    import torch
    import mlflow
    import mlflow.pytorch
    from lightning.pytorch.loggers import MLFlowLogger
    import optuna
    import numpy as np
    import pandas as pd
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import Dataset, DataLoader
    from sklearn.model_selection import train_test_split
    import matplotlib.pyplot as plt
    from gensim.models import KeyedVectors
    from scipy.spatial.distance import cdist
    from sklearn.utils.class_weight import compute_class_weight
    from optuna.integration.mlflow import MLflowCallback
    import lightning as L
    return (
        DataLoader,
        Dataset,
        F,
        KeyedVectors,
        L,
        MLFlowLogger,
        compute_class_weight,
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
def _(torch):
    torch.backends.cuda.matmul.fp32_precision = 'high'
    torch.backends.cudnn.conv.fp32_precision = 'high'
    return


@app.cell
def _(mlflow):
    exp_id=None
    exp_name = 'DLEM Model Experimentation'
    mlflow.end_run()
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


@app.cell
def _(mlflow):
    mlflow.config.enable_system_metrics_logging()
    mlflow.config.set_system_metrics_sampling_interval(1)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. Model Architecture Definition
    """)
    return


@app.cell
def _(nn):
    # -------------------------------
    # Single Convolutional Unit
    # Paper: "pipeline of 1D-convolution, batch normalization and max-pooling"
    # -------------------------------

    class ConvUnit(nn.Module):
        """Conv1d → BatchNorm → MaxPool (as per paper)."""

        def __init__(self, in_channels, out_channels, kernel_size, pool_size=2):
            super().__init__()
            self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, padding=kernel_size // 2)
            self.bn = nn.BatchNorm1d(out_channels)
            self.pool = nn.MaxPool1d(pool_size)

        def forward(self, x):
            x = self.conv(x)
            x = self.bn(x)
            x = self.pool(x)
            return x
    return (ConvUnit,)


@app.cell
def _(ConvUnit, nn, torch):

    # -------------------------------
    # Convolutional Block with Saliency
    # Paper: "Each stacked block contains three consecutive convolutional blocks"
    # and: "attention layer is built from outputs and their saliency"
    # Saliency = learned importance weighting
    # -------------------------------

    class ConvBlock(nn.Module):
        """Three consecutive ConvUnits + saliency weighting."""

        def __init__(self, in_channels, out_channels, kernel_size, pool_size=2):
            super().__init__()
            self.unit1 = ConvUnit(in_channels, out_channels, kernel_size, pool_size)
            self.unit2 = ConvUnit(out_channels, out_channels, kernel_size, pool_size)
            self.unit3 = ConvUnit(out_channels, out_channels, kernel_size, pool_size)
            # Saliency: learns which parts are important
            self.saliency_conv = nn.Conv1d(out_channels, 1, kernel_size=1)

        def forward(self, x):
            x = self.unit1(x)
            x = self.unit2(x)
            x = self.unit3(x)

            # Learn saliency (importance) scores
            saliency = torch.sigmoid(self.saliency_conv(x))  # [batch, 1, time]

            # Weight the output by saliency
            weighted_x = x * saliency

            return weighted_x, saliency
    return (ConvBlock,)


@app.cell
def _(ConvBlock, nn):
    # -------------------------------
    # Six Stacked Blocks with Different Kernel Sizes
    # Paper: "six stacked blocks with different kernel sizes, ranged from 1 to 10"
    # Purpose: "construct distributed representations of the sentence"
    # -------------------------------

    class StackedConvBlocks(nn.Module):
        """Six convolutional blocks with different receptive fields."""

        def __init__(self, in_channels, out_channels, pool_size=2):
            super().__init__()
            kernel_sizes = [1, 2, 3, 5, 7, 10]
            self.blocks = nn.ModuleList([
                ConvBlock(in_channels, out_channels, k, pool_size)
                for k in kernel_sizes
            ])

        def forward(self, x):
            outputs, saliencies = [], []
            for block in self.blocks:
                out, sal = block(x)
                outputs.append(out)
                saliencies.append(sal)
            return outputs, saliencies
    return (StackedConvBlocks,)


@app.cell
def _(F, nn, torch):
    # -------------------------------
    # Attention Layer
    # Diagram shows: Attention Activation → Attention Weights → Context Vector
    # Paper: "built from outputs from stacked blocks and their saliency"
    # 
    # The diagram reveals the process:
    # 1. Concatenate all 6 outputs
    # 2. Apply Attention Activation (sigmoid on saliency)
    # 3. Compute Attention Weights (from saliency)
    # 4. Create Context Vector (weighted combination)
    # -------------------------------

    class AttentionLayer(nn.Module):
        """
        Attention layer matching the diagram exactly.

        Diagram steps:
        - Attention Activation: Apply sigmoid to saliency maps
        - Attention Weights: Use activated saliency as weights
        - Context Vector: Apply weights to create context
        """

        def __init__(self, embed_dim):
            super().__init__()
            self.query = nn.Linear(embed_dim, embed_dim)
            self.key = nn.Linear(embed_dim, embed_dim)
            self.value = nn.Linear(embed_dim, embed_dim)
            self.scale = embed_dim ** (-0.5)

        def forward(self, conv_outputs, saliencies):
            """
            Args:
                conv_outputs: list of 6 tensors [batch, channels, time]
                             (already saliency-weighted from ConvBlocks)
                saliencies: list of 6 tensors [batch, 1, time]
                           (raw saliency maps from each block)

            Returns:
                context: [batch, time, embed_dim] - context vector
                attn_weights: [batch, time, time] - attention weight matrix

            Diagram flow: Conv outputs → Attention Activation → Attention Weights → Context Vector
            """

            # Align all time dimensions to the minimum
            min_time = min(o.shape[2] for o in conv_outputs)

            # Prepare outputs and align time dimensions
            weighted = []
            aligned_saliencies = []
            for o, s in zip(conv_outputs, saliencies):
                # Adaptive pool to match dimensions
                w = F.adaptive_avg_pool1d(o, min_time)
                sal = F.adaptive_avg_pool1d(s, min_time)
                weighted.append(w)
                aligned_saliencies.append(sal)

            # Concatenate all 6 outputs
            # [batch, total_channels, min_time]
            x = torch.cat(weighted, dim=1)

            # Concatenate all 6 saliency maps
            # [batch, 6, min_time]
            saliency_concat = torch.cat(aligned_saliencies, dim=1)

            # STEP 1: Attention Activation (diagram shows this explicitly)
            # Apply sigmoid to saliency maps as activation
            attention_activation = torch.sigmoid(saliency_concat)  # [batch, 6, time]

            # Permute for attention: [batch, time, total_channels]
            x = x.permute(0, 2, 1)  # [batch, time, channels]

            # Permute attention activation: [batch, 6, time] → [batch, time, 6]
            attention_activation = attention_activation.permute(0, 2, 1)  # [batch, time, 6]

            # STEP 2: Compute Transformer-style attention WITH attention activation
            Q = self.query(x)
            K = self.key(x)
            V = self.value(x)

            # Compute attention scores
            scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale  # [batch, time, time]

            # STEP 2b: Modulate attention scores by attention_activation
            # Apply activation as a weight factor to the scores
            # Expand activation for broadcasting: [batch, time, 6] → [batch, time, 1]
            activation_weight = attention_activation.mean(dim=-1, keepdim=True)  # [batch, time, 1]
            scores = scores * activation_weight

            # Normalize to get attention weights
            attention_weights = F.softmax(scores, dim=-1)  # [batch, time, time]

            # STEP 3: Context Vector - weighted sum of values
            context = torch.matmul(attention_weights, V)  # [batch, time, embed_dim]

            return context, attention_weights
    return (AttentionLayer,)


@app.cell
def _(nn):
    # -------------------------------
    # Fully Connected Embedding
    # -------------------------------
    class FullyConnectedEmbedding(nn.Module):
        def __init__(self, input_dim, output_dim=128, dropout=0.5):
            super().__init__()
            self.dropout = nn.Dropout(dropout)  # ✅ First
            self.fc = nn.Linear(input_dim, output_dim)  # ✅ Second
            self.relu = nn.ReLU()  # ✅ Third

        def forward(self, context):
            # context: [batch, time, input_dim]
            x = self.dropout(context)  # Apply dropout first
            x = self.fc(x)  # Then linear
            embedding = x.mean(dim=1)  # [batch, output_dim]
            embedding = self.relu(embedding)  # ReLU last
            return embedding
    return (FullyConnectedEmbedding,)


@app.cell
def _(AttentionLayer, FullyConnectedEmbedding, StackedConvBlocks, nn):
    # -------------------------------
    # Deep Embedding Model (Encoder)
    # Paper: "DLEM consists of input layer, CNN layer and attention layer"
    # Flow: Input → WordVec → StackedConvBlocks → AttentionLayer → FC+ReLU → Embedding
    # -------------------------------

    class ModelEncoder(nn.Module):
        """
        Encoder: produces embedding g(x) as per paper.
        g(x) is the final embedding used for matching.
        """

        def __init__(self, in_channels, conv_out_channels, attention_embed_dim, embedding_dim):
            super().__init__()
            self.conv_blocks = StackedConvBlocks(in_channels, conv_out_channels)
            self.attention = AttentionLayer(attention_embed_dim)
            self.fc_embedding = FullyConnectedEmbedding(attention_embed_dim, output_dim=embedding_dim)

        def forward(self, x):
            # CNN: 6 stacked blocks with saliency
            conv_outs, saliencies = self.conv_blocks(x)

            # Attention layer
            context, attn_weights = self.attention(conv_outs, saliencies)

            # FC + ReLU to produce final embedding
            embedding = self.fc_embedding(context)

            return embedding, attn_weights
    return (ModelEncoder,)


@app.cell
def _(ModelEncoder, nn, torch):
    # -------------------------------
    # DLEM: Full Model
    # Paper: "p = σ(w^T [g(c); g(j)] + b)"
    # where g(c) and g(j) are embeddings from DLEM
    # σ is sigmoid, applied via BCEWithLogitsLoss
    # -------------------------------

    class DLEM(nn.Module):
        """
        Full Deep Learning Embedding Model for Candidate-Job Matching.

        Paper: Takes candidate and job inputs, produces matching probability.
        p = σ(w^T [g(c); g(j)] + b)
        """

        def __init__(self, in_channels, conv_out_channels, attention_dim, embedding_dim):
            super().__init__()
            self.encoder = ModelEncoder(in_channels, conv_out_channels, attention_dim, embedding_dim)
            # Classifier: takes concatenated embeddings [g(c); g(j)]
            self.classifier = nn.Linear(2 * embedding_dim, 1)

        def forward(self, cand_input, job_input):
            """
            Args:
                cand_input: [batch, in_channels, seq_len] candidate input
                job_input: [batch, in_channels, seq_len] job input

            Returns:
                logits: [batch, 1] raw logits (sigmoid applied in loss)
                attention: (cand_attn, job_attn) for interpretability
            """
            cand_embed, cand_attn = self.encoder(cand_input)
            job_embed, job_attn = self.encoder(job_input)

            # Concatenate embeddings
            combined = torch.cat([cand_embed, job_embed], dim=1)

            # Classifier produces logit
            logits = self.classifier(combined)

            return logits, (cand_attn, job_attn)

        def get_embedding(self, input):
            """Get embeddings for both job and candidate"""
            job_embed, _ = self.encoder(input)
            return job_embed

        def get_job_embedding(self, job_input):
            """Get embedding for a job."""
            job_embed, _ = self.encoder(job_input)
            return job_embed

        def get_candidate_embedding(self, cand_input):
            """Get embedding for a candidate."""
            cand_embed, _ = self.encoder(cand_input)
            return cand_embed
    return (DLEM,)


@app.cell
def _(L, nn, torch):
    # -------------------------------
    # Lightning Wrapper
    # Paper: "we choose relevance-based binary cross-entropy as the loss function"
    # BCEWithLogitsLoss with pos_weight for class imbalance
    # -------------------------------

    class LitDLEM(L.LightningModule):
        """DLEM wrapped in Lightning for training."""

        def __init__(self, dlem_model, device, lr=0.001, pos_weight=1.0):
            super().__init__()
            self.model = dlem_model
            self.lr = lr
            self.pos_weight = pos_weight
            # Binary cross-entropy loss as per paper
            self.criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight], device=device))

        def forward(self, cand, job):
            """Inference."""
            logits, attn_weights = self.model(cand, job)
            return logits, attn_weights

        def training_step(self, batch, batch_idx):
            """Training step."""
            cand, job, labels = batch
            logits, _ = self(cand, job)
            logits = logits.squeeze(1)
            loss = self.criterion(logits, labels.float())
            self.log("train_loss", loss)
            return loss

        def validation_step(self, batch, batch_idx):
            """Validation step."""
            cand, job, labels = batch
            logits, _ = self(cand, job)
            logits = logits.squeeze(1)
            loss = self.criterion(logits, labels.float())
            self.log("val_loss", loss)

        def test_step(self, batch, batch_idx):
            """Test step."""
            cand, job, labels = batch
            logits, _ = self(cand, job)
            logits = logits.squeeze(1)
            loss = self.criterion(logits, labels.float())
            self.log("test_loss", loss)

        def configure_optimizers(self):
            """Optimizer and scheduler."""
            optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', factor=0.5, patience=3
            )
            return {
                'optimizer': optimizer,
                'lr_scheduler': {'scheduler': scheduler, 'monitor': 'val_loss'}
            }

        def get_embedding(self, input):
            """Delegate to model"""
            return self.model.get_embedding(input)

        def get_job_embedding(self, job_input):
            """Delegate to model."""
            return self.model.get_job_embedding(job_input)

        def get_candidate_embedding(self, cand_input):
            """Delegate to model."""
            return self.model.get_candidate_embedding(cand_input)
    return (LitDLEM,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. Data Preparation
    """)
    return


@app.cell
def _(Dataset, np, torch, train_test_split):
    class CandidateJobDataset(Dataset):
        def __init__(self, data, word2vec, embed_dim=128, max_len=500):
            self.job_text = data['job_string'].values
            self.resume_text = data['candidate_latest_resume_text'].values
            self.labels = data['label'].values
            self.word2vec = word2vec
            self.embed_dim = embed_dim
            self.max_len = max_len

        def __len__(self):
            return len(self.job_text)

        def __getitem__(self, idx):
            job_tensor = self.sentence_to_tensor(self.job_text[idx])
            resume_tensor = self.sentence_to_tensor(self.resume_text[idx])
            label = torch.tensor(self.labels[idx]).float()

            return job_tensor, resume_tensor, label

        def sentence_to_tensor(self, text):
            """Convert text to word2vec embedding tensor."""
            tokens = str(text).lower().split()

            # Get embeddings for each token
            embeddings = []
            for token in tokens[:self.max_len]:
                if token in self.word2vec:
                    embeddings.append(self.word2vec[token])
                else:
                    embeddings.append(np.zeros(self.embed_dim))

            # Pad if necessary
            if len(embeddings) < self.max_len:
                padding = np.zeros((self.max_len - len(embeddings), self.embed_dim))
                embeddings = np.vstack([embeddings, padding])
            else:
                embeddings = np.array(embeddings[:self.max_len])

            # Convert to tensor [max_len, embed_dim]
            tensor = torch.tensor(embeddings, dtype=torch.float32)
            # Transpose to [embed_dim, max_len] for CNN
            tensor = tensor.transpose(0, 1)

            return tensor



    def segment_data(df, word2vec, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, 
                     embed_dim=128, max_len=500, random_state=42):  # ✅ Remove batch_size
        """
        Segment dataframe into train/val/test and create datasets (not loaders).
        """
        assert train_ratio + val_ratio + test_ratio == 1.0, "Ratios must sum to 1.0"
        print(f"Total samples: {len(df)}")
        print(f"Train ratio: {train_ratio*100}% | Val ratio: {val_ratio*100}% | Test ratio: {test_ratio*100}%")

        # Split data
        train_val_df, test_df = train_test_split(
            df, test_size=test_ratio, random_state=random_state, stratify=df['label']
        )

        val_ratio_adjusted = val_ratio / (train_ratio + val_ratio)
        train_df, val_df = train_test_split(
            train_val_df, test_size=val_ratio_adjusted, random_state=random_state, stratify=train_val_df['label']
        )

        print(f"\nTrain samples: {len(train_df)}")
        print(f"Val samples: {len(val_df)}")
        print(f"Test samples: {len(test_df)}")

        # ✅ Create datasets (not loaders)
        train_dataset = CandidateJobDataset(train_df, word2vec, embed_dim, max_len)
        val_dataset = CandidateJobDataset(val_df, word2vec, embed_dim, max_len)
        test_dataset = CandidateJobDataset(test_df, word2vec, embed_dim, max_len)

        # ✅ Return datasets, not loaders
        return train_dataset, val_dataset, test_dataset, train_df, val_df, test_df
    return (segment_data,)


@app.cell
def _(KeyedVectors, compute_class_weight, np, pd, torch):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    job_candidate_matching_annotation_df = pd.read_csv('../datasets/helping_datasets/job_candidate_matching_annotations_dlem.csv')
    job_candidate_matching_annotation_df = job_candidate_matching_annotation_df[job_candidate_matching_annotation_df['Score'] != -1]

    job_candidate_matching_annotation_df.rename(columns={'Score': 'label'}, inplace=True)



    word2vec_model = KeyedVectors.load('models/job_candidate_word2vec.kv')

    class_weights = compute_class_weight('balanced', 
                                         classes=np.unique(job_candidate_matching_annotation_df['label']), 
                                         y=job_candidate_matching_annotation_df['label'])
    pos_weight = torch.tensor([class_weights[1] / class_weights[0]], dtype=torch.float32, device=device)
    return device, job_candidate_matching_annotation_df, word2vec_model


@app.cell
def _(job_candidate_matching_annotation_df, segment_data, word2vec_model):

    # ✅ Call it without batch_size
    train_dataset, val_dataset, test_dataset, _, _, _ = segment_data(
        job_candidate_matching_annotation_df,
        word2vec_model,
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
        embed_dim=128,
        max_len=5000,
        random_state=42
    )
    return test_dataset, train_dataset, val_dataset


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. Training and Hyperparameter Tuning
    """)
    return


@app.cell
def _(
    DLEM,
    DataLoader,
    L,
    LitDLEM,
    MLFlowLogger,
    device,
    exp_id,
    mlflow,
    optuna,
    petname,
    random,
    test_dataset,
    train_dataset,
    val_dataset,
):
    def objective(trial, train_dataset, val_dataset, test_dataset):
        """Optuna objective with nested MLflow run."""
        run_name = f"trial_{trial.number}"
        with mlflow.start_run(nested=True, run_name=run_name) as child_run:
            learning_rate = trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True)
            batch_size = trial.suggest_categorical("batch_size", [16, 32])
            conv_out_channels = trial.suggest_categorical("conv_out_channels", [16, 32])
            # embedding_dim = trial.suggest_categorical("embedding_dim", [64, 128])
            pos_weight_multiplier = trial.suggest_float("pos_weight_multiplier", 1.0, 8.0)
            patience = trial.suggest_int("patience", 5, 10)
            attention_dim = 6 * conv_out_channels
            epochs = trial.suggest_int("epochs", 5, 10)

            params = {
                "learning_rate": learning_rate,
                "batch_size": batch_size,
                "conv_out_channels": conv_out_channels,
                "embedding_dim": 64,
                "attn_dim": attention_dim,
                "pos_weight_multiplier": pos_weight_multiplier,
                "patience": patience,
                "epochs": epochs,
            }
            mlflow.log_params(params)

            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=15)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=15)

            model = DLEM(
                in_channels=128,
                conv_out_channels=conv_out_channels,
                attention_dim=attention_dim,
                embedding_dim=64
            )
            pl_model = LitDLEM(model, device=device, lr=learning_rate, pos_weight=pos_weight_multiplier)

            run_id = mlflow.active_run().info.run_id
            trainer = L.Trainer(
                max_epochs=epochs,
                accelerator='gpu',
                devices=1,
                enable_progress_bar=False,
                enable_checkpointing=False,  # ✅ Disable all checkpointing
                logger=MLFlowLogger(
                    run_name=run_name,
                    experiment_name="DLEM Model Experimentation",
                    tracking_uri="sqlite:///mlflow_database/mlflow.db",
                    log_model=False,
                    tags={'model': 'dlem'},
                    run_id=run_id
                ),
                callbacks=[
                    L.pytorch.callbacks.EarlyStopping('val_loss', patience=patience)
                ]
            )

            trainer.fit(pl_model, train_loader, val_loader)
            val_loss = trainer.callback_metrics['val_loss'].item()
            mlflow.log_metrics({"val_loss": val_loss})

            # ✅ Log as parameter so it appears as "error" column
            model_code = petname.generate(words=2, separator="-") + "-" + str(random.randint(0, 999))
            model_name = f"dlem-{model_code}"

            # ✅ Log model but don't register it
            mlflow.pytorch.log_model(
                pytorch_model=pl_model,
                artifact_path=model_name  # ✅ Use artifact_path
            )

            mlflow.log_param("error", f"{val_loss:.4f}")
            mlflow.set_tag("Model Name", model_name)
            trial.set_user_attr("run_id", child_run.info.run_id)
            trial.set_user_attr("model_name", model_name)  # ✅ Store the name
            return val_loss


    while mlflow.active_run() is not None:
        print("Closing leftover run:", mlflow.active_run().info.run_id)
        mlflow.end_run()

    paren_run_name = "dlem_study"

    # Main optimization loop
    with mlflow.start_run(run_name=paren_run_name, experiment_id=exp_id) as parent_run:
        n_trials = 10
        mlflow.log_param("n_trials", n_trials)

        study = optuna.create_study(direction="minimize")
        study.optimize(
            lambda trial: objective(trial, train_dataset, val_dataset, test_dataset),
            n_trials=n_trials
        )

        # ✅ Log best trial results
        best_trial = study.best_trial
        mlflow.log_params(best_trial.params)
        mlflow.log_metrics({"best_val_loss": study.best_value})

        # ✅ ONLY register the BEST model
        if best_run_id := best_trial.user_attrs.get("run_id"):
            best_model_name = best_trial.user_attrs.get("model_name")
            mlflow.register_model(
                model_uri=f"runs:/{best_run_id}/{best_model_name}",
                    name='dlem'
            )
            mlflow.log_param("best_child_run_id", best_run_id)
    return


if __name__ == "__main__":
    app.run()
