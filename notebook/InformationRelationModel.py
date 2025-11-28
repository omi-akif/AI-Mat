import marimo

__generated_with = "0.17.7"
app = marimo.App(width="full")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Information Relation Model (Graph Embedding)

    This notebook implements a graph embedding model to learn representations for jobs and skills based on their relationships.
    It uses a triplet loss function to ensure that related items (e.g., a job and its required skills, or similar jobs) are closer in the embedding space than unrelated items.

    The model is trained using PyTorch Lightning and hyperparameters are optimized using Optuna and MLflow.
    """)
    return


@app.cell
def _():
    import marimo as mo
    import petname
    import mlflow.pytorch
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, Dataset
    import lightning as L
    from lightning.pytorch.loggers import MLFlowLogger
    import mlflow
    import optuna
    import random
    import pandas as pd
    return (
        DataLoader,
        Dataset,
        L,
        MLFlowLogger,
        mlflow,
        mo,
        nn,
        optuna,
        pd,
        petname,
        random,
        torch,
    )


@app.cell
def _(torch):
    torch.backends.cuda.matmul.fp32_precision = 'high'
    torch.backends.cudnn.conv.fp32_precision = 'high'
    return


@app.cell
def _(mlflow, torch):
    exp_id=None
    exp_name = "Representation Learning Model Experimentation"
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
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    return device, exp_id


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. Data Loading and Triplet Generation
    """)
    return


@app.cell
def _(DataLoader, Dataset, L, torch):
    class TripletDataset(Dataset):
        def __init__(self, triplets, x_idx, y_idx, z_idx):
            self.triplets = triplets
            self.x_idx = x_idx
            self.y_idx = y_idx
            self.z_idx = z_idx

        def __len__(self):
            return len(self.triplets)

        def __getitem__(self, idx):
            x, y, z = self.triplets[idx]
            return self.x_idx[x], self.y_idx[y], self.z_idx[z]



    class TripletDataModule(L.LightningDataModule):
        def __init__(self, triplets_dict, batch_size=128, train_split=0.6, val_split=0.2, test_split=0.2):
            super().__init__()
            self.triplets_dict = triplets_dict
            self.batch_size = batch_size
            self.train_split = train_split
            self.val_split = val_split
            self.test_split = test_split

        def setup(self, stage=None):
            for key in self.triplets_dict:
                triplets = self.triplets_dict[key]['triplets']
                x_idx = self.triplets_dict[key]['x_idx']
                y_idx = self.triplets_dict[key]['y_idx']
                z_idx = self.triplets_dict[key]['z_idx']

                n_train = int(len(triplets) * self.train_split)
                n_val = int(len(triplets) * self.val_split)

                perm = torch.randperm(len(triplets))

                train_triplets = [triplets[i] for i in perm[:n_train]]
                val_triplets = [triplets[i] for i in perm[n_train:n_train+n_val]]
                test_triplets = [triplets[i] for i in perm[n_train+n_val:]]

                self.triplets_dict[key]['train'] = TripletDataset(train_triplets, x_idx, y_idx, z_idx)
                self.triplets_dict[key]['val'] = TripletDataset(val_triplets, x_idx, y_idx, z_idx)
                self.triplets_dict[key]['test'] = TripletDataset(test_triplets, x_idx, y_idx, z_idx)

        def train_dataloader(self):
            loaders = [DataLoader(self.triplets_dict[k]['train'], batch_size=self.batch_size, shuffle=True) 
                       for k in ['jj', 'ss', 'js']]
            return zip(*loaders)

        def val_dataloader(self):
            loaders = [DataLoader(self.triplets_dict[k]['val'], batch_size=self.batch_size, shuffle=True) 
                       for k in ['jj', 'ss', 'js']]
            return zip(*loaders)

        def test_dataloader(self):
            loaders = [DataLoader(self.triplets_dict[k]['test'], batch_size=self.batch_size, shuffle=True) 
                       for k in ['jj', 'ss', 'js']]
            return zip(*loaders)
    return (TripletDataModule,)


@app.cell
def _(pd, random):
    def parse_edges_from_csv(job_job_csv, skill_skill_csv, skill_job_csv):
        job_job = list(pd.read_csv(job_job_csv).dropna().itertuples(index=False, name=None)) if job_job_csv else []
        skill_skill = list(pd.read_csv(skill_skill_csv).dropna().itertuples(index=False, name=None)) if skill_skill_csv else []
        skill_job_raw = list(pd.read_csv(skill_job_csv).dropna().itertuples(index=False, name=None)) if skill_job_csv else []

        # Swap columns: (job, skill) -> (skill, job)
        skill_job = [(skill, job) for job, skill in skill_job_raw]

        return job_job, skill_skill, skill_job


    def build_node_sets(job_job, skill_skill, skill_job):
        job_nodes = set()
        skill_nodes = set()

        for src, dst in job_job:
            job_nodes.add(str(src))
            job_nodes.add(str(dst))

        for src, dst in skill_skill:
            skill_nodes.add(str(src))
            skill_nodes.add(str(dst))

        for skill, job in skill_job:
            skill_nodes.add(str(skill))
            job_nodes.add(str(job))

        return sorted(job_nodes), sorted(skill_nodes)



    def generate_triplets(job_job, skill_skill, skill_job, job_nodes, skill_nodes, seed=42, num_samples=10):
        random.seed(seed)

        def build_adj(edges):
            adj = {}
            for src, dst in edges:
                adj.setdefault(src, set()).add(dst)
            return adj

        job_adj = build_adj(job_job)
        skill_adj = build_adj(skill_skill)
        skill_job_dict = {}
        for skill, job in skill_job:
            skill_job_dict.setdefault(job, set()).add(skill)

        triplets = {'jj': [], 'ss': [], 'js': []}

        # Job-job triplets
        for anchor_job in job_nodes:
            positive_jobs = job_adj.get(anchor_job, set())
            if not positive_jobs:
                continue
            negative_jobs = set(job_nodes) - positive_jobs - {anchor_job}
            if not negative_jobs:
                continue
            positive_jobs_list = random.sample(list(positive_jobs), min(len(positive_jobs), num_samples))
            negative_jobs_list = random.sample(list(negative_jobs), min(len(negative_jobs), num_samples))
            for positive_job in positive_jobs_list:
                for negative_job in negative_jobs_list:
                    triplets['jj'].append((anchor_job, positive_job, negative_job))

        # Skill-skill triplets
        for anchor_skill in skill_nodes:
            positive_skills = skill_adj.get(anchor_skill, set())
            if not positive_skills:
                continue
            negative_skills = set(skill_nodes) - positive_skills - {anchor_skill}
            if not negative_skills:
                continue
            positive_skills_list = random.sample(list(positive_skills), min(len(positive_skills), num_samples))
            negative_skills_list = random.sample(list(negative_skills), min(len(negative_skills), num_samples))
            for positive_skill in positive_skills_list:
                for negative_skill in negative_skills_list:
                    triplets['ss'].append((anchor_skill, positive_skill, negative_skill))

        # Skill-job triplets
        for anchor_job in job_nodes:
            positive_skills = skill_job_dict.get(anchor_job, set())
            if not positive_skills:
                continue
            negative_skills = set(skill_nodes) - positive_skills
            if not negative_skills:
                continue
            positive_skills_list = random.sample(list(positive_skills), min(len(positive_skills), num_samples))
            negative_skills_list = random.sample(list(negative_skills), min(len(negative_skills), num_samples))
            for positive_skill in positive_skills_list:
                for negative_skill in negative_skills_list:
                    triplets['js'].append((anchor_job, positive_skill, negative_skill))

        return triplets
    return build_node_sets, generate_triplets, parse_edges_from_csv


@app.cell
def _(build_node_sets, generate_triplets, parse_edges_from_csv):

    job_job, skill_skill, skill_job = parse_edges_from_csv(

        '../datasets/helping_datasets/job_to_job_transition_inf_lear.csv', 
        '../datasets/helping_datasets/skill_to_skill_relation_inf_lear.csv', 
        '../datasets/helping_datasets/job_to_skill_relation_inf_lear.csv'
    )

    job_nodes, skill_nodes = build_node_sets(job_job, skill_skill, skill_job)

    triplets = generate_triplets(job_job, skill_skill, skill_job, job_nodes, skill_nodes)
    return job_nodes, skill_nodes, triplets


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. Model Definition
    """)
    return


@app.cell
def _(L, nn, torch):
    class GraphEmbedding(L.LightningModule):
        def __init__(self, num_jobs, num_skills, dim=50, lr=0.01, lambda_reg=0.001, seed=42,
                     job_nodes=None, skill_nodes=None):
            super().__init__()
            self.save_hyperparameters(ignore=['job_nodes', 'skill_nodes'])

            torch.manual_seed(seed)
            self.W_job = nn.Parameter(torch.normal(0, 0.1, size=(num_jobs, dim)))
            self.W_skill = nn.Parameter(torch.normal(0, 0.1, size=(num_skills, dim)))

            self.lr = lr
            self.lambda_reg = lambda_reg

            # Store mappings for encoding
            self.job2idx = {j: i for i, j in enumerate(job_nodes)} if job_nodes else {}
            self.skill2idx = {s: i for i, s in enumerate(skill_nodes)} if skill_nodes else {}

        def forward(self, x_idx, y_idx, z_idx, embedding_type):
            if embedding_type == 'job':
                W_x = self.W_job
                W_y = self.W_job
                W_z = self.W_job
            elif embedding_type == 'skill':
                W_x = self.W_skill
                W_y = self.W_skill
                W_z = self.W_skill
            else:  # skill_job
                W_x = self.W_job
                W_y = self.W_skill
                W_z = self.W_skill

            return W_x[x_idx], W_y[y_idx], W_z[z_idx]

        def triplet_loss(self, wx, wy, wz):
            pos_sim = torch.sum(wx * wy, dim=1)
            neg_sim = torch.sum(wx * wz, dim=1)
            diff = pos_sim - neg_sim
            return -torch.log(torch.sigmoid(diff) + 1e-8).mean()

        def training_step(self, batch, batch_idx):
            jj_batch, ss_batch, js_batch = batch

            loss_jj = self.triplet_loss(*self.forward(*jj_batch, 'job'))
            loss_ss = self.triplet_loss(*self.forward(*ss_batch, 'skill'))
            loss_js = self.triplet_loss(*self.forward(*js_batch, 'skill_job'))

            reg = self.lambda_reg * (torch.norm(self.W_job) ** 2 + torch.norm(self.W_skill) ** 2)
            loss = loss_jj + loss_ss + loss_js + reg

            self.log('train_loss', loss, prog_bar=True)
            return loss

        def validation_step(self, batch, batch_idx):
            jj_batch, ss_batch, js_batch = batch

            loss_jj = self.triplet_loss(*self.forward(*jj_batch, 'job'))
            loss_ss = self.triplet_loss(*self.forward(*ss_batch, 'skill'))
            loss_js = self.triplet_loss(*self.forward(*js_batch, 'skill_job'))

            reg = self.lambda_reg * (torch.norm(self.W_job) ** 2 + torch.norm(self.W_skill) ** 2)
            loss = loss_jj + loss_ss + loss_js + reg

            self.log('val_loss', loss, prog_bar=True)

        def test_step(self, batch, batch_idx):
            jj_batch, ss_batch, js_batch = batch

            loss_jj = self.triplet_loss(*self.forward(*jj_batch, 'job'))
            loss_ss = self.triplet_loss(*self.forward(*ss_batch, 'skill'))
            loss_js = self.triplet_loss(*self.forward(*js_batch, 'skill_job'))

            reg = self.lambda_reg * (torch.norm(self.W_job) ** 2 + torch.norm(self.W_skill) ** 2)
            loss = loss_jj + loss_ss + loss_js + reg

            self.log('test_loss', loss, prog_bar=True)

        def configure_optimizers(self):
            return torch.optim.SGD(self.parameters(), lr=self.lr)

        def encode(self, node_name, node_type):
            """
            Encode a job or skill by name to get its embedding.
            Args:
                node_name: Name of the job or skill
                node_type: 'job' or 'skill'
            Returns:
                embedding: The embedding vector, or zero embedding if not found
            """
            if node_type == 'job':
                if node_name in self.job2idx:
                    idx = self.job2idx[node_name]
                    return self.W_job[idx].detach().cpu()
                else:
                    # Return zero embedding using torch instead of numpy
                    return torch.zeros(self.W_job.shape[1])
            elif node_type == 'skill':
                if node_name in self.skill2idx:
                    idx = self.skill2idx[node_name]
                    return self.W_skill[idx].detach().cpu()
                else:
                    # Return zero embedding using torch instead of numpy
                    return torch.zeros(self.W_skill.shape[1])
            else:
                raise ValueError("node_type must be 'job' or 'skill'")
    return (GraphEmbedding,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. Training and Hyperparameter Tuning
    """)
    return


@app.cell
def _(
    GraphEmbedding,
    L,
    MLFlowLogger,
    TripletDataModule,
    device,
    exp_id,
    job_nodes,
    mlflow,
    optuna,
    petname,
    random,
    skill_nodes,
    triplets,
):
    def objective(trial, triplets, job_nodes, skill_nodes, device='cuda'):
        """Optuna objective with nested MLflow run."""
        run_name = f"trial_{trial.number}"
        with mlflow.start_run(nested=True, run_name=run_name) as child_run:
            learning_rate = trial.suggest_float("lr", 1e-4, 1e-1, log=True)
            batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])
            dim = trial.suggest_categorical("dim", [32, 50, 64])
            lambda_reg = trial.suggest_float("lambda_reg", 1e-4, 1e-2, log=True)
            patience = trial.suggest_int("patience", 3, 10)
            epochs = trial.suggest_int("epochs", 20, 100)

            params = {
                "lr": learning_rate,
                "batch_size": batch_size,
                "dim": dim,
                "lambda_reg": lambda_reg,
                "patience": patience,
                "epochs": epochs,
            }
            mlflow.log_params(params)

            job2idx = {j: i for i, j in enumerate(job_nodes)}
            skill2idx = {s: i for i, s in enumerate(skill_nodes)}

            triplets_data = {
                'jj': {'triplets': triplets['jj'], 'x_idx': job2idx, 'y_idx': job2idx, 'z_idx': job2idx},
                'ss': {'triplets': triplets['ss'], 'x_idx': skill2idx, 'y_idx': skill2idx, 'z_idx': skill2idx},
                'js': {'triplets': triplets['js'], 'x_idx': job2idx, 'y_idx': skill2idx, 'z_idx': skill2idx}
            }

            datamodule = TripletDataModule(triplets_data, batch_size=batch_size)
            model = GraphEmbedding(
                len(job_nodes), 
                len(skill_nodes), 
                dim=dim, 
                lr=learning_rate, 
                lambda_reg=lambda_reg,
                job_nodes=job_nodes,
                skill_nodes=skill_nodes
            )

            run_id = mlflow.active_run().info.run_id
            trainer = L.Trainer(
                max_epochs=epochs,
                accelerator='gpu',
                devices=1,
                enable_progress_bar=False,
                enable_checkpointing=False,
                logger=MLFlowLogger(
                    run_name=run_name,
                    experiment_name="Representation Learning Model Experimentation",
                    tracking_uri="sqlite:///mlflow_database/mlflow.db",
                    log_model=False,
                    tags={'model': 'representation-learning'},
                    run_id=run_id
                ),
                callbacks=[
                    L.pytorch.callbacks.EarlyStopping('val_loss', patience=patience)
                ]
            )

            trainer.fit(model, datamodule)
            val_loss = trainer.callback_metrics['val_loss'].item()
            mlflow.log_metrics({"val_loss": val_loss})

            # Generate model name
            model_code = petname.generate(words=2, separator="-") + "-" + str(random.randint(0, 999))
            model_name = f"representation-learning-{model_code}"

            # Log model
            mlflow.pytorch.log_model(
                pytorch_model=model,
                name=model_name
            )

            mlflow.log_param("error", f"{val_loss:.4f}")
            mlflow.set_tag("Model Name", model_name)

            trial.set_user_attr("run_id", child_run.info.run_id)
            trial.set_user_attr("model_name", model_name)

            return val_loss


    paren_run_name = "rep_lear_study"

    with mlflow.start_run(run_name=paren_run_name, experiment_id=exp_id) as parent_run:
        n_trials = 10
        mlflow.log_param("n_trials", n_trials)

        mlflow.set_tag("Run Name", paren_run_name)

        study = optuna.create_study(direction="minimize")
        study.optimize(
            lambda trial: objective(trial, triplets, job_nodes, skill_nodes, device),
            n_trials=n_trials
        )

        # Log best trial results
        best_trial = study.best_trial
        mlflow.log_params(best_trial.params)
        mlflow.log_metrics({"val_loss": study.best_value})

        # Register best model
        if best_run_id := best_trial.user_attrs.get("run_id"):
            best_model_name = best_trial.user_attrs.get("model_name")
            mlflow.register_model(
                model_uri=f"runs:/{best_run_id}/{best_model_name}",
                name="RL-graph-emb"
            )
            mlflow.log_param("best_child_run_id", best_run_id)
    return


if __name__ == "__main__":
    app.run()
