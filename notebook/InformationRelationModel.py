import marimo

__generated_with = "0.17.7"
app = marimo.App(width="full")


@app.cell
def _():
    # import torch
    # import numpy as np
    # import random
    # import torch.nn.functional as F
    # # import matplotlib.pyplot as plt
    # # import plotly.graph_objects as go
    # from sklearn.manifold import TSNE
    # # from typing import Tuple
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

        'helping_datasets/job_to_job_transition_inf_lear.csv', 
        'helping_datasets/skill_to_skill_relation_inf_lear.csv', 
        'helping_datasets/job_to_skill_relation_inf_lear.csv'
    )

    job_nodes, skill_nodes = build_node_sets(job_job, skill_skill, skill_job)

    triplets = generate_triplets(job_job, skill_skill, skill_job, job_nodes, skill_nodes)
    return job_nodes, skill_nodes, triplets


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


@app.cell
def _():
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


@app.cell
def _():
    # def parse_edges_from_csv(job_job_csv, skill_skill_csv, skill_job_csv):
    #     job_job = list(pd.read_csv(job_job_csv).dropna().itertuples(index=False, name=None)) if job_job_csv else []
    #     skill_skill = list(pd.read_csv(skill_skill_csv).dropna().itertuples(index=False, name=None)) if skill_skill_csv else []
    #     skill_job_raw = list(pd.read_csv(skill_job_csv).dropna().itertuples(index=False, name=None)) if skill_job_csv else []

    #     # Swap columns: (job, skill) -> (skill, job)
    #     skill_job = [(skill, job) for job, skill in skill_job_raw]

    #     return job_job, skill_skill, skill_job


    # def build_node_sets(job_job, skill_skill, skill_job):
    #     job_nodes = set()
    #     skill_nodes = set()

    #     for src, dst in job_job:
    #         job_nodes.add(str(src))
    #         job_nodes.add(str(dst))

    #     for src, dst in skill_skill:
    #         skill_nodes.add(str(src))
    #         skill_nodes.add(str(dst))

    #     for skill, job in skill_job:
    #         skill_nodes.add(str(skill))
    #         job_nodes.add(str(job))

    #     return sorted(job_nodes), sorted(skill_nodes)



    # def generate_triplets(job_job, skill_skill, skill_job, job_nodes, skill_nodes, seed=42, num_samples=10):
    #     random.seed(seed)

    #     def build_adj(edges):
    #         adj = {}
    #         for src, dst in edges:
    #             adj.setdefault(src, set()).add(dst)
    #         return adj

    #     job_adj = build_adj(job_job)
    #     skill_adj = build_adj(skill_skill)
    #     skill_job_dict = {}
    #     for skill, job in skill_job:
    #         skill_job_dict.setdefault(job, set()).add(skill)

    #     triplets = {'jj': [], 'ss': [], 'js': []}

    #     # Job-job triplets
    #     for anchor_job in job_nodes:
    #         positive_jobs = job_adj.get(anchor_job, set())
    #         if not positive_jobs:
    #             continue
    #         negative_jobs = set(job_nodes) - positive_jobs - {anchor_job}
    #         if not negative_jobs:
    #             continue
    #         positive_jobs_list = random.sample(list(positive_jobs), min(len(positive_jobs), num_samples))
    #         negative_jobs_list = random.sample(list(negative_jobs), min(len(negative_jobs), num_samples))
    #         for positive_job in positive_jobs_list:
    #             for negative_job in negative_jobs_list:
    #                 triplets['jj'].append((anchor_job, positive_job, negative_job))

    #     # Skill-skill triplets
    #     for anchor_skill in skill_nodes:
    #         positive_skills = skill_adj.get(anchor_skill, set())
    #         if not positive_skills:
    #             continue
    #         negative_skills = set(skill_nodes) - positive_skills - {anchor_skill}
    #         if not negative_skills:
    #             continue
    #         positive_skills_list = random.sample(list(positive_skills), min(len(positive_skills), num_samples))
    #         negative_skills_list = random.sample(list(negative_skills), min(len(negative_skills), num_samples))
    #         for positive_skill in positive_skills_list:
    #             for negative_skill in negative_skills_list:
    #                 triplets['ss'].append((anchor_skill, positive_skill, negative_skill))

    #     # Skill-job triplets
    #     for anchor_job in job_nodes:
    #         positive_skills = skill_job_dict.get(anchor_job, set())
    #         if not positive_skills:
    #             continue
    #         negative_skills = set(skill_nodes) - positive_skills
    #         if not negative_skills:
    #             continue
    #         positive_skills_list = random.sample(list(positive_skills), min(len(positive_skills), num_samples))
    #         negative_skills_list = random.sample(list(negative_skills), min(len(negative_skills), num_samples))
    #         for positive_skill in positive_skills_list:
    #             for negative_skill in negative_skills_list:
    #                 triplets['js'].append((anchor_job, positive_skill, negative_skill))

    #     return triplets
    return


@app.cell
def _():
    # job_job, skill_skill, skill_job = parse_edges_from_csv(

    #     'helping_datasets/job_to_job_transition_inf_lear.csv', 
    #     'helping_datasets/skill_to_skill_relation_inf_lear.csv', 
    #     'helping_datasets/job_to_skill_relation_inf_lear.csv'
    # )

    # job_nodes, skill_nodes = build_node_sets(job_job, skill_skill, skill_job)

    # triplets = generate_triplets(job_job, skill_skill, skill_job, job_nodes, skill_nodes)
    return


@app.cell
def _():
    # class TripletDataModule(L.LightningDataModule):
    #     def __init__(self, triplets_dict, batch_size=128, train_split=0.6, val_split=0.2, test_split=0.2):
    #         super().__init__()
    #         self.triplets_dict = triplets_dict
    #         self.batch_size = batch_size
    #         self.train_split = train_split
    #         self.val_split = val_split
    #         self.test_split = test_split

    #     def setup(self, stage=None):
    #         for key in self.triplets_dict:
    #             triplets = self.triplets_dict[key]['triplets']
    #             x_idx = self.triplets_dict[key]['x_idx']
    #             y_idx = self.triplets_dict[key]['y_idx']
    #             z_idx = self.triplets_dict[key]['z_idx']

    #             n_train = int(len(triplets) * self.train_split)
    #             n_val = int(len(triplets) * self.val_split)

    #             perm = torch.randperm(len(triplets))

    #             train_triplets = [triplets[i] for i in perm[:n_train]]
    #             val_triplets = [triplets[i] for i in perm[n_train:n_train+n_val]]
    #             test_triplets = [triplets[i] for i in perm[n_train+n_val:]]

    #             self.triplets_dict[key]['train'] = TripletDataset(train_triplets, x_idx, y_idx, z_idx)
    #             self.triplets_dict[key]['val'] = TripletDataset(val_triplets, x_idx, y_idx, z_idx)
    #             self.triplets_dict[key]['test'] = TripletDataset(test_triplets, x_idx, y_idx, z_idx)

    #     def train_dataloader(self):
    #         loaders = [DataLoader(self.triplets_dict[k]['train'], batch_size=self.batch_size, shuffle=True) 
    #                    for k in ['jj', 'ss', 'js']]
    #         return zip(*loaders)

    #     def val_dataloader(self):
    #         loaders = [DataLoader(self.triplets_dict[k]['val'], batch_size=self.batch_size, shuffle=False) 
    #                    for k in ['jj', 'ss', 'js']]
    #         return zip(*loaders)

    #     def test_dataloader(self):
    #         loaders = [DataLoader(self.triplets_dict[k]['test'], batch_size=self.batch_size, shuffle=False) 
    #                    for k in ['jj', 'ss', 'js']]
    #         return zip(*loaders)


    # def objective(trial, triplets, job_nodes, skill_nodes, device='cuda'):
    #     run_name = f"trial_{trial.number}"

    #     with mlflow.start_run(nested=True, run_name=run_name):
    #         lr = trial.suggest_float("lr", 1e-4, 1e-1, log=True)
    #         batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])
    #         dim = trial.suggest_categorical("dim", [32, 50, 64])
    #         lambda_reg = trial.suggest_float("lambda_reg", 1e-4, 1e-2, log=True)
    #         patience = trial.suggest_int("patience", 3, 10)
    #         epochs = trial.suggest_int("epochs", 20, 100)

    #         params = {"lr": lr, "batch_size": batch_size, "dim": dim, "lambda_reg": lambda_reg, 
    #                   "patience": patience, "epochs": epochs}
    #         mlflow.log_params(params)

    #         job2idx = {j: i for i, j in enumerate(job_nodes)}
    #         skill2idx = {s: i for i, s in enumerate(skill_nodes)}

    #         triplets_data = {
    #             'jj': {'triplets': triplets['jj'], 'x_idx': job2idx, 'y_idx': job2idx, 'z_idx': job2idx},
    #             'ss': {'triplets': triplets['ss'], 'x_idx': skill2idx, 'y_idx': skill2idx, 'z_idx': skill2idx},
    #             'js': {'triplets': triplets['js'], 'x_idx': job2idx, 'y_idx': skill2idx, 'z_idx': skill2idx}
    #         }

    #         datamodule = TripletDataModule(triplets_data, batch_size=batch_size)
    #         model = GraphEmbedding(len(job_nodes), len(skill_nodes), dim=dim, lr=lr, lambda_reg=lambda_reg)

    #         trainer = L.Trainer(
    #             max_epochs=epochs,
    #             logger=MLFlowLogger(experiment_name="graph-embedding-hpo", tracking_uri="sqlite:///mlflow_database/mlflow.db"),
    #             enable_progress_bar=False,
    #             enable_checkpointing=False,
    #             accelerator='auto',
    #             devices=1,
    #             callbacks=[EarlyStopping(monitor='val_loss', patience=patience)]
    #         )

    #         trainer.fit(model, datamodule)
    #         val_loss = trainer.callback_metrics['val_loss'].item()

    #         mlflow.log_metrics({"val_loss": val_loss})
    #         mlflow.pytorch.log_model(model, f"model-{random.randint(0, 9999):04d}")

    #         return val_loss


    # with mlflow.start_run(run_name="graph_embedding_hpo") as parent_run:
    #         n_trials = 10
    #         mlflow.log_param("n_trials", n_trials)
    #         mlflow.log_param("device", device)

    #         study = optuna.create_study(direction="minimize")
    #         study.optimize(
    #             lambda trial: objective(trial, triplets, job_nodes, skill_nodes, device),
    #             n_trials=n_trials
    #         )

    #         # Log best trial results
    #         best_trial = study.best_trial
    #         mlflow.log_params(best_trial.params)
    #         mlflow.log_metrics({"best_val_loss": study.best_value})

    #         # Register best model
    #         if best_run_id := best_trial.user_attrs.get("run_id"):
    #             best_model_name = best_trial.user_attrs.get("model_name")
    #             mlflow.register_model(
    #                 model_uri=f"runs:/{best_run_id}/{best_model_name}",
    #                 name="RL-graph-emb"
    #             )
    #             mlflow.log_param("best_child_run_id", best_run_id)
    #             # print(f"\n✅ Best model registered: graph-embedding-best (Run ID: {best_run_id})")
    #             # print(f"   Best validation loss: {study.best_value:.4f}")
    #             # print(f"   Best hyperparameters: {best_trial.params}")
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


@app.cell
def _():
    # import random

    # # Seed for reproducibility
    # random.seed(42)

    # # Example realistic jobs and skills (expandable for larger dataset)
    # job_nodes = [
    #     "Data Scientist", "Data Analyst", "Machine Learning Engineer", "Software Engineer",
    #     "DevOps Engineer", "Business Analyst", "AI Researcher", "Cloud Engineer",
    #     "Frontend Developer", "Backend Developer"
    # ]

    # skill_nodes = [
    #     "Python", "R", "SQL", "Pandas", "NumPy", "Deep Learning", "Neural Networks",
    #     "Machine Learning", "Statistics", "DevOps", "Docker", "Kubernetes", "AWS",
    #     "React", "JavaScript", "Node.js", "API Design", "Data Visualization"
    # ]

    # # Mapping skills to jobs logically
    # job_skill_map = {
    #     "Data Scientist": ["Python", "R", "SQL", "Pandas", "NumPy", "Machine Learning", "Statistics", "Data Visualization"],
    #     "Data Analyst": ["SQL", "Python", "Pandas", "Data Visualization"],
    #     "Machine Learning Engineer": ["Python", "Machine Learning", "Deep Learning", "Neural Networks", "NumPy", "Pandas"],
    #     "Software Engineer": ["Python", "JavaScript", "Node.js", "React", "API Design"],
    #     "DevOps Engineer": ["Docker", "Kubernetes", "AWS", "Linux"],
    #     "Business Analyst": ["SQL", "Data Visualization", "Statistics"],
    #     "AI Researcher": ["Python", "Machine Learning", "Deep Learning", "Neural Networks", "Statistics"],
    #     "Cloud Engineer": ["AWS", "Docker", "Kubernetes", "Linux"],
    #     "Frontend Developer": ["JavaScript", "React", "CSS", "HTML"],
    #     "Backend Developer": ["Python", "Node.js", "API Design", "SQL"]
    # }

    # # --- Generate Job-Job Pairs (1000 pairs) ---
    # job_job_edges = []
    # for _ in range(1000):
    #     j1, j2 = random.sample(job_nodes, 2)
    #     job_job_edges.append(f"{j1} ::: {j2}")

    # # --- Generate Skill-Skill Pairs (1000 pairs) ---
    # skill_skill_edges = []
    # for _ in range(1000):
    #     s1, s2 = random.sample(skill_nodes, 2)
    #     skill_skill_edges.append(f"{s1} ::: {s2}")

    # # --- Generate Skill-Job Pairs (matching job title) ---
    # skill_job_edges = []
    # for job, skills in job_skill_map.items():
    #     for skill in skills:
    #         skill_job_edges.append(f"{skill} ::: {job}")

    # # --- Combine into text file format ---
    # text_data = "===JOB-JOB===\n" + "\n".join(job_job_edges) + "\n\n"
    # text_data += "===SKILL-SKILL===\n" + "\n".join(skill_skill_edges) + "\n\n"
    # text_data += "===SKILL-JOB===\n" + "\n".join(skill_job_edges)

    # # Save to file
    # file_path = "/mnt/data/graph_edges_large.txt"
    # with open(file_path, "w") as f:
    #     f.write(text_data)

    # # file_path
    return


@app.cell
def _():
    # # num_jobs = 4
    # # num_skills = 5
    # dim = 50
    # mean = 0.0
    # std = 0.1
    # seed = 42
    # lambda_reg = 0.001     # slightly stronger regularization than 0.0001
    # num_epochs = 200       # more epochs since dataset is small
    # lr = 0.01              # smaller learning rate for stability
    # batch_size = 128       # large enough to cover many triplets per batch
    # device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # dtype = torch.float32
    return


@app.cell
def _():
    # torch.manual_seed(seed)
    return


@app.cell
def _():
    # def parse_edges(file_path):
    #     section = None
    #     job_job, skill_skill, skill_job = [], [], []

    #     with open(file_path, "r") as f:
    #         for line in f:
    #             line = line.strip()
    #             if not line:
    #                 continue

    #             if line.startswith("===JOB-JOB==="):
    #                 section = "job_job"
    #             elif line.startswith("===SKILL-SKILL==="):
    #                 section = "skill_skill"
    #             elif line.startswith("===SKILL-JOB==="):
    #                 section = "skill_job"
    #             else:
    #                 src, dst = [x.strip() for x in line.split(":::")]
    #                 if section == "job_job":
    #                     job_job.append((src, dst))
    #                 elif section == "skill_skill":
    #                     skill_skill.append((src, dst))
    #                 elif section == "skill_job":
    #                     skill_job.append((src, dst))

    #     return job_job, skill_skill, skill_job
    return


@app.cell
def _():
    # job_job, skill_skill, skill_job = parse_edges('graph_edges.txt')
    return


@app.cell
def _():
    # # assuming these are already defined
    # # job_job, skill_skill, skill_job = parse_edges('graph_edges.txt')

    # # --- Unique job nodes ---
    # job_nodes = set()
    # for src, dst in job_job:
    #     job_nodes.add(src.strip())
    #     job_nodes.add(dst.strip())
    # for _, job in skill_job:  # skill → job edges
    #     job_nodes.add(job.strip())

    # # --- Unique skill nodes ---
    # skill_nodes = set()
    # for src, dst in skill_skill:
    #     skill_nodes.add(src.strip())
    #     skill_nodes.add(dst.strip())
    # for skill, _ in skill_job:  # skill → job edges
    #     skill_nodes.add(skill.strip())

    # # Convert to sorted lists for consistency
    # job_nodes = sorted(job_nodes)
    # skill_nodes = sorted(skill_nodes)
    return


@app.cell
def _():
    # num_jobs = len(job_nodes)
    # num_skills = len(skill_nodes)
    return


@app.cell
def _():
    # W_skill = torch.normal(mean=mean, std=std, size=(num_skills, dim), device=device, dtype=torch.float32, requires_grad=True)
    # W_job = torch.normal(mean=mean, std=std, size=(num_jobs, dim), device=device, dtype=torch.float32, requires_grad=True)
    return


@app.cell
def _():
    # W_skill
    return


@app.cell
def _():
    # W_job
    return


@app.cell
def _():
    # W_skill_T = W_skill.T
    return


@app.cell
def _():
    # W_skill_T
    return


@app.cell
def _():
    # job2tensor = {name: W_job[i] for i, name in enumerate(job_nodes)}
    # skill2tensor = {name: W_skill[i] for i, name in enumerate(skill_nodes)}
    return


@app.cell
def _():
    # job2tensor
    return


@app.cell
def _():
    # skill2tensor
    return


@app.cell
def _():
    # job_skill_tensor = {**job2tensor, **skill2tensor}
    return


@app.cell
def _():
    # job_skill_tensor
    return


@app.cell
def _():
    # def generate_triplets(job_job, skill_skill, skill_job, job_nodes, skill_nodes):
    #     # --- Helper to build adjacency dictionary ---
    #     def build_adj_dict(edges):
    #         adj = {}
    #         for src, dst in edges:
    #             adj.setdefault(src, set()).add(dst)
    #         return adj

    #     job_adj = build_adj_dict(job_job)
    #     skill_adj = build_adj_dict(skill_skill)
    #     skill_job_adj = build_adj_dict(skill_job)

    #     # --- D_jj ---
    #     D_jj = []
    #     for j1 in job_nodes:
    #         pos = job_adj.get(j1, set())
    #         neg = set(job_nodes) - pos - {j1}
    #         for j2 in pos:
    #             for j3 in neg:
    #                 D_jj.append((j1, j2, j3))

    #     # --- D_ss ---
    #     D_ss = []
    #     for s1 in skill_nodes:
    #         pos = skill_adj.get(s1, set())
    #         neg = set(skill_nodes) - pos - {s1}
    #         for s2 in pos:
    #             for s3 in neg:
    #                 D_ss.append((s1, s2, s3))

    #     # --- D_js ---
    #     D_js = []
    #     for j1 in job_nodes:
    #         pos_skills = {s for s in skill_nodes if (s, j1) in skill_job}
    #         neg_skills = set(skill_nodes) - pos_skills
    #         for s1 in pos_skills:
    #             for s2 in neg_skills:
    #                 D_js.append((j1, s1, s2))

    #     return D_jj, D_ss, D_js
    return


@app.cell
def _():
    # D_jj, D_ss, D_js = generate_triplets(job_job, skill_skill, skill_job, job_nodes, skill_nodes)
    return


@app.cell
def _():
    # D_jj
    return


@app.cell
def _():
    # import random
    # num_samples_jj = 5
    # num_samples_ss = 10
    # num_samples_js = 100

    # # Sample independently (uniformly)
    # sampled_D_jj = random.choices(D_jj, k=num_samples_jj)
    # sampled_D_ss = random.choices(D_ss, k=num_samples_ss)
    # sampled_D_js = random.choices(D_js, k=num_samples_js)
    return


@app.cell
def _():
    # def triplet_loss(triplets, embeddings, use_sigmoid=True, device='cuda'):
    #     """
    #     triplets: list of (x, y, z)
    #     embeddings: dict[node] -> embedding vector
    #     """
    #     total_loss = 0.0
    #     for x, y, z in triplets:
    #         wx = embeddings[x]
    #         wy = embeddings[y]
    #         wz = embeddings[z]
    #         pos_sim = torch.dot(wx, wy)
    #         neg_sim = torch.dot(wx, wz)
    #         diff = pos_sim - neg_sim
    #         if use_sigmoid:
    #             total_loss = total_loss + -torch.log(torch.sigmoid(diff) + 1e-08)
    #         else:
    #             total_loss = total_loss + F.softplus(-diff)
    #     return total_loss / len(triplets)  # equivalent numerically stable form
    return


@app.cell
def _():
    # job2tensor['Data Analyst']
    return


@app.cell
def _():
    # # D_jj, D_ss, D_js are your triplet sets
    # loss_jj = triplet_loss(D_jj, job2tensor)
    # loss_ss = triplet_loss(D_ss, skill2tensor)
    # loss_js = triplet_loss(D_js, job_skill_tensor)
    return


@app.cell
def _():
    # lambda_reg = 0.01
    return


@app.cell
def _():
    # O = loss_jj + loss_ss + loss_js + lambda_reg * (torch.linalg.matrix_norm(W_job.T)**2 + torch.linalg.matrix_norm(W_skill.T)**2)
    return


@app.cell
def _():
    # def get_mini_batches(triplets, batch_size):
    #     """
    #     Yields mini-batches from a list of triplets
    #     """
    #     triplets = triplets.copy()
    #     random.shuffle(triplets)
    #     for i in range(0, len(triplets), batch_size):
    #         yield triplets[i:i+batch_size]
    return


@app.cell
def _():
    # optimizer = torch.optim.SGD([W_job, W_skill], lr=0.01)
    # --- Training loop ---
    return


@app.cell
def _():
    # W_job
    return


@app.cell
def _():
    # W_job.T
    return


@app.cell
def _():
    # num_epochs = 2000
    return


@app.cell
def _():
    # optimizer = torch.optim.SGD([W_job, W_skill], lr=lr)
    return


@app.cell
def _():
    # # optimizer = torch.optim.SGD([W_job, W_skill], lr=lr)
    # for epoch in range(num_epochs):
    #     total_loss_epoch = 0.0
    #     jj_batches = list(get_mini_batches(D_jj, batch_size))
    #     ss_batches = list(get_mini_batches(D_ss, batch_size))
    #     js_batches = list(get_mini_batches(D_js, batch_size))
    #     num_batches = min(len(jj_batches), len(ss_batches), len(js_batches))
    #     for jj_batch, ss_batch, js_batch in zip(jj_batches, ss_batches, js_batches):
    #         loss_jj = triplet_loss(jj_batch, job2tensor)  # Convert generators to lists so len() works
    #         loss_ss = triplet_loss(ss_batch, skill2tensor)
    #         loss_js = triplet_loss(js_batch, job_skill_tensor)
    #         reg_term = lambda_reg * (torch.linalg.matrix_norm(W_job.T) ** 2 + torch.linalg.matrix_norm(W_skill.T) ** 2)
    #         total_loss = loss_jj + loss_ss + loss_js + reg_term
    #         optimizer.zero_grad()
    #         total_loss.backward()
    #         optimizer.step()
    #         total_loss_epoch = total_loss_epoch + total_loss.item()  # Compute losses for each type
    #     print(f'Epoch {epoch + 1}, Avg Loss: {total_loss_epoch / num_batches:.6f}')  # Regularization term (Frobenius norm)  # Total objective  # Backpropagation  # Average loss per epoch  # print(f"Epoch {epoch+1}, Avg Loss: {total_loss_epoch:.6f}")
    return


@app.cell
def _():
    # W_job
    return


@app.cell
def _():
    # W_skill
    return


@app.cell
def _():
    # job2tensor_updated = {name: W_job[i] for i, name in enumerate(job_nodes)}
    # skill2tensor_updated = {name: W_skill[i] for i, name in enumerate(skill_nodes)}
    return


@app.cell
def _():
    # # labels = list(job2tensor_updated.keys())
    # # X = torch.stack(list(job2tensor_updated.values())).detach().cpu().numpy()


    # # labels = list(skill2tensor_updated.keys())
    # # X = torch.stack(list(skill2tensor_updated.values())).detach().cpu().numpy()

    # labels = list(job_skill_tensor.keys())
    # X = torch.stack(list(job_skill_tensor.values())).detach().cpu().numpy()

    # # job_skill_tensor
    return


@app.cell
def _():
    # # --- t-SNE ---
    # _tsne = TSNE(n_components=2, perplexity=5, random_state=42, learning_rate='auto', init='pca')
    # X_2d = _tsne.fit_transform(X)
    return


@app.cell
def _():
    # # # --- Plot ---
    # # plt.figure(figsize=(7, 6))
    # # plt.scatter(X_2d[:, 0], X_2d[:, 1], s=120, color='mediumseagreen', alpha=0.7, edgecolor='k')
    # _fig = go.Figure()
    # # for i, label in enumerate(labels):
    # #     plt.text(X_2d[i, 0] + 0.02, X_2d[i, 1] + 0.02, label, fontsize=9, weight='bold')
    # _fig.add_trace(go.Scatter(x=X_2d[:, 0], y=X_2d[:, 1], mode='markers+text', text=labels, textposition='top center', marker=dict(size=12, color='mediumseagreen', line=dict(width=1, color='black'), opacity=0.7)))
    # # plt.title("t-SNE Visualization of Job Embeddings")
    # # plt.xlabel("t-SNE dimension 1")
    # # plt.ylabel("t-SNE dimension 2")
    # # plt.grid(True, linestyle='--', alpha=0.6)
    # # plt.show()
    # _fig.update_layout(title='t-SNE Visualization of Job Embeddings', xaxis_title='t-SNE dimension 1', yaxis_title='t-SNE dimension 2', template='plotly_white', height=800)
    # # --- Plotly Scatter ---
    # _fig.show()  # increase graph height
    return


@app.cell
def _():
    # all_labels = list(job2tensor_updated.keys()) + list(skill2tensor_updated.keys())
    # all_embeddings = torch.stack([job2tensor_updated[label] for label in job2tensor_updated] + [skill2tensor_updated[label] for label in skill2tensor_updated]).cpu().detach().numpy()
    # _tsne = TSNE(n_components=2, perplexity=10, random_state=42)
    # X_2d_1 = _tsne.fit_transform(all_embeddings)
    # colors = ['mediumseagreen'] * len(job2tensor_updated) + ['tomato'] * len(skill2tensor_updated)
    # _fig = go.Figure()
    # _fig.add_trace(go.Scatter(x=X_2d_1[:, 0], y=X_2d_1[:, 1], mode='markers+text', text=all_labels, textposition='top center', marker=dict(size=12, color=colors, line=dict(width=1, color='black'), opacity=0.8)))
    # _fig.update_layout(title='t-SNE Visualization of Job and Skill Embeddings', xaxis_title='t-SNE dimension 1', yaxis_title='t-SNE dimension 2', template='plotly_white', height=800)
    # _fig.show()
    return


@app.cell
def _():
    # # ---- Objective Components ----
    # def objective(W_, Wp, λ):
    #     # Job–Job similarity
    #     O_jj = torch.norm(A_jj - torch.sigmoid(W @ W.T))**2
    #     # Job–Skill similarity
    #     O_js = torch.norm(A_js - torch.sigmoid(W @ Wp.T))**2
    #     # Skill–Skill similarity
    #     O_ss = torch.norm(A_ss - torch.sigmoid(Wp @ Wp.T))**2
    #     # Regularization (Frobenius norm)
    #     reg = λ * (torch.norm(W)**2 + torch.norm(Wp)**2)
    #     return O_jj + O_js + O_ss + reg
    return


@app.cell
def _():
    # num_jobs_1, num_skills_1, dim_1 = (4, 5, 50)
    # lr_1 = 0.01
    # # Setup
    # λ = 0.001
    # device_1 = 'cuda' if torch.cuda.is_available() else 'cpu'
    # W = torch.normal(0.0, 0.1, size=(num_jobs_1, dim_1), device=device_1, requires_grad=True)  # regularization term
    # Wp = torch.normal(0.0, 0.1, size=(num_skills_1, dim_1), device=device_1, requires_grad=True)
    # A_jj = torch.randint(0, 2, (num_jobs_1, num_jobs_1), dtype=torch.float32, device=device_1)
    # # Initialize W and W′ with Normal(0, 0.1)
    # A_js = torch.randint(0, 2, (num_jobs_1, num_skills_1), dtype=torch.float32, device=device_1)
    # A_ss = torch.randint(0, 2, (num_skills_1, num_skills_1), dtype=torch.float32, device=device_1)

    # # Example adjacency matrices (toy)
    # def objective(W, Wp, λ):
    #     O_jj = torch.norm(A_jj - torch.sigmoid(W @ W.T)) ** 2
    #     O_js = torch.norm(A_js - torch.sigmoid(W @ Wp.T)) ** 2
    #     O_ss = torch.norm(A_ss - torch.sigmoid(Wp @ Wp.T)) ** 2
    # # ---- Objective Components ----
    #     reg = λ * (torch.norm(W) ** 2 + torch.norm(Wp) ** 2)
    #     return O_jj + O_js + O_ss + reg  # Job–Job similarity
    # for t in range(100):
    #     loss = objective(W, Wp, λ)  # Job–Skill similarity
    #     loss.backward()
    #     with torch.no_grad():  # Skill–Skill similarity
    #         W = W - lr_1 * W.grad
    #         Wp = Wp - lr_1 * Wp.grad  # Regularization (Frobenius norm)
    #     W.grad.zero_()
    #     Wp.grad.zero_()
    #     if (t + 1) % 10 == 0:
    # # ---- Training Step ----
    #         print(f'Iter {t + 1} | Loss = {loss.item():.6f}')  # number of iterations  # Compute gradients: ∂O/∂W and ∂O/∂W'  # gradient update (no autograd tracking)  # Reset gradients to zero for next iteration
    return


if __name__ == "__main__":
    app.run()
