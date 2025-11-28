import marimo

__generated_with = "0.17.7"
app = marimo.App(width="full")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Combining Embeddings Experimentation
    
    This notebook is a playground for experimenting with different techniques to combine embeddings, such as concatenation, attention mechanisms, and clustering.
    It uses synthetic data to demonstrate these concepts and visualizes the results using t-SNE.
    """)
    return


@app.cell
def _():
    import torch
    import torch.nn.functional as F
    import torch.nn as nn
    import marimo as mo
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.manifold import TSNE
    from itertools import combinations
    from sklearn.preprocessing import StandardScaler
    import plotly.graph_objects as go
    return (
        F,
        StandardScaler,
        TSNE,
        combinations,
        go,
        mo,
        nn,
        np,
        plt,
        torch,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## 1. Simple Concatenation""")
    return


@app.cell
def _(torch):
    # Synthetic data
    years = [4.0, 2.0, 3.0]
    years_tensor = torch.tensor(years, dtype=torch.float32)
    
    # Normalize
    years_tensor_norm = years_tensor / 100
    
    # Create random embeddings
    embedding_dim = 8
    skill_embedding = torch.randn(len(years), embedding_dim)
    
    # Concatenate
    years_tensor_unsqueezed = years_tensor_norm.unsqueeze(-1)
    combined_input = torch.cat([skill_embedding, years_tensor_unsqueezed], dim=-1)
    
    print(f"Combined shape: {combined_input.shape}")
    return (
        combined_input,
        embedding_dim,
        skill_embedding,
        years,
        years_tensor,
        years_tensor_norm,
        years_tensor_unsqueezed,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## 2. Attention Mechanism""")
    return


@app.cell
def _(torch):
    batch_size = 3
    seq_len = 5
    embed_dim = 8

    # Random embeddings
    embeddings = torch.randn(batch_size, seq_len, embed_dim)

    # Mask (1 = valid embedding, 0 = padding)
    mask = torch.tensor([
        [1, 1, 1, 0, 0],  # entity 1 has 3 embeddings
        [1, 1, 1, 1, 1],  # entity 2 has 5 embeddings
        [1, 1, 0, 0, 0]   # entity 3 has 2 embeddings
    ])
    return batch_size, embed_dim, embeddings, mask, seq_len


@app.cell
def _(embed_dim, nn):
    # ----- Attention mechanism -----
    attn_fc1 = nn.Linear(embed_dim, embed_dim)
    attn_fc2 = nn.Linear(embed_dim, 1)
    output_fc = nn.Linear(embed_dim, embed_dim)
    return attn_fc1, attn_fc2, output_fc


@app.cell
def _(F, attn_fc1, attn_fc2, embeddings, mask, output_fc, torch):
    # Compute attention scores
    scores = torch.tanh(attn_fc1(embeddings))           # (batch, seq, embed_dim)
    scores = attn_fc2(scores).squeeze(-1)               # (batch, seq)

    # Apply mask before softmax
    scores_masked = scores.masked_fill(mask == 0, float('-inf'))

    # Normalize scores
    attn_weights = F.softmax(scores_masked, dim=1).unsqueeze(-1)  # (batch, seq, 1)
    weighted_sum = (embeddings * attn_weights).sum(dim=1)  # (batch, embed_dim)

    # Optional linear transformation
    final_embeddings = output_fc(weighted_sum)
    
    print(f"Final embeddings shape: {final_embeddings.shape}")
    return attn_weights, final_embeddings, scores, scores_masked, weighted_sum


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## 3. Clustering and Visualization""")
    return


@app.cell
def _():
    # Sample skills and embeddings
    skills = ['Python', 'C++', 'SQL', 'Data Analysis', 'Machine Learning', 'Communication']
    embeddings_dict = {
        "Python":           [0.9, 0.1, 0.0, 0.8, 0.2, 0.1],
        "C++":              [0.85, 0.15, 0.0, 0.75, 0.25, 0.05],
        "SQL":              [0.1, 0.9, 0.2, 0.0, 0.85, 0.1],
        "Data Analysis":    [0.2, 0.8, 0.1, 0.1, 0.9, 0.2],
        "Machine Learning": [0.8, 0.2, 0.0, 0.85, 0.25, 0.15],
        "Communication":    [0.0, 0.1, 0.9, 0.2, 0.1, 0.85]
    }
    return embeddings_dict, skills


@app.cell
def _(np):
    # Function to compute cluster vector
    def cluster_vector(skills_in_cluster, embeddings_dict):
        points = np.array([embeddings_dict[s] for s in skills_in_cluster])
        centroid = points.mean(axis=0)
        dist_to_centroid = np.linalg.norm(points - centroid, axis=1)
        avg_dist = dist_to_centroid.mean()
        size = len(points)
        return np.concatenate([centroid, [avg_dist, size]])
    return (cluster_vector,)


@app.cell
def _(cluster_vector, combinations, embeddings_dict, skills):
    # Generate clusters of size 2 and 3
    cluster_list = []
    cluster_labels = []
    
    for size in [2, 3]:
        for combo in combinations(skills, size):
            vec = cluster_vector(combo, embeddings_dict)
            cluster_list.append(vec)
            cluster_labels.append(', '.join(combo))
    return cluster_labels, cluster_list


@app.cell
def _(StandardScaler, TSNE, cluster_list, go, np):
    cluster_array = np.array(cluster_list)
    
    # Normalize vectors
    scaler = StandardScaler()
    cluster_array_norm = scaler.fit_transform(cluster_array)
    
    # t-SNE 2D projection
    tsne = TSNE(n_components=2, random_state=42, perplexity=5)
    cluster_2d = tsne.fit_transform(cluster_array_norm)
    
    # Create Plotly scatter
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=cluster_2d[:,0],
        y=cluster_2d[:,1],
        mode='markers+text',
        text=cluster_labels, # type: ignore
        textposition='top center',
        marker=dict(size=12, color='skyblue', line=dict(width=1, color='black'))
    ))

    fig.update_layout(
        title="t-SNE of multiple skill clusters (sizes 2-3)",
        xaxis_title="t-SNE dim 1",
        yaxis_title="t-SNE dim 2",
        width=900,
        height=700
    )

    fig.show()
    return cluster_2d, cluster_array, cluster_array_norm, fig, scaler, tsne


if __name__ == "__main__":
    app.run()
