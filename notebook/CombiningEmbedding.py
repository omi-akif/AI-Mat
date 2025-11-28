import marimo

__generated_with = "0.17.7"
app = marimo.App(width="full")


@app.cell
def _():
    import torch
    import torch.nn.functional as F
    import torch.nn as nn
    return F, nn, torch


@app.cell
def _():
    years = [4.0, 2.0, 3.0]  # any length
    return (years,)


@app.cell
def _(torch, years):
    years_tensor = torch.tensor(years, dtype=torch.float32)
    return (years_tensor,)


@app.cell
def _(years_tensor):
    years_tensor
    return


@app.cell
def _(years_tensor):
    # Normalize (optional)
    years_tensor_1 = years_tensor / 100  # scale to 0-1
    return (years_tensor_1,)


@app.cell
def _(years_tensor_1):
    years_tensor_1
    return


@app.cell
def _(torch, years):
    embedding_dim = 8  # for example
    skill_embedding = torch.randn(len(years), embedding_dim)  # one embedding per year
    return embedding_dim, skill_embedding


@app.cell
def _(skill_embedding):
    skill_embedding
    return


@app.cell
def _(embedding_dim, nn):
    output_dim = 16
    linear_layer = nn.Linear(embedding_dim + 1, output_dim)  # +1 for the year feature
    return


@app.cell
def _(years_tensor_1):
    years_tensor_unsqueezed = years_tensor_1.unsqueeze(-1)
    return (years_tensor_unsqueezed,)


@app.cell
def _(skill_embedding, torch, years_tensor_unsqueezed):
    combined_input = torch.cat([skill_embedding, years_tensor_unsqueezed], dim=-1)  # shape -> (num_items, embedding_dim+1)
    return (combined_input,)


@app.cell
def _(combined_input):
    combined_input
    return


@app.cell
def _():
    batch_size = 3
    seq_len = 5
    embed_dim = 8
    return batch_size, embed_dim, seq_len


@app.cell
def _(batch_size, embed_dim, seq_len, torch):
    # Random embeddings
    embeddings = torch.randn(batch_size, seq_len, embed_dim)

    # Mask (1 = valid embedding, 0 = padding)
    mask = torch.tensor([
        [1, 1, 1, 0, 0],  # entity 1 has 3 embeddings
        [1, 1, 1, 1, 1],  # entity 2 has 5 embeddings
        [1, 1, 0, 0, 0]   # entity 3 has 2 embeddings
    ])
    return embeddings, mask


@app.cell
def _(embed_dim, nn):
    # ----- Attention mechanism -----
    attn_fc1 = nn.Linear(embed_dim, embed_dim)
    attn_fc2 = nn.Linear(embed_dim, 1)
    output_fc = nn.Linear(embed_dim, embed_dim)
    return attn_fc1, attn_fc2, output_fc


@app.cell
def _(attn_fc1, attn_fc2, embeddings, torch):
    # Compute attention scores
    scores = torch.tanh(attn_fc1(embeddings))           # (batch, seq, embed_dim)
    scores = attn_fc2(scores).squeeze(-1)               # (batch, seq)
    return (scores,)


@app.cell
def _(mask, scores):
    # Apply mask before softmax
    scores_1 = scores.masked_fill(mask == 0, float('-inf'))
    return (scores_1,)


@app.cell
def _(F, embeddings, scores_1):
    # Normalize scores
    attn_weights = F.softmax(scores_1, dim=1).unsqueeze(-1)  # (batch, seq, 1)
    weighted_sum = (embeddings * attn_weights).sum(dim=1)  # (batch, embed_dim)
    return (weighted_sum,)


@app.cell
def _(output_fc, weighted_sum):
    # Optional linear transformation
    final_embeddings = output_fc(weighted_sum)
    return (final_embeddings,)


@app.cell
def _(final_embeddings):
    final_embeddings
    return


@app.cell
def _():
    # Sample skills
    skills = ['Python', 'C++', 'SQL', 'Data Analysis', 'Machine Learning', 'Communication']
    # 6-dimensional embeddings (random but coherent for illustration)
    # Similarity edges (for illustration, connect
    embeddings_1 = {'Python': [0.9, 0.1, 0.0, 0.8, 0.2, 0.1], 'C++': [0.85, 0.15, 0.0, 0.75, 0.25, 0.05], 'SQL': [0.1, 0.9, 0.2, 0.0, 0.85, 0.1], 'Data Analysis': [0.2, 0.8, 0.1, 0.1, 0.9, 0.2], 'Machine Learning': [0.8, 0.2, 0.0, 0.85, 0.25, 0.15], 'Communication': [0.0, 0.1, 0.9, 0.2, 0.1, 0.85]}
    return


@app.cell
def _():
    # Similarity edges (for illustration, connect nodes with cosine similarity > 0.9 roughly)
    _edges = [('Python', 'C++'), ('Python', 'Machine Learning'), ('C++', 'Machine Learning'), ('SQL', 'Data Analysis')]
    return


@app.cell
def _():
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.manifold import TSNE
    return TSNE, np, plt


@app.cell
def _(TSNE, np):
    skills_1 = ['Python', 'C++', 'SQL', 'Data Analysis', 'Machine Learning', 'Communication']
    embeddings_2 = np.array([[0.9, 0.1, 0.0, 0.8, 0.2, 0.1], [0.85, 0.15, 0.0, 0.75, 0.25, 0.05], [0.1, 0.9, 0.2, 0.0, 0.85, 0.1], [0.2, 0.8, 0.1, 0.1, 0.9, 0.2], [0.8, 0.2, 0.0, 0.85, 0.25, 0.15], [0.0, 0.1, 0.9, 0.2, 0.1, 0.85]])
    _edges = [('Python', 'C++'), ('Python', 'Machine Learning'), ('C++', 'Machine Learning'), ('SQL', 'Data Analysis')]
    _tsne = TSNE(n_components=2, random_state=42, perplexity=3)
    emb_2d = _tsne.fit_transform(embeddings_2)
    return emb_2d, skills_1


@app.cell
def _(emb_2d, plt, skills_1):
    # # Plot
    # plt.figure(figsize=(7,6))
    plt.figure(figsize=(7, 6))
    # # Draw edges
    # for s1, s2 in edges:
    #     i, j = skills.index(s1), skills.index(s2)
    #     x_values = [emb_2d[i,0], emb_2d[j,0]]
    #     y_values = [emb_2d[i,1], emb_2d[j,1]]
    #     plt.plot(x_values, y_values, 'k-', linewidth=1, alpha=0.5)
    for i, skill in enumerate(skills_1):
    # # Draw nodes with labels
    # for i, skill in enumerate(skills):
    #     x, y = emb_2d[i]
    #     plt.scatter(x, y, s=120, color='skyblue', edgecolor='k', zorder=3)
    #     plt.text(x+0.03, y+0.03, skill, fontsize=10, zorder=4)
        x, y = emb_2d[i]
    # plt.title("t-SNE visualization of skill embeddings with labels")
    # plt.xlabel("t-SNE dim 1")
    # plt.ylabel("t-SNE dim 2")
    # plt.grid(True)
    # plt.show()
        plt.scatter(x, y, s=120, color='skyblue', edgecolor='k', zorder=3)
        plt.text(x + 0.03, y + 0.03, skill, fontsize=10, zorder=4)
    # Plot nodes with labels
    plt.title('t-SNE visualization of skill embeddings with labels')
    plt.xlabel('t-SNE dim 1')
    plt.ylabel('t-SNE dim 2')
    plt.grid(True)
    plt.show()
    return


@app.cell
def _(np):
    def cluster_vector(skills_in_cluster, embeddings_dict):
        # Get embeddings
        points = np.array([embeddings_dict[s] for s in skills_in_cluster])
        # Compute centroid (mean of each dimension)
        centroid = points.mean(axis=0)
        # Compute average distance to centroid
        dist_to_centroid = np.linalg.norm(points - centroid, axis=1)
        avg_dist = dist_to_centroid.mean()
        # Cluster size
        size = len(points)
        # Concatenate into single vector
        cluster_vec = np.concatenate([centroid, [avg_dist, size]])
        return cluster_vec
    return (cluster_vector,)


@app.cell
def _():
    # Skill embeddings
    embeddings_dict = {
        "Python":           [0.9, 0.1, 0.0, 0.8, 0.2, 0.1],
        "C++":              [0.85, 0.15, 0.0, 0.75, 0.25, 0.05],
        "SQL":              [0.1, 0.9, 0.2, 0.0, 0.85, 0.1],
        "Data Analysis":    [0.2, 0.8, 0.1, 0.1, 0.9, 0.2],
        "Machine Learning": [0.8, 0.2, 0.0, 0.85, 0.25, 0.15],
        "Communication":    [0.0, 0.1, 0.9, 0.2, 0.1, 0.85]
    }
    return (embeddings_dict,)


@app.cell
def _(cluster_vector, embeddings_dict):
    # Example cluster
    cluster_skills = ["Data Analysis", "Communication", "Machine Learning"]
    vec = cluster_vector(cluster_skills, embeddings_dict)
    return (vec,)


@app.cell
def _(vec):
    vec
    return


@app.cell
def _():
    from itertools import combinations
    from sklearn.preprocessing import StandardScaler
    return StandardScaler, combinations


@app.cell
def _(np):
    # Function to compute cluster vector
    def cluster_vector_1(skills_in_cluster, embeddings_dict):
        points = np.array([embeddings_dict[s] for s in skills_in_cluster])
        centroid = points.mean(axis=0)
        dist_to_centroid = np.linalg.norm(points - centroid, axis=1)
        avg_dist = dist_to_centroid.mean()
        size = len(points)
        return np.concatenate([centroid, [avg_dist, size]])
    return (cluster_vector_1,)


@app.cell
def _(embeddings_dict):
    # Generate clusters of size 2 and 3
    all_skills = list(embeddings_dict.keys())
    cluster_list = []
    cluster_labels = []
    return all_skills, cluster_labels, cluster_list


@app.cell
def _(cluster_labels):
    len(cluster_labels)
    return


@app.cell
def _(
    all_skills,
    cluster_labels,
    cluster_list,
    cluster_vector_1,
    combinations,
    embeddings_dict,
):
    for size in [2, 3]:
        for combo in combinations(all_skills, size):
            vec_1 = cluster_vector_1(combo, embeddings_dict)
            cluster_list.append(vec_1)
            cluster_labels.append(', '.join(combo))
    return


@app.cell
def _(cluster_list, np):
    cluster_array = np.array(cluster_list)
    return (cluster_array,)


@app.cell
def _(cluster_array):
    cluster_array[1]
    return


@app.cell
def _(StandardScaler, cluster_array):
    # Normalize vectors
    scaler = StandardScaler()
    cluster_array_norm = scaler.fit_transform(cluster_array)
    return (cluster_array_norm,)


@app.cell
def _(cluster_array_norm):
    len(cluster_array_norm)
    return


@app.cell
def _(TSNE, cluster_array_norm):
    # t-SNE 2D projection
    _tsne = TSNE(n_components=2, random_state=42, perplexity=5)
    cluster_2d = _tsne.fit_transform(cluster_array_norm)
    return (cluster_2d,)


@app.cell
def _(cluster_2d, cluster_labels):
    # # Plot
    # plt.figure(figsize=(10,8))
    # for i, label in enumerate(cluster_labels):
    #     x, y = cluster_2d[i]
    #     plt.scatter(x, y, s=100, color='skyblue', edgecolor='k', zorder=3)
    #     plt.text(x+0.03, y+0.03, label, fontsize=9, zorder=4)

    # plt.title("t-SNE of multiple skill clusters (sizes 2-3)")
    # plt.xlabel("t-SNE dim 1")
    # plt.ylabel("t-SNE dim 2")
    # plt.grid(True)
    # plt.show()


    import plotly.graph_objects as go

    # Create Plotly scatter
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=cluster_2d[:,0],
        y=cluster_2d[:,1],
        mode='markers+text',
        text=cluster_labels,
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
    return


if __name__ == "__main__":
    app.run()
