import marimo

__generated_with = "0.17.7"
app = marimo.App(width="full")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Word2Vec Model Training

    This notebook trains a Word2Vec model on the job and candidate text data.
    The resulting model is used to generate embeddings for words, which are then used as input features for other models like DLEM.
    """)
    return


@app.cell
def _():
    from datasets import load_dataset
    from gensim.utils import simple_preprocess
    from gensim.models import Word2Vec
    import logging
    import marimo as mo
    import os
    return Word2Vec, logging, mo, os, simple_preprocess


@app.cell
def _(logging):
    logging.basicConfig(format="%(asctime)s : %(levelname)s : %(message)s", level=logging.INFO)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. Load and Preprocess Data
    """)
    return


@app.cell
def _(simple_preprocess):
    input_file = '../datasets/helping_datasets/job_candidate_text_data_word2vec.txt'

    with open(input_file, 'r') as f:
        sentences = [simple_preprocess(line) for line in f]

    print(f"Loaded {len(sentences)} sentences.")
    return (sentences,)


@app.cell
def _(sentences):
    # Preview a sentence
    if len(sentences) > 2:
        print(sentences[2])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. Train Word2Vec Model
    """)
    return


@app.cell
def _(Word2Vec, sentences):
    # Train Word2Vec
    print("Training Word2Vec model...")
    word2_vec_model = Word2Vec(
        sentences=sentences,
        vector_size=128,
        window=8,
        min_count=5,
        workers=4,
        sg=1,
        epochs=10 
    )
    print("Training complete.")
    return (word2_vec_model,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. Model Evaluation and Saving
    """)
    return


@app.cell
def _(word2_vec_model):
    # 1. Check vocabulary size
    print(f"Vocabulary size: {len(word2_vec_model.wv)}")

    # 2. Check word similarity (manual inspection)
    # Words that are semantically similar should have high similarity scores (close to 1)
    try:
        print("Similarity between 'python' and 'java':", word2_vec_model.wv.similarity('python', 'java'))
        print("Similarity between 'python' and 'dog':", word2_vec_model.wv.similarity('python', 'dog'))
    except KeyError as e:
        print(f"Word not found in vocabulary: {e}")

    # 3. Find most similar words
    try:
        print("\nMost similar to 'marketing':")
        print(word2_vec_model.wv.most_similar('marketing', topn=5))
    except KeyError:
        pass

    try:
        print("\nMost similar to 'developer':")
        print(word2_vec_model.wv.most_similar('developer', topn=5))
    except KeyError:
        pass

    # 4. Word analogies (if you have meaningful relationships)
    # Example: python is to django as java is to ?
    try:
        result = word2_vec_model.wv.most_similar(positive=['django', 'java'], negative=['python'], topn=5)
        print("\nWord analogy - django is to python as ? is to java:")
        print(result)
    except:
        print("\nNot enough common words for analogy")

    # 5. Get model training info
    print(f"\nTraining epochs: {word2_vec_model.epochs}")
    print(f"Vector size: {word2_vec_model.vector_size}")
    print(f"Window size: {word2_vec_model.window}")
    print(f"Min count: {word2_vec_model.min_count}")
    return


@app.cell
def _(os, word2_vec_model):
    output_dir = 'models'
    os.makedirs(output_dir, exist_ok=True)

    output_path = os.path.join(output_dir, "job_candidate_word2vec.kv")
    word2_vec_model.wv.save(output_path)
    print(f"Model saved to {output_path}")
    return


if __name__ == "__main__":
    app.run()
