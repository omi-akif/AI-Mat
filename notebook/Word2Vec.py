import marimo

__generated_with = "0.17.7"
app = marimo.App(width="full")


@app.cell
def _():
    from datasets import load_dataset
    from gensim.utils import simple_preprocess
    from gensim.models import Word2Vec
    import logging
    return Word2Vec, logging, simple_preprocess


@app.cell
def _(logging):
    logging.basicConfig(format="%(asctime)s : %(levelname)s : %(message)s", level=logging.INFO)
    return


@app.cell
def _(simple_preprocess):
    with open('helping_datasets/job_candidate_text_data_word2vec.txt', 'r') as f:
        sentences = [simple_preprocess(line) for line in f]
    return (sentences,)


@app.cell
def _(sentences):
    len(sentences)
    return


@app.cell
def _(sentences):
    sentences[2]
    return


@app.cell
def _(Word2Vec, sentences):
    # Train Word2Vec
    word2_vec_model = Word2Vec(
        sentences=sentences,
        vector_size=128,
        window=8,
        min_count=5,
        workers=4,
        sg=1,
        epochs=10 
    )
    return (word2_vec_model,)


@app.cell
def _(word2_vec_model):
    # 1. Check vocabulary size
    print(f"Vocabulary size: {len(word2_vec_model.wv)}")

    # 2. Check word similarity (manual inspection)
    # Words that are semantically similar should have high similarity scores (close to 1)
    print("Similarity between 'python' and 'java':", word2_vec_model.wv.similarity('python', 'java'))
    print("Similarity between 'python' and 'dog':", word2_vec_model.wv.similarity('python', 'dog'))

    # 3. Find most similar words
    print("\nMost similar to 'python':")
    print(word2_vec_model.wv.most_similar('marketing', topn=20))

    print("\nMost similar to 'developer':")
    print(word2_vec_model.wv.most_similar('developer', topn=20))

    # 4. Word analogies (if you have meaningful relationships)
    # Example: python is to django as java is to ?
    try:
        result = word2_vec_model.wv.most_similar(positive=['django', 'java'], negative=['python'], topn=5)
        print("\nWord analogy - django is to python as ? is to java:")
        print(result)
    except:
        print("Not enough common words for analogy")

    # 5. Get model training info
    print(f"\nTraining epochs: {word2_vec_model.epochs}")
    print(f"Vector size: {word2_vec_model.vector_size}")
    print(f"Window size: {word2_vec_model.window}")
    print(f"Min count: {word2_vec_model.min_count}")
    return


@app.cell
def _(word2_vec_model):
    word2_vec_model.wv.save("models/job_candidate_word2vec.kv")
    return


if __name__ == "__main__":
    app.run()
