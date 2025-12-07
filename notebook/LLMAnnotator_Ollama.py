import marimo

__generated_with = "0.18.3"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo
    import pandas as pd
    import requests
    import json
    import os
    import time
    from tqdm.auto import tqdm
    return json, mo, os, pd, requests, time, tqdm


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # LLM Annotator using Ollama (Qwen3:8b)

    This notebook annotates the relevance of candidate information to job post information using a local Ollama model (`qwen3:8b`).

    **Steps:**
    1.  **Load Data**: Reads the processed Two-Tower annotation dataset.
    2.  **Define Model**: Connects to local Ollama instance.
    3.  **Annotate**: Iterates through the dataset and generates scores (0/1) and feedback.
    4.  **Save**: Saves annotations incrementally to CSV.
    """)
    return


@app.cell
def _(pd):
    # Load the processed dataset
    # We assume the previous notebook has already run and created this file.
    # If not, we might need to recreate the logic to generate it.
    # For now, let's try to load it.
    input_file = '../datasets/processed_dataset/two_tower_annotations.csv'

    # Check if file exists, if not, fallback to raw data and process (simplified)
    try:
        df = pd.read_csv(input_file)
        print(f"Loaded processed dataset from {input_file} with {len(df)} rows.")
    except FileNotFoundError:
        print(f"Processed file not found at {input_file}. Please run the data preparation step in the other notebook or adjust path.")
        # Fallback logic could go here, but let's assume it exists for now based on user context.
        df = pd.DataFrame()
    return (df,)


@app.cell
def _(df, os, pd):
    # Logic to read existing annotations
    annotations_dir = '../datasets/annotation_datasets/ollama_annotations'
    existing_ids = set()

    if os.path.exists(annotations_dir):
        for f in os.listdir(annotations_dir):
            if f.endswith('.csv'):
                try:
                    temp_df = pd.read_csv(os.path.join(annotations_dir, f))
                    # Assuming 'post_id' and 'id' are the keys
                    if 'post_id' in temp_df.columns and 'id' in temp_df.columns:
                        for _, existing_row in temp_df.iterrows():
                            existing_ids.add((existing_row['post_id'], existing_row['id']))
                except Exception as e:
                    print(f"Error reading {f}: {e}")

    print(f"Found {len(existing_ids)} already annotated rows.")

    # Filter df
    if not df.empty:
        # Create a set of keys in df for efficient filtering
        # We use a vectorized approach to create a boolean mask
        # But for simplicity and readability with pandas:

        # Add a temporary tuple column for filtering
        # Note: This might be memory intensive if df is huge, but for <100k rows it's fine.
        # A more memory efficient way is to iterate if needed, but let's try this first.

        # Actually, let's use a list comprehension for the mask, it's often faster than apply for simple tuple checks
        # keys = set(zip(df['post_id'], df['id']))
        # This doesn't help filter the DF directly without re-indexing.

        # Let's use the index trick
        df_indexed = df.set_index(['post_id', 'id'])
        # Filter out existing
        # We need to ensure existing_ids contains tuples of the same type as index

        # Let's stick to a simple apply for correctness first, or just iterate if we want to be safe.
        # Given the user wants "fast", let's try to be efficient.

        # Create a set of existing IDs for O(1) lookup
        # existing_ids is already a set of tuples

        # Filter
        # df['is_new'] = df.apply(lambda x: (x['post_id'], x['id']) not in existing_ids, axis=1)
        # filtered_df = df[df['is_new']].drop(columns=['is_new'])

        # To avoid apply overhead:
        filtered_df = df[~df[['post_id', 'id']].apply(tuple, axis=1).isin(existing_ids)]

        print(f"Rows remaining after filtering: {len(filtered_df)}")

        n_sample = 50000
        if len(filtered_df) < n_sample:
            n_sample = len(filtered_df)

        work_df = filtered_df.sample(n=n_sample, random_state=42).copy() if not filtered_df.empty else pd.DataFrame()
    else:
        work_df = pd.DataFrame()

    print(f"Working with {len(work_df)} samples.")
    return (work_df,)


@app.cell
def _(json, requests):
    def query_ollama(model, prompt, system_prompt=""):
        url = "http://localhost:11434/api/generate"

        payload = {
            "model": model,
            "prompt": prompt,
            "system": system_prompt,
            "stream": False,
            "format": "json" # Enforce JSON output
        }

        try:
            response = requests.post(url, json=payload)
            response.raise_for_status()
            return response.json()['response']
        except Exception as e:
            print(f"Error querying Ollama: {e}")
            return None

    def parse_response(response_text):
        try:
            data = json.loads(response_text)
            # Support both old and new keys just in case, but prefer new ones
            label = data.get("Label")
            if label is None:
                 label = data.get("Score") # Fallback to old key if model messes up, though prompt asks for Label

            score = data.get("Score")
            # If score is 0 or 1 and label is missing, it might be the old format.
            # But we want the 0-10 score.
            # Let's trust the keys.

            # If the model returns "Score" as 0/1 (old behavior) instead of 0-10, we might have an issue.
            # But the prompt is explicit.

            return label, score, data.get("Feedback")
        except json.JSONDecodeError:
            # Fallback parsing if JSON is malformed
            print(f"Failed to parse JSON: {response_text}")
            return -1, -1, "Error parsing response"
    return parse_response, query_ollama


@app.cell
def _(parse_response, query_ollama):
    system_prompt = """
    You are an expert HR assistant. Your task is to evaluate the relevance of a candidate for a job post.

    Output MUST be a JSON object with three keys: "Label", "Score", and "Feedback".

    1. "Label": An integer, either 0 or 1.
       - 0: Candidate is largely irrelevant.
       - 1: Candidate demonstrates reasonable relevance.

    2. "Score": An integer between 0 and 10 indicating the quality of the match.
       - 0: Irrelevant. No matching skills or experience.
       - 1: Negligible. Very weak match, maybe one minor skill overlap.
       - 2: Poor. Lacks key skills, very limited relevance.
       - 3: Weak. Some minor transferable skills, but largely unqualified.
       - 4: Below Average. Missing several key requirements, but has some potential.
       - 5: Average. Partial match, meets some core requirements but missing others.
       - 6: Above Average. Meets most core requirements, but lacks depth.
       - 7: Good. Meets core requirements well.
       - 8: Very Good. Strong match, meets all core requirements and some preferred.
       - 9: Excellent. Meets all requirements and exceeds in some areas.
       - 10: Perfect. Ideal candidate, perfect match for all requirements.

    3. "Feedback": A short 1-line string explaining the score.
    """

    def annotate_row(job_info, candidate_info, model="qwen3:8b"):
        prompt = f"""
        Job Post Information:
        {job_info}

        Candidate Information:
        {candidate_info}

        Determine if the candidate matches the job post.../datasets/annotation_datasets/ollama_annotations/annotations_ollama_1765016637.csv
        Respond with JSON: {{"Label": 0 or 1, "Score": 0-10, "Feedback": "..."}}
        """

        response_text = query_ollama(model, prompt, system_prompt)
        if response_text:
            return parse_response(response_text)
        return -1, -1, "API Error"
    return (annotate_row,)


@app.cell
def _(annotate_row, os, pd, time, tqdm, work_df):
    # Output directory
    save_dir = '../datasets/annotation_datasets/ollama_annotations'
    os.makedirs(save_dir, exist_ok=True)
    output_file = os.path.join(save_dir, f'annotations_ollama_{int(time.time())}.csv')

    results = []

    if work_df.empty:
        print("Error: Dataset is empty. Cannot annotate.")
    else:
        print(f"Starting annotation of {len(work_df)} rows...")

        # Use tqdm for progress bar
        try:
            for index, current_row in tqdm(work_df.iterrows(), total=len(work_df), desc="Annotating"):
                job_info = current_row.get('job_post_information', '')
                cand_info = current_row.get('candidate_information', '')

                if not job_info or not cand_info:
                    continue

                label, score, feedback = annotate_row(job_info, cand_info)

                results.append({
                    'post_id': current_row.get('post_id'),
                    'id': current_row.get('id'),
                    'job_post_information': job_info,
                    'candidate_information': cand_info,
                    'Label': label,
                    'Score': score,
                    'Feedback': feedback
                })

                if len(results) % 10 == 0:
                    pd.DataFrame(results).to_csv(output_file, index=False)
        except KeyboardInterrupt:
            print("\nAnnotation interrupted by user.")
            print("Saving progress so far...")
        except Exception as e:
            print(f"Error occurred during annotation: {e}")
            print("Saving progress so far...")
        finally:
            if results:
                pd.DataFrame(results).to_csv(output_file, index=False)
                print(f"Process ended. Saved {len(results)} rows to {output_file}")
            else:
                print("No results generated.")
    return output_file, results


@app.cell
def _(mo, output_file, results):
    if results:
        mo.md(f"### Annotation Complete\nSaved {len(results)} rows to `{output_file}`")
        mo.ui.table(results)
    return


if __name__ == "__main__":
    app.run()
