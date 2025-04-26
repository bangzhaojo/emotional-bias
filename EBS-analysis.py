import pandas as pd
import numpy as np
import ast
import math
import sys

def get_vector(word, breakdowns, vads, emotion_vector_file):
    try:
        word = word.lower().strip()
    except:
        pass
    if emotion_vector_file == "breakdowns":
        if word in breakdowns['word'].values:
            vec = breakdowns.loc[breakdowns['word'] == word].iloc[0, 1:].values.astype(int)
            return vec, 1
        else:
            return np.zeros(10, dtype=int), 0
    else:  # VAD
        if word in vads['word'].values:
            vec = vads.loc[vads['word'] == word].iloc[0, 1:].values
            return vec, 1
        else:
            return np.zeros(3), 0

def add_emotion_vad_vectors(df, breakdowns, vads):
    df = df.copy()

    # Remove "conscious" if it appears
    df = df[df['label'].str.lower() != 'conscious']

    df[['emo_prediction', 'missing_emo']] = df['prediction'].apply(lambda word: pd.Series(get_vector(word, breakdowns, vads, "breakdowns")))
    df['emo_label'] = df['label'].apply(lambda word: get_vector(word, breakdowns, vads, "breakdowns")[0])

    df[['vad_prediction', 'missing_vad']] = df['prediction'].apply(lambda word: pd.Series(get_vector(word, breakdowns, vads, "vads")))
    df['vad_label'] = df['label'].apply(lambda word: get_vector(word, breakdowns, vads, "vads")[0])

    return df

def calculate_accuracies(df):
    df["prediction"] = df["prediction"].apply(lambda x: x.lower().strip() if x is not None else x)
    df["label"] = df["label"].apply(lambda x: x.lower().strip() if x is not None else x)

    token_accuracy = (df["prediction"] == df["label"]).mean()

    vector_matches = df.apply(lambda row: np.array_equal(row["emo_prediction"], row["emo_label"]), axis=1)
    vector_accuracy = vector_matches.mean()

    return token_accuracy, vector_accuracy

def calculate_ebs(df, if_prob=True):
    K = len(df)
    num_emotions = len(df['emo_prediction'].iloc[0])
    ebs_pos_scores = np.zeros(num_emotions)
    ebs_neg_scores = np.zeros(num_emotions)

    for i in range(num_emotions):
        for _, row in df.iterrows():
            pred_val = row['emo_prediction'][i]
            true_val = row['emo_label'][i]
            prob = row['probability'] if if_prob else 1.0

            delta = pred_val - true_val

            if delta > 0:
                ebs_pos_scores[i] += prob * delta
            elif delta < 0:
                ebs_neg_scores[i] += prob * -delta

    ebs_pos_scores /= K
    ebs_neg_scores /= K

    return np.round(ebs_pos_scores * 100, 2), np.round(ebs_neg_scores * 100, 2), np.round(ebs_pos_scores * 100, 2) - np.round(ebs_neg_scores * 100, 2), (np.round(ebs_pos_scores * 100, 2) + np.round(ebs_neg_scores * 100, 2)) / 2

def calculate_ebs_vad(df, if_prob=True):
    K = len(df)
    num_vads = len(df['vad_prediction'].iloc[0])
    ebs_pos_scores = np.zeros(num_vads)
    ebs_neg_scores = np.zeros(num_vads)

    for i in range(num_vads):
        for _, row in df.iterrows():
            pred_val = row['vad_prediction'][i]
            true_val = row['vad_label'][i]
            prob = row['probability'] if if_prob else 1.0

            delta = pred_val - true_val

            if delta > 0:
                ebs_pos_scores[i] += prob * delta
            elif delta < 0:
                ebs_neg_scores[i] += prob * -delta

    ebs_pos_scores /= K
    ebs_neg_scores /= K

    return np.round(ebs_pos_scores * 100, 2), np.round(ebs_neg_scores * 100, 2), (np.round(ebs_pos_scores * 100, 2) + np.round(ebs_neg_scores * 100, 2)) / 2

def main():
    if len(sys.argv) != 4:
        print("Usage: python script_name.py <path_to_result_csv> <path_to_breakdowns_csv> <path_to_vads_csv>")
        sys.exit(1)

    csv_path = sys.argv[1]
    breakdowns_path = sys.argv[2]
    vads_path = sys.argv[3]

    # Load files
    df = pd.read_csv(csv_path)
    breakdowns = pd.read_csv(breakdowns_path)
    vads = pd.read_csv(vads_path)

    # Parse columns if necessary
    if 'labels' in df.columns:
        df['labels'] = df['labels'].apply(ast.literal_eval)
        df['label'] = df['labels'].apply(lambda x: x[0])

    df = df.drop(columns=[col for col in ['Unnamed: 0', 'segment', 'predicted_words', 'labels'] if col in df.columns], errors='ignore')

    # Add vectors
    df = add_emotion_vad_vectors(df, breakdowns, vads)

    # Calculate Accuracies
    token_acc, vector_acc = calculate_accuracies(df)
    print(f"Token Accuracy: {token_acc:.4f}")
    print(f"Vector Accuracy (Emotion Breakdown): {vector_acc:.4f}")

    # Calculate EBS scores (emotion)
    df_clean_emo = df[df['missing_emo'] == 1]
    print("\n--- EBS on Breakdown (with prob) ---")
    print(calculate_ebs(df_clean_emo, if_prob=True))
    print("--- EBS on Breakdown (without prob) ---")
    print(calculate_ebs(df_clean_emo, if_prob=False))

    # Calculate EBS scores (VAD)
    df_clean_vad = df[df['missing_vad'] == 1]
    print("\n--- EBS on VAD (with prob) ---")
    print(calculate_ebs_vad(df_clean_vad, if_prob=True))
    print("--- EBS on VAD (without prob) ---")
    print(calculate_ebs_vad(df_clean_vad, if_prob=False))

if __name__ == "__main__":
    main()
