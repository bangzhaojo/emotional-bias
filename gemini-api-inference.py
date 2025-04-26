import google.generativeai as genai
import pandas as pd
import math
import json
from tqdm import tqdm
import sys

# Check arguments
if len(sys.argv) != 3:
    print("Usage: python script_name.py <input_csv_path> <output_json_path>")
    sys.exit(1)

input_csv_path = sys.argv[1]
output_json_path = sys.argv[2]

# Configure API
genai.configure(api_key=None)

# Load model
model = genai.GenerativeModel("gemini-1.5-pro")

# Load input CSV
data = pd.read_csv(input_csv_path)

# Prompt template
template = """The following text describes an emotional experience shared on social media. The emotion word has been replaced with a <mask> token. Based on the context, predict the most suitable emotion word to replace <mask>. Provide only the predicted emotion word, with no additional text or reasoning.

Text:
{segment}

Answer:
"""

# Load previous results if any
try:
    with open(output_json_path, "r") as f:
        results = json.load(f)
except FileNotFoundError:
    results = {}

# Process data
for i in tqdm(range(len(data)), desc="Processing"):
    row_index = str(data.iloc[i]['index'])

    # Skip if already processed
    if row_index in results:
        continue

    prompt = template.format(segment=data.iloc[i]['masked_SIT'])

    try:
        response = model.generate_content(
            prompt,
            generation_config={
                "temperature": 0
            }
        )

        avg_logprob = response.candidates[0].avg_logprobs  # already a float
        token_count = response.usage_metadata.candidates_token_count
        cumulative_logprob = avg_logprob * token_count
        cumulative_prob = math.exp(cumulative_logprob)
        output_text = response.text

    except Exception as e:
        cumulative_logprob = None
        cumulative_prob = None
        output_text = None
        print(f"Error at row {i}: {e}")

    # Store results
    results[row_index] = {
        "final_output": output_text,
        "logprob": cumulative_logprob,
        "probability": cumulative_prob
    }

    # Save every 10 rows
    if (i + 1) % 10 == 0 or i == len(data) - 1:
        with open(output_json_path, "w") as f:
            json.dump(results, f, indent=2)

print("Processing complete. Results saved to:", output_json_path)
