from together import Together
import pandas as pd
import math
import json
from tqdm import tqdm
import os
import sys

# Check arguments
if len(sys.argv) != 3:
    print("Usage: python script_name.py <input_csv_path> <output_json_path>")
    sys.exit(1)

input_csv_path = sys.argv[1]
output_json_path = sys.argv[2]

# Load Together API
os.environ["TOGETHER_API_KEY"] = None
client = Together()

# Load data
data = pd.read_csv(input_csv_path)

# Prompt template
template = """The following text describes an emotional experience shared on social media. The emotion word has been replaced with a <mask> token. Based on the context, predict the most suitable emotion word to replace <mask>. Provide only the predicted emotion word, with no additional text or reasoning.

Text:
{segment}

Answer:
"""

# Try to load previous results if any
try:
    with open(output_json_path, "r") as f:
        results = json.load(f)
except FileNotFoundError:
    results = {}

# Start processing
for i in tqdm(range(len(data)), desc="Processing"):
    row_index = str(data.iloc[i]['index'])

    # Skip if already processed
    if row_index in results:
        continue

    prompt = template.format(segment=data.iloc[i]['masked_SIT'])

    response = client.chat.completions.create(
        model="deepseek-ai/DeepSeek-V3",
        messages=[{"role": "user", "content": prompt}],
        logprobs=1,
        temperature=0
    )

    tokens = response.choices[0].logprobs.tokens
    logprobs = response.choices[0].logprobs.token_logprobs
    output_text = response.choices[0].message.content

    token_probs = []
    cumulative_logprob = 0.0

    for token, logprob in zip(tokens, logprobs):
        prob = math.exp(logprob)
        cumulative_logprob += logprob
        token_probs.append({
            "token": token,
            "probability": prob
        })

    # Store the result
    results[row_index] = {
        "tokens_with_probabilities": token_probs,
        "final_output": output_text,
        "cumulative_probability": math.exp(cumulative_logprob)
    }

    # Save every 10 examples
    if (i + 1) % 10 == 0 or i == len(data) - 1:
        with open(output_json_path, "w") as f:
            json.dump(results, f, indent=2)

print("Processing complete. Results saved to:", output_json_path)
