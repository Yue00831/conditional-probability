import torch
import pandas as pd
from minicons import scorer

# Ensure we use GPU if available
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load DistilGPT2 Scorer
model_scorer = scorer.IncrementalLMScorer("distilgpt2", device=device)

# Load dataset
input_file = "/your input file path/your_dataset_file.xlsx"
df = pd.read_excel(input_file)

def calculate_conditional_probability(context, target):
    """
    Computes log conditional probability of `target` given `context` using DistilGPT2.
    """
    prefixes = [context]  # Context is the prefix
    queries = [target]  # Target is what we want probability for
    result = model_scorer.conditional_score(prefixes, queries)  # Log P(target | context)
    return result[0]  # Return log probability

# Compute conditional probability for each row
df["log_conditional_probability"] = df.apply(lambda row: 
    calculate_conditional_probability(row["English_context"], row["English_target"]), axis=1)

# Save results
output_file = "/your output file path/your_result_file.xlsx"
df.to_excel(output_file, index=False)

print(f"Saved results to {output_file}")
