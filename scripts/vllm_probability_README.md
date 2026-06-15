  Mode 3: Batch Processing (most useful)

  # First generate text with vllm_generate.py
  python generate_text/vllm_generate.py --model pythia-410m --output generated_outputs.pkl

  # Then extract logits for all generated sequences
  python vllm_probability_distributions.py \
      --model pythia-410m \
      --pickle generated_outputs.pkl \
      --output logits_results.pkl



