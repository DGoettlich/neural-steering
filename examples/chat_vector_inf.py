"""Run seeded sampled chat-vector inference from a saved vector."""

import textwrap
from pathlib import Path

import torch
from datasets import load_dataset
from transformer_lens import HookedTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer

from neural_steering.steering_vector import AdditiveSteering


MODEL_NAME = "Qwen/Qwen3-4B"
HF_MODEL_NAME = "uzh-echist-org/Ranke-4B-1913"
DATASET = "uzh-echist-org/ranke-1913-sft-dataset"

ARTIFACT_PATH = Path("artifacts/chat_vectors.pt")
STEERING_COEFFICIENT = 1.0
MAX_NEW_TOKENS = 96
EXAMPLE_ROWS = (0, 7, 13, 42, 77, 100)
SEED = 1913
DTYPE = torch.bfloat16
DEVICE = "cuda"
SAMPLING_KWARGS = {
    "do_sample": True,
    "temperature": 0.6,
    "top_p": 0.95,
    "top_k": 20,
}


def load_ranke():
    hf_model = AutoModelForCausalLM.from_pretrained(
        HF_MODEL_NAME,
        torch_dtype=DTYPE,
    )
    tokenizer = AutoTokenizer.from_pretrained(HF_MODEL_NAME)
    return HookedTransformer.from_pretrained_no_processing(
        MODEL_NAME,
        hf_model=hf_model,
        tokenizer=tokenizer,
        dtype=DTYPE,
    )


def print_wrapped(text):
    print(textwrap.fill(text, width=100))


def compare_prompt(row_idx, prompt):
    row_seed = SEED + row_idx

    print(f"\n=== Row {row_idx} | seed {row_seed} ===")
    print("Prompt:")
    print_wrapped(prompt)

    torch.manual_seed(row_seed)
    torch.cuda.manual_seed_all(row_seed)
    base_response = model.generate(
        prompt,
        max_new_tokens=MAX_NEW_TOKENS,
        verbose=False,
        **SAMPLING_KWARGS,
    )
    print("\nBase generation:")
    print_wrapped(base_response)

    torch.manual_seed(row_seed)
    torch.cuda.manual_seed_all(row_seed)
    with chat_vector.steer(model):
        steered_response = model.generate(
            prompt,
            max_new_tokens=MAX_NEW_TOKENS,
            verbose=False,
            **SAMPLING_KWARGS,
        )
    print("\nSteered generation:")
    print_wrapped(steered_response)


if __name__ == "__main__":
    assert torch.cuda.is_available()

    torch.set_grad_enabled(False)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    model = load_ranke().to(DEVICE)

    saved_vectors = torch.load(ARTIFACT_PATH, map_location="cpu", weights_only=False)
    vectors = [
        AdditiveSteering(
            value=item["value"].to(DEVICE),
            src_layer=item["src_layer"],
            method_name=item["method_name"],
        )
        for item in saved_vectors
    ]
    chat_vector = vectors[0] * STEERING_COEFFICIENT
    for vector in vectors[1:]:
        chat_vector = chat_vector & (vector * STEERING_COEFFICIENT)

    rows = [
        row
        for row in load_dataset(DATASET, split="train")
        if "Correct answer:" not in row["formatted_response"]
        and "Reformatted response:" not in row["formatted_response"]
        and not row["formatted_response"].startswith(("Input question:", "Question:"))
    ][: max(EXAMPLE_ROWS) + 1]
    for row_idx in EXAMPLE_ROWS:
        compare_prompt(row_idx, rows[row_idx]["chat_question"])
