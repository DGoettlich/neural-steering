"""Learn and try a chat vector on Ranke-4B-1913."""

import textwrap
from pathlib import Path

import torch
from datasets import load_dataset
from transformer_lens import HookedTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer

from neural_steering.contrastive import ContrastiveSteering


MODEL_NAME = "Qwen/Qwen3-4B"
HF_MODEL_NAME = "uzh-echist-org/Ranke-4B-1913"
DATASET = "uzh-echist-org/ranke-1913-sft-dataset"

N_EXAMPLES = 512
LAYERS = (13, 16)
TOKEN_WINDOW = 5
STEERING_COEFFICIENT = 1.0
MAX_NEW_TOKENS = 64
EXAMPLE_ROWS = (0, 7, 42, 100)
DTYPE = torch.bfloat16
ARTIFACT_PATH = Path("artifacts/chat_vectors.pt")


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


def compare_prompt(prompt):
    print("Prompt:")
    print_wrapped(prompt)

    print("\nBase generation:")
    base_response = model.generate(
        prompt,
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=False,
    )
    print_wrapped(base_response)

    print("\nSteered generation:")
    with chat_vector.steer(model):
        steered_response = model.generate(
            prompt,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
        )
    print_wrapped(steered_response)


if __name__ == "__main__":
    torch.set_grad_enabled(False)

    model = load_ranke()

    rows = [
        row
        for row in load_dataset(DATASET, split="train")
        if "Correct answer:" not in row["formatted_response"]
        and "Reformatted response:" not in row["formatted_response"]
        and not row["formatted_response"].startswith(("Input question:", "Question:"))
    ][:N_EXAMPLES]
    assert len(rows) == N_EXAMPLES
    positive = [row["chat_question"] + " " + row["formatted_response"] for row in rows]
    negative = [row["chat_question"] + " " + row["response"] for row in rows]

    steering = ContrastiveSteering(
        model,
        token_indices=slice(-TOKEN_WINDOW, None),
        selected_layers=LAYERS,
    )
    vectors = steering.fit(positive, negative)

    ARTIFACT_PATH.parent.mkdir(exist_ok=True)
    torch.save(
        [
            {
                "value": vector.value.cpu(),
                "src_layer": vector.src_layer,
                "method_name": vector.method_name,
            }
            for vector in vectors
        ],
        ARTIFACT_PATH,
    )
    print(f"Saved chat vectors to {ARTIFACT_PATH}")

    chat_vector = vectors[0] * STEERING_COEFFICIENT
    for vector in vectors[1:]:
        chat_vector = chat_vector & (vector * STEERING_COEFFICIENT)

    for row_idx in EXAMPLE_ROWS:
        print(f"\n=== Row {row_idx} ===")
        compare_prompt(rows[row_idx]["chat_question"])
