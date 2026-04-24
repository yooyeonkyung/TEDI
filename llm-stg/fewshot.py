import argparse
import os
import random
import pandas as pd
import torch
from tqdm import tqdm
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    AutoModelForCausalLM,
)

STYLE_DESC = {
    "star1": "very negative review style",
    "star5": "very positive review style",
}


def infer_styles_from_filename(path: str):
    name = os.path.basename(path).lower()
    if "test1" in name:
        return "star1", "star5"
    elif "test5" in name:
        return "star5", "star1"
    else:
        raise ValueError(
            f"Cannot infer source/target style from filename: {path}. "
            f"Expected filename containing 'test1' or 'test5'."
        )


def infer_domain_from_filename(path: str):
    name = os.path.basename(path).lower()
    if "amazon" in name:
        return "amazon"
    elif "yelp" in name:
        return "yelp"
    return "unknown"


def infer_source_style_from_row(row):
    if "star1" in row and float(row["star1"]) == 1.0:
        return "star1"
    if "star5" in row and float(row["star5"]) == 1.0:
        return "star5"
    return None


def load_and_filter_train(train_csv):
    df = pd.read_csv(train_csv).copy()
    df["source_style"] = df.apply(infer_source_style_from_row, axis=1)
    df = df[df["source_style"].isin(["star1", "star5"])].copy()
    return df


def sample_exemplars(train_df, style, k, seed):
    candidates = train_df[train_df["source_style"] == style]
    if len(candidates) == 0:
        return []

    if len(candidates) <= k:
        return [row for _, row in candidates.iterrows()]

    rnd = random.Random(seed)
    idxs = list(candidates.index)
    sampled = rnd.sample(idxs, k)
    return [row for _, row in candidates.loc[sampled].iterrows()]


def build_few_shot_prompt(text, source_style, target_style, domain, source_examples, target_examples):
    domain_str = "Amazon review" if domain == "amazon" else "Yelp review" if domain == "yelp" else "review"

    header = (
        "You are a text style transfer assistant.\n"
        f"Rewrite the input {domain_str} into the target review style while preserving the original meaning.\n"
        "Use the examples below to understand the tone of each style.\n\n"
    )

    examples = []
    for ex in source_examples:
        examples.append(
            f"Example of current style ({STYLE_DESC[source_style]}):\n"
            f"Text: {str(ex['text'])}\n"
        )

    for ex in target_examples:
        examples.append(
            f"Example of target style ({STYLE_DESC[target_style]}):\n"
            f"Text: {str(ex['text'])}\n"
        )

    query = (
        "\nNow rewrite the following review.\n"
        f"Current style: {STYLE_DESC[source_style]}\n"
        f"Target style: {STYLE_DESC[target_style]}\n"
        f"Input: {text}\n"
        "Output:"
    )

    return header + "\n".join(examples) + query


def load_model_and_tokenizer(model_name, model_type, device):
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    if model_type == "seq2seq":
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    elif model_type == "causal":
        model = AutoModelForCausalLM.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    model.to(device)
    model.eval()
    return model, tokenizer


@torch.no_grad()
def generate_text(model, tokenizer, prompt, model_type, device, max_input_length=2048, max_new_tokens=128):
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=max_input_length,
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id,
    )

    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

    if model_type == "causal" and decoded.startswith(prompt):
        decoded = decoded[len(prompt):].strip()

    decoded = decoded.split("\n")[0].strip()
    return decoded


def process_file(input_csv, train_csv, output_dir, model, tokenizer, model_type, device, k_shot, seed, max_new_tokens):
    df = pd.read_csv(input_csv)
    train_df = load_and_filter_train(train_csv)

    source_style, target_style = infer_styles_from_filename(input_csv)
    domain = infer_domain_from_filename(input_csv)

    source_examples = sample_exemplars(train_df, source_style, k_shot, seed)
    target_examples = sample_exemplars(train_df, target_style, k_shot, seed + 100)

    predictions = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc=os.path.basename(input_csv)):
        text = str(row["text"])

        prompt = build_few_shot_prompt(
            text=text,
            source_style=source_style,
            target_style=target_style,
            domain=domain,
            source_examples=source_examples,
            target_examples=target_examples,
        )

        prediction = generate_text(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            model_type=model_type,
            device=device,
            max_new_tokens=max_new_tokens,
        )

        predictions.append({"gen": prediction})

    out_df = pd.DataFrame(predictions)
    os.makedirs(output_dir, exist_ok=True)

    base = os.path.basename(input_csv).replace(".csv", "")
    output_csv = os.path.join(output_dir, f"{base}_generated.csv")
    out_df.to_csv(output_csv, index=False)
    print(f"Saved: {output_csv}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--amazon_train_csv", type=str, required=True)
    parser.add_argument("--yelp_train_csv", type=str, required=True)
    parser.add_argument("--input_csvs", type=str, nargs="+", required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--model_type", type=str, choices=["seq2seq", "causal"], required=True)
    parser.add_argument("--gpu_id", type=int, default=None, help="GPU id to use, e.g. 0")
    parser.add_argument("--k_shot", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    args = parser.parse_args()

    if args.gpu_id is not None and torch.cuda.is_available():
        device = torch.device(f"cuda:{args.gpu_id}")
    else:
        device = torch.device("cpu")

    print(f"Using device: {device}")

    model, tokenizer = load_model_and_tokenizer(
        model_name=args.model_name,
        model_type=args.model_type,
        device=device,
    )
    model.to(device)

    for input_csv in args.input_csvs:
        domain = infer_domain_from_filename(input_csv)
        if domain == "amazon":
            train_csv = args.amazon_train_csv
        elif domain == "yelp":
            train_csv = args.yelp_train_csv
        else:
            raise ValueError(f"Unknown domain for file: {input_csv}")

        process_file(
            input_csv=f"/home/ykyoo/yeonk/tedi/data/real/{input_csv}",
            train_csv=f"/home/ykyoo/yeonk/tedi/data/real/{train_csv}",
            output_dir=args.output_dir,
            model=model,
            tokenizer=tokenizer,
            model_type=args.model_type,
            device=device,
            k_shot=args.k_shot,
            seed=args.seed,
            max_new_tokens=args.max_new_tokens,
        )


if __name__ == "__main__":
    main()