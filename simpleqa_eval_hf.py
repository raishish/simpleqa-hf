from hf_chat_completion_sampler import HFChatCompletionSampler
from simpleqa_eval import SimpleQAEval
import torch
import json
import os
from argparse import ArgumentParser
import common


def main():
    parser = ArgumentParser()
    parser.add_argument("--model", type=str, required=True, default="tiiuae/falcon-rw-1b")
    parser.add_argument("--system_message", type=str, default=None)
    parser.add_argument("--max_tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--device", type=str, default=torch.device("cuda:0"))
    parser.add_argument("--grader_model", type=str, required=True, default="tiiuae/falcon-180B")
    parser.add_argument("--grader_max_tokens", type=int, default=1024)
    parser.add_argument("--grader_temperature", type=float, default=0.7)
    parser.add_argument("--grader_device", type=str, default=torch.device("cuda:1"))
    parser.add_argument("--num_examples", type=int | None, default=None)
    args = parser.parse_args()

    # Initialize components
    model = HFChatCompletionSampler(
        model=args.model,
        API_TOKEN=os.environ.get("HF_TOKEN", None),
        system_message=args.system_message,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        device=args.device
    )

    grader_model = HFChatCompletionSampler(
        model=args.grader_model,
        API_TOKEN=os.environ.get("HF_TOKEN", None),
        max_tokens=args.grader_max_tokens,
        temperature=args.grader_temperature,
        device=args.grader_device
    )

    evaluator = SimpleQAEval(
        grader_model=grader_model,
        num_examples=args.num_examples
    )

    # Generate responses
    print("Generating responses...")
    results = evaluator.generate_responses(model)

    # Evaluate responses
    print("Evaluating responses...")
    eval_result = evaluator.evaluate(results)

    # Dump results to file
    file_name = f"simpleqa_{args.model}_{args.grader_model}_{args.num_examples}"
    report_filename = f"/tmp/{file_name}.html"
    print(f"Writing report to {report_filename}")
    with open(report_filename, "w") as fh:
        fh.write(common.make_report(eval_result))

    metrics = eval_result.metrics | {"score": eval_result.score}
    print(metrics)
    result_filename = f"/tmp/{file_name}.json"
    with open(result_filename, "w") as f:
        f.write(json.dumps(metrics, indent=2))
    print(f"Writing results to {result_filename}")

    # Print results
    print("\nResults:")
    print(f"Accuracy (given attempted): {eval_result.score:.3f}")
    print("\nDetailed metrics:")
    for k, v in eval_result.metrics.items():
        print(f"{k}: {v:.3f}")


if __name__ == "__main__":
    main()
