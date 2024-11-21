from hf_chat_completion_sampler import HFChatCompletionSampler
from simpleqa_eval import SimpleQAEval
import torch
import json
import os
from pathlib import Path
from argparse import ArgumentParser
import common


def main():
    parser = ArgumentParser()
    # Response generation
    parser.add_argument("--generate_responses", action="store_true", default=False)
    parser.add_argument("--model_name_hf", type=str)
    parser.add_argument("--model_dir", type=str, default=None)
    parser.add_argument("--system_message", type=str, default=None)
    parser.add_argument("--max_tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--device", type=str)
    parser.add_argument("--num_examples", type=int, default=None)

    # Response grading
    parser.add_argument("--grade_responses", action="store_true", default=False)
    parser.add_argument("--responses_file", type=Path)
    parser.add_argument("--grader_model_name_hf", type=str)
    parser.add_argument("--grader_model_dir", type=str, default=None)
    parser.add_argument("--grader_max_tokens", type=int, default=1024)
    parser.add_argument("--grader_temperature", type=float, default=0.7)
    parser.add_argument("--grader_device", type=str)

    parser.add_argument("--results_dir", type=Path, default=Path("results"))
    args = parser.parse_args()

    if not args.generate_responses and not args.grade_responses:
        raise ValueError("Either --generate_responses or --grade_responses must be True")

    args.results_dir.mkdir(exist_ok=True)  # create results dir
    responses = None

    if args.generate_responses:
        assert args.model_name_hf or args.model_dir, \
            "model_name_hf or model_dir must be provided if --generate_responses is True"

        model = HFChatCompletionSampler(
            model=args.model_name_hf,
            model_dir=args.model_dir,
            API_TOKEN=os.environ.get("HF_TOKEN", None),
            system_message=args.system_message,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            device=args.device
        )

        print("Generating responses...")
        responses = SimpleQAEval.generate_responses(model, num_examples=args.num_examples)

        print("Responses generated, writing to file...")
        model_name = args.model_name_hf.split("/")[-1]
        if args.num_examples:
            responses_file = f"simpleqa_{model_name}_{args.num_examples}_responses.json"
        else:
            responses_file = f"simpleqa_{model_name}_responses.json"
        responses_file = args.results_dir / responses_file

        with open(responses_file, "w") as fh:
            fh.write(json.dumps(responses, indent=4))
        print(f"Model responses written to {responses_file}")

        # Removing model from GPU memory
        model.model = model.model.cpu()
        del model
        torch.cuda.empty_cache()
        print(f"Model {args.model_name_hf} removed from GPU memory")

    if args.grade_responses:
        assert responses or args.responses_file, "responses_file must be provided if --grade_responses is True"
        assert args.grader_model_name_hf or args.grader_model_dir, \
            "grader_model_name_hf or grader_model_dir must be provided if --grade_responses is True"

        if not responses:
            responses = json.load(open(args.responses_file))
            model_name = args.responses_file.name.split("/")[-1].split("_")[1]
            num_examples = args.responses_file.name.split("/")[-1].split("_")[-2]
            try:
                num_examples = int(num_examples)
            except ValueError:
                num_examples = None  # contains all examples

        # System message for the grader model is defined in simpleqa_eval.py
        grader_model = HFChatCompletionSampler(
            model=args.grader_model_name_hf,
            model_dir=args.grader_model_dir,
            API_TOKEN=os.environ.get("HF_TOKEN", None),
            max_tokens=args.grader_max_tokens,
            temperature=args.grader_temperature,
            device=args.grader_device
        )

        # Evaluate responses
        print("Evaluating responses...")
        eval_result = SimpleQAEval.evaluate(grader_model, responses)

        # Dump results to file
        grader_model_name = args.grader_model_name_hf.split("/")[-1]
        if args.num_examples:
            html_file_name = Path(f"simpleqa_{model_name}_{grader_model_name}_{num_examples}.html")
        else:
            html_file_name = Path(f"simpleqa_{model_name}_{grader_model_name}.html")
        report_filename = args.results_dir / html_file_name

        print(f"Writing report to {report_filename}")
        with open(report_filename, "w") as fh:
            fh.write(common.make_report(eval_result))

        metrics = eval_result.metrics | {"score": eval_result.score}
        print(metrics)
        if args.num_examples:
            json_file_name = Path(f"simpleqa_{model_name}_{grader_model_name}_{num_examples}.json")
        else:
            json_file_name = Path(f"simpleqa_{model_name}_{grader_model_name}.json")
        result_filename = args.results_dir / json_file_name
        with open(result_filename, "w") as f:
            f.write(json.dumps(metrics, indent=4))
        print(f"Writing results to {result_filename}")

        # Print results
        print("\nResults:")
        print(f"Accuracy (given attempted): {eval_result.score:.3f}")
        print("\nDetailed metrics:")
        for k, v in eval_result.metrics.items():
            print(f"{k}: {v:.3f}")

        # Removing grader model from GPU memory
        grader_model.model = grader_model.model.cpu()
        del grader_model
        torch.cuda.empty_cache()
        print(f"Grader model {args.grader_model_name_hf} removed from GPU memory")


if __name__ == "__main__":
    main()
