# SimpleQA Evaluation for Hugging Face Models

This repository provides a framework for evaluating open source models following the HuggingFace's AutoModelForCausalLM API on [OpenAI's SimpleQA dataset](https://openai.com/index/introducing-simpleqa/).

This is basically a trimmed version of [openai/simple-evals](https://github.com/openai/simple-evals) with modifications to support HF models.

## Requirements

Ensure you have the required packages installed. You can install them using the following command:

```sh
pip install -r requirements.txt
```

## Usage

To run the main script, use the following command:

```sh
python simpleqa_eval_hf.py --model <model_name> --grader_model <grader_model_name> [options]
```

### Arguments

- `--model`: The name of the model to be evaluated (required).
- `--grader_model`: The name of the model used for grading the responses (required).
- `--system_message`: Optional system message to be included in the prompt.
- `--max_tokens`: Maximum number of tokens for the model's response (default: 1024).
- `--temperature`: Sampling temperature for the model (default: 0.7).
- `--device`: Device to run the model on (default: `cuda:0`).
- `--grader_max_tokens`: Maximum number of tokens for the grader model's response (default: 1024).
- `--grader_temperature`: Sampling temperature for the grader model (default: 0.7).
- `--grader_device`: Device to run the grader model on (default: `cuda:1`).
- `--num_examples`: Number of examples to evaluate (default: None).

### Example

```sh
python simpleqa_eval_hf.py --model tiiuae/falcon-rw-1b --grader_model tiiuae/falcon-180B --num_examples 100
```

## Output

The script generates an HTML report and a JSON file with the evaluation metrics. The files are saved in the `/tmp` directory with names based on the model and grader model used.

## License

This project is licensed under the MIT License. See the [LICENSE](../LICENSE) file for details.