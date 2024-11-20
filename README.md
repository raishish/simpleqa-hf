# SimpleQA Evaluation for Hugging Face Models

This repository provides a framework for evaluating open source models following the HuggingFace's AutoModelForCausalLM API on [OpenAI's SimpleQA dataset](https://openai.com/index/introducing-simpleqa/).

This is basically a trimmed version of [openai/simple-evals](https://github.com/openai/simple-evals) with modifications to support HF models.

## Features
- Efficient evaluation with separation of response generation and grading
- HF `accelerate` for loading models across all available resources

## SimpleQA Evaluation

```
>>> # Preview the dataset
>>> df = pd.read_csv("https://openaipublic.blob.core.windows.net/simple-evals/simple_qa_test_set.csv")
>>> df.head()
                                            metadata                                            problem               answer
0  {'topic': 'Science and technology', 'answer_ty...  Who received the IEEE Frank Rosenblatt Award i...        Michio Sugeno
1  {'topic': 'Science and technology', 'answer_ty...  Who was awarded the Oceanography Society's Jer...       Annick Bricaud
2  {'topic': 'Geography', 'answer_type': 'Place',...  What's the name of the women's liberal arts co...    Radcliffe College
3  {'topic': 'Sports', 'answer_type': 'Person', '...  In whose honor was the Leipzig 1877 tournament...      Adolf Anderssen
4  {'topic': 'Art', 'answer_type': 'Person', 'url...  According to Karl Küchler, what did Empress El...  Poet Henrich Heine.
```

Evaluation is split into two stages:

1. **Response generation**: generate responses to the questions in the dataset using the specified model.
2. **Response grading**: grade the generated responses using the specified grader model.

Since the grader model is usually a much larger model, it makes sense to separate the two stages to avoid running out of memory on low resource machines.

## Requirements

Ensure you have the required packages installed. You can install them using the following command:

```sh
pip install -r requirements.txt
```

## Usage

To evaluate a model on the entire dataset, use the following command:

```sh
HF_TOKEN=<HF_TOKEN> python simpleqa_eval_hf.py --generate_responses --model_name_hf <model_name_hf> --grade_responses --grader_model_name_hf <grader_model_name_hf> [options]
```

This runs both response generation and grading.

### Response Generation

To only generate responses for a model on the dataset questions, use:

```sh
HF_TOKEN=<HF_TOKEN> python simpleqa_eval_hf.py --generate_responses --model_name_hf <model_name_hf> [options]
```

Responses are saved in a JSON file in the `results` directory.

#### Response Generation Arguments

- `--generate_responses`: Flag to generate responses using the specified model.
- `--model_name_hf`: The name of the model to be evaluated (required if `--generate_responses` is True).
- `--system_message`: Optional system message to be included in the prompt.
- `--max_tokens`: Maximum number of tokens for the model's response (default: 1024).
- `--temperature`: Sampling temperature for the model (default: 0.7).
- `--device`: Device to run the model on.
- `--num_examples`: Number of examples to evaluate.

### Response Grading

To grade the responses, use the following command:

```sh
HF_TOKEN=<HF_TOKEN> python simpleqa_eval_hf.py --grade_responses --responses_file <responses_file> --grader_model_name_hf <grader_model_name_hf> [options]
```

#### Response Grading Arguments

- `--grade_responses`: Flag to grade the responses using the specified grader model.
- `--responses_file`: Path to the file containing the generated responses (if standalone response grading is done).
- `--grader_model_name_hf`: The name of the model used for grading the responses.
- `--grader_max_tokens`: Maximum number of tokens for the grader model's response (default: 1024).
- `--grader_temperature`: Sampling temperature for the grader model (default: 0.7).
- `--grader_device`: Device to run the grader model on.

### Example

To generate responses:

```sh
python simpleqa_eval_hf.py --generate_responses --model_name_hf google/gemma-2b-it --num_examples 100
```

To grade the above generated responses:

```sh
python simpleqa_eval_hf.py --grade_responses --responses_file results/simpleqa_gemma-2b-it_100_responses.json --grader_model_name_hf tiiuae/falcon-180B
```

## Output

The script generates an HTML report and a JSON file with the evaluation metrics. The files are saved in the `results` directory with names based on the model and grader model used.

## License

This project is licensed under the MIT License. See the [LICENSE](../LICENSE) file for details.