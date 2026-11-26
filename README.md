# fmr-nanodegree-agenticai
Nanodegree Course on Agentic AI

# Applied AI Projects

This repository contains two projects: an AWS workflow for training and deploying a dog-breed classifier, and a multimodal content moderation app built with Gemini.

| Project | Folder | Technologies |
| --- | --- | --- |
| AWS Dog Breed Classification | [aws-dog-breed-classifier/](./aws-dog-breed-classifier/) | PyTorch, ResNet50, SageMaker, S3, Lambda, EC2 |
| ACME Multimodal Content Moderation | [acme-multimodal-moderation/](./acme-multimodal-moderation/) | Gemini, Pydantic AI, Gradio, Phoenix, `uv` |

## AWS Dog Breed Classification

I used a pretrained ResNet50 model to build a classifier for 133 dog breeds. The project uses Amazon S3 for data storage, SageMaker for managed training and model hosting, and AWS Lambda to invoke the inference endpoint. It also includes a separate PyTorch training run on EC2.

### Project Files

- `train_and_deploy-solution.ipynb` walks through SageMaker training and deployment.
- `train_dog_classifier.py` runs standalone PyTorch training on EC2.
- `writeup.pdf` documents the deployment and analyzes IAM permissions, Lambda provisioned concurrency, SageMaker endpoint auto-scaling, and EC2 instance selection.

### Run the Project

Open the notebook in a SageMaker environment, configure your AWS region, S3 paths, and execution role, then run the training and deployment cells.

For the standalone EC2 run, install the Python dependencies and place the dataset in `dogImages/` with `train/`, `valid/`, and `test/` subdirectories. Run these commands from the project folder:

```bash
mkdir -p TrainedModels
python train_dog_classifier.py
```

The SageMaker endpoint reached `InService`, and the EC2 script saved a model artifact to `TrainedModels/model.pth`. The EC2 run used the CPU because the script did not move the model or training tensors to CUDA. The standalone script currently points its training loader at the test directory, so correct that path before measuring held-out accuracy.

Delete the SageMaker endpoint and terminate the EC2 instance when you finish testing.

## ACME Multimodal Content Moderation

ACME is a Gradio app that uses Gemini and Pydantic AI to moderate text, images, audio, and video. It routes content to an agent for the corresponding input type and returns a structured moderation decision. When content is flagged, the interface displays the rationale. Phoenix traces show how requests move through the app.

### Project Files

- `gradio_app.py` provides the user interface and connects it to the moderation agents.
- `agents/` contains the agents for each input type.
- `tests/` covers moderation schemas, agents, and application wiring.
- `evals/` contains cases for evaluating model behavior.

### Run the Project

From the ACME project folder, run `uv sync`. Create a local `.env` file with the following values:

```dotenv
GEMINI_API_KEY=your_gemini_api_key
USER_API_KEY=your_local_user_api_key
DEFAULT_GOOGLE_MODEL=your_configured_gemini_model
```

Start the app and open the local Gradio URL printed in the terminal:

```bash
uv run gradio_app.py
```

Run the automated tests and the text moderation evaluation:

```bash
uv run pytest tests/ -vv --ignore=tests/test_env_setup.py
uv run evals/text/test_cases.py
```

The tests check application wiring with stand-in models. The evaluation calls the configured Gemini model and compares its decisions against expected outcomes.