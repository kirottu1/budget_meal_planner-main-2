[![Python 3.12.3](https://img.shields.io/badge/python-3.12.3-blue.svg)](https://www.python.org/downloads/release/python-3123/)
[![Poetry](https://img.shields.io/endpoint?url=https://python-poetry.org/badge/v0.json)](https://python-poetry.org/)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
![Coverage](coverage.svg)
[![Dependabot](https://badgen.net/badge/Dependabot/enabled/orange?icon=dependabot)](https://dependabot.com/)
[![Actions status](https://github.com/astral-sh/ruff/workflows/CI/badge.svg)](https://github.com/ShiNik/python-cicd-demo/actions)
[![image](https://img.shields.io/pypi/l/ruff.svg)](https://github.com/ShiNik/python-cicd-demo/blob/main/LICENSE)


## Step 1: Install python:
sudo apt-get install python3.12.3

## Step 2: install poetry:
curl -sSL https://install.python-poetry.org | python3 -

## Step 3: Set Up the Virtual Environment
1. Create a virtual environment by typing: `poetry install`
2. poetry run pre-commit install --hook-type commit-msg

### Step 4: Download Ollama and Choose a Model
1. Visit [Ollama](https://ollama.com/) to download Ollama.
2. Go to the [Ollama Library](https://ollama.com/library) to choose a model.

### Step 5:  Use Llama3
For using llama3, follow these steps:

1. Visit the [Llama3 Blog](https://ollama.com/blog/llama3) for usage instructions.
2. To download llama3, open a terminal and type: `ollama pull llama3:latest`.
   To download llava, open a terminal and type: `ollama pull llava:latest`.
   To download llama3.2, open a terminal and type: `ollama pull llama3.2:3b-instruct-fp16`.
3. To check the list of models, in terminal type: `ollama list`.
4. To run the model, type: `ollama run MODEL_NAME`.
5. If you would like to see the output of the model for debugging, you can run ollama as a server by typing: `ollama serve`.

# run pre-commit hook
by typing: `pre-commit run --all-files`

# coverage: Run pytest test coverage
```
poetry run coverage run && poetry run coverage report && poetry run coverage xml && poetry run coverage-badge -f -o coverage.svg
```

## cookbooks
Download the 2 cookbooks and copy them into project_root/data/recipes/
- [2022CookingAroundtheWorldCookbook.pdf](https://www.nutrition.va.gov/docs/UpdatedPatientEd/2022CookingAroundtheWorldCookbook.pdf)
- [cookbook.pdf](https://foodhero.org/sites/foodhero-prod/files/health-tools/cookbook.pdf)

##  Install hooks in the project
` pre-commit install --hook-type commit-msg`

## Run hooks manually on all files
- ` pre-commit run --all-files`
- ` pre-commit run <hook-id>`
- `pre-commit run --files file1.py file2.py`

## Start the Redpanda Cluster 
` docker compose up -d`
## Browse the Redpanda Console UI
` http://localhost:8080/ `

## Use command line to manage redpanda cluster
``` 
- docker exec -it redpanda-0  rpk cluster info
- docker exec -it redpanda-0  rpk topic create  topic-name -r 3
- docker exec -it redpanda-0  rpk topic describe topic-name 
- docker exec -it redpanda-0  rpk topic produce  topic-name
- docker exec -it redpanda-0  rpk topic consume  topic-name --num 1
```

### Run the Recipe and flyer service
```
PYTHONPATH=/home/shima/projects/budget_meal_planner/src poetry run python3 ./recommendation_app/app.py
PYTHONPATH=/home/shima/projects/budget_meal_planner/src poetry run python3  ./flyer_service/main.py
PYTHONPATH=/home/shima/projects/budget_meal_planner/src poetry run python3  ./recipe_service/main.py
```

useful links
https://www.kaggle.com/datasets/shuyangli94/food-com-recipes-and-user-interactions/code
https://www.kaggle.com/code/engyyyy/recipe-recommendation-system
https://www.kaggle.com/code/stevenadisantoso/food-recommendation-using-tfrs
https://www.youtube.com/watch?v=jeCYqCEhqd8

https://www.youtube.com/@fastandsimpledevelopment/videos
https://www.youtube.com/watch?v=V0cEMA_D8jw
https://drive.google.com/drive/folders/1A7xufWkxwzgt40rl-vaEwC0xk2sp8bvJ


Chat With Multiple PDF Documents With Langchain And Google Gemini Pro
https://www.youtube.com/watch?v=uus5eLz6smA&list=PLZoTAELRMXVORE4VF7WQ_fAl0L1Gljtar&index=16
https://github.com/krishnaik06/Complete-Langchain-Tutorials/blob/main/chatmultipledocuments/chatpdf1.py

Huggingface
https://medium.com/@scholarly360/langchain-huggingface-complete-guide-on-colab-dfafe04fe661

AI Agents RAG With LangGraph 
https://www.youtube.com/watch?v=N1FM-PcVXNA&ab_channel=KrishNaik
https://github.com/langchain-ai/langchain/discussions/23374

Kafka:
https://www.youtube.com/watch?v=c6GW9KalwvA&list=PLaNsxqNgctlO5wIfkJhnrXbdUlqxTNoZ5&index=1&ab_channel=SumanshuNankana

Free Vision model
https://huggingface.co/Qwen/Qwen-VL-Chat
https://huggingface.co/spaces/opencompass/open_vlm_leaderboard

# Implementing Large Multimodal Models (LMM) in few lines of code using Langchain and Ollama
https://medium.com/primastat/implementing-large-multimodal-models-lmm-in-few-lines-of-code-using-langchain-and-ollama-6c08b1c25fdd

# LCEL Runable
https://github.com/0xmerkle/lcel-deepdive-runnables/blob/main/LCEL_Deep_Dive_Part_2_Runnables.ipynb
https://github.com/pinecone-io/examples/blob/master/learn/generation/langchain/handbook/11-langchain-expression-language.ipynb
https://www.youtube.com/watch?v=yF9kGESAi3M&ab_channel=codewithbrandon
https://github.com/Coding-Crashkurse/LCEL-Deepdive/blob/main/lcel.ipynb
https://github.com/bhancockio/langchain-crash-course/tree/main

optionally send a multimodal message into a ChatPromptTemplate:
https://github.com/langchain-ai/langchain/discussions/23374

Hybrid search:
https://medium.com/etoai/hybrid-search-combining-bm25-and-semantic-search-for-better-results-with-lan-1358038fe7e6

Agentic RAG example:
RAG agent with self-correction
https://www.youtube.com/watch?v=bq1Plo2RhYI&ab_channel=LangChain
https://www.youtube.com/watch?v=-ROS6gfYIts&ab_channel=LangChain
https://www.youtube.com/watch?v=9dXp5q3OFdQ&ab_channel=LangChain
https://www.youtube.com/watch?v=QQAkXHRJcZg&ab_channel=PromptEngineering

cool idea(write a pipeline):
chain of responsibility and registry pattern
https://www.youtube.com/watch?v=KY8n96Erp5Q&ab_channel=DaveEbbelaar


LlamaOCR (great example to improve my ocr):
We can add the OCR as a tool for LLM model so we can make an agentic solution
https://www.youtube.com/watch?v=5vScHI8F_xo&ab_channel=SamWitteveen
https://colab.research.google.com/drive/11lGQDbeEhj9hxI9sGfPCv303fAFihHyF?usp=sharing
