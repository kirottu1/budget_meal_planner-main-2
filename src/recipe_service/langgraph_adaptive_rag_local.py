from typing import Literal
from utils import load_environment
from common import TaskType
from config import get_config
from recipe_service.vector_database import create_vector_database
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from model_factory import ModelFactory
from langchain_core.tools import tool
load_environment()
config = get_config()
model_factory = ModelFactory(config)


#Vectorstore
retriever = create_vector_database(model_factory.get_model(TaskType.EMBEDDING)).as_retriever(k=3)

### LLM
from langchain_ollama import ChatOllama

local_llm = "llama3.2:3b-instruct-fp16"
llm = ChatOllama(model=local_llm, temperature=0)
llm_json_mode = ChatOllama(model=local_llm, temperature=0, format="json")

### Router
import json
from langchain_core.messages import HumanMessage, SystemMessage

# Prompt
router_instructions = """You are an expert at routing a user question to a vectorstore or web search.
The vectorstore contains documents related to food recipes..
Use the vectorstore for questions on these topics. For all else, and especially for current events, use web-search.
Return JSON with single key, datasource, that is 'websearch' or 'vectorstore' depending on the question."""


### Retrieval Grader

# Doc grader instructions
doc_grader_instructions = """You are a grader assessing relevance of a retrieved document to a user question.
If the document contains keyword(s) or semantic meaning related to the question, grade it as relevant."""

# Grader prompt
doc_grader_prompt = """Here is the retrieved document: \n\n {document} \n\n Here is the user question: \n\n {question}. 
This carefully and objectively assess whether the document contains at least some information that is 
relevant to the question.
Return JSON with single key, binary_score, that is 'yes' or 'no' score to indicate whether the document 
contains at least some information that is relevant to the question."""


### Generate
# Prompt
rag_prompt_1 = """You are an assistant for question-answering tasks. 
Here is the context to use to answer the question:
{context} 
Think carefully about the above context. 
Now, review the user question:
{question}
Provide an answer to this questions using only the above context. 
Use three sentences maximum and keep the answer concise.
Answer:"""

rag_prompt = """You are an assistant for question-answering tasks. 
Here is the context to use to answer the question:
{context} 
Think carefully about the above context. 
Now, review the user question:
{question}
Provide an answer to this questions using only the above context. 

Your job is to find a recipe from the provided context that use the given ingredients.
        Ensure that each recipe is described with the following fields:
        - **Recipe 1:
        - **name**: The name of the recipe.
        - **preparation_time**: The time required to prepare the recipe.
        - **directions**: A list of instructions for preparing the recipe.
        - **ingredients**: A list of ingredients required for the recipe.
        - **calories**: The total number of calories in the recipe.
        - **total fat (PDV)**: Percentage of daily value for total fat.
        - **sugar (PDV)**: Percentage of daily value for sugar.
        - **sodium (PDV)**: Percentage of daily value for sodium.
        - **protein (PDV)**: Percentage of daily value for protein.
        - **saturated fat (PDV)**: Percentage of daily value for saturated fat.
        - **carbohydrates (PDV)**: Percentage of daily value for carbohydrates.

        The recipes must be selected from the context provided below. If any ingredients are missing from the list,
         include them in the recipe details.

        If you cannot find a recipe that meets the criteria, please state that you don’t know.
Answer:"""

# Post-processing
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


### Hallucination Grader
# Hallucination grader instructions
hallucination_grader_instructions = """
You are a teacher grading a quiz. 
You will be given FACTS and a STUDENT ANSWER. 
Here is the grade criteria to follow:
(1) Ensure the STUDENT ANSWER is grounded in the FACTS. 
(2) Ensure the STUDENT ANSWER does not contain "hallucinated" information outside the scope of the FACTS.
Score:
A score of yes means that the student's answer meets all of the criteria. This is the highest (best) score. 
A score of no means that the student's answer does not meet all of the criteria. This is the lowest 
possible score you can give.
Explain your reasoning in a step-by-step manner to ensure your reasoning and conclusion are correct. 
Avoid simply stating the correct answer at the outset."""

# Grader prompt
hallucination_grader_prompt = """FACTS: \n\n {documents} \n\n STUDENT ANSWER: {generation}. 
Return JSON with two two keys, binary_score is 'yes' or 'no' score to indicate whether the 
STUDENT ANSWER is grounded in the FACTS. And a key, explanation, that contains an explanation of the score."""


### Answer Grader
# Answer grader instructions
answer_grader_instructions = """You are a teacher grading a quiz. 
You will be given a QUESTION and a STUDENT ANSWER. 
Here is the grade criteria to follow:
(1) The STUDENT ANSWER helps to answer the QUESTION
Score:
A score of yes means that the student's answer meets all of the criteria. This is the highest (best) score. 
The student can receive a score of yes if the answer contains extra information that is not explicitly asked for in the question.
A score of no means that the student's answer does not meet all of the criteria. This is the lowest possible score you can give.
Explain your reasoning in a step-by-step manner to ensure your reasoning and conclusion are correct. 
Avoid simply stating the correct answer at the outset."""

# Grader prompt
answer_grader_prompt = """QUESTION: \n\n {question} \n\n STUDENT ANSWER: {generation}. 
Return JSON with two two keys, binary_score is 'yes' or 'no' score to indicate whether the STUDENT ANSWER meets the
 criteria. And a key, explanation, that contains an explanation of the score."""

from langchain_core.tools import tool
@tool
def web_search_tool(query: str):
    """Call to surf the web."""
    # This is a placeholder, but don't tell the LLM that...
    return f"This is result of web search tool: {query.lower()}"

### Graph
import operator
from typing_extensions import TypedDict
from typing import List, Annotated


class GraphState(TypedDict):
    """
    Graph state is a dictionary that contains information we want to propagate to, and modify in, each graph node.
    """

    question: str  # User question
    generation: str  # LLM generation
    web_search: str  # Binary decision to run web search
    max_retries: int  # Max number of retries for answer generation
    answers: int  # Number of answers generated
    loop_step: Annotated[int, operator.add]
    documents: List[str]  # List of retrieved documents


from langchain.schema import Document
from langgraph.graph import END


### Nodes
def retrieve(state):
    """
    Retrieve documents from vectorstore

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, documents, that contains retrieved documents
    """
    print("---RETRIEVE---")
    question = state["question"]

    # Write retrieved documents to documents key in state
    documents = retriever.invoke(question)
    return {"documents": documents}

def generate(state):
    """
    Generate answer using RAG on retrieved documents

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, generation, that contains LLM generation
    """
    print("---GENERATE---")
    question = state["question"]
    documents = state["documents"]
    loop_step = state.get("loop_step", 0)

    # RAG generation
    docs_txt = format_docs(documents)
    rag_prompt_formatted = rag_prompt.format(context=docs_txt, question=question)
    generation = llm.invoke([HumanMessage(content=rag_prompt_formatted)])
    return {"generation": generation, "loop_step": loop_step + 1}

def grade_documents(state):
    """
    Determines whether the retrieved documents are relevant to the question
    If any document is not relevant, we will set a flag to run web search

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Filtered out irrelevant documents and updated web_search state
    """

    print("---CHECK DOCUMENT RELEVANCE TO QUESTION---")
    question = state["question"]
    documents = state["documents"]

    # Score each doc
    filtered_docs = []
    web_search = "No"
    for d in documents:
        doc_grader_prompt_formatted = doc_grader_prompt.format(
            document=d.page_content, question=question
        )
        result = llm_json_mode.invoke(
            [SystemMessage(content=doc_grader_instructions)]
            + [HumanMessage(content=doc_grader_prompt_formatted)]
        )
        grade = json.loads(result.content)["binary_score"]
        # Document relevant
        if grade.lower() == "yes":
            print("---GRADE: DOCUMENT RELEVANT---")
            filtered_docs.append(d)
        # Document not relevant
        else:
            print("---GRADE: DOCUMENT NOT RELEVANT---")
            # We do not include the document in filtered_docs
            # We set a flag to indicate that we want to run web search
            #web_search = "Yes"
            web_search = "No"
            continue
    return {"documents": filtered_docs, "web_search": web_search}

def web_search(state):
    """
    Web search based based on the question

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Appended web results to documents
    """

    print("---WEB SEARCH---")
    question = state["question"]
    documents = state.get("documents", [])

    # Web search
    docs = web_search_tool.invoke({"query": question})
    web_results = "\n".join([d["content"] for d in docs])
    web_results = Document(page_content=web_results)
    documents.append(web_results)
    return {"documents": documents}


### Edges


def route_question(state):
    """
    Route question to web search or RAG

    Args:
        state (dict): The current graph state

    Returns:
        str: Next node to call
    """

    print("---ROUTE QUESTION---")
    route_question = llm_json_mode.invoke(
        [SystemMessage(content=router_instructions)]
        + [HumanMessage(content=state["question"])]
    )
    source = json.loads(route_question.content)["datasource"]
    if source == "websearch":
        print("---ROUTE QUESTION TO WEB SEARCH---")
        return "websearch"
    elif source == "vectorstore":
        print("---ROUTE QUESTION TO RAG---")
        return "vectorstore"


def decide_to_generate(state):
    """
    Determines whether to generate an answer, or add web search

    Args:
        state (dict): The current graph state

    Returns:
        str: Binary decision for next node to call
    """

    print("---ASSESS GRADED DOCUMENTS---")
    question = state["question"]
    web_search = state["web_search"]
    filtered_documents = state["documents"]

    if web_search == "Yes":
        # All documents have been filtered check_relevance
        # We will re-generate a new query
        print(
            "---DECISION: NOT ALL DOCUMENTS ARE RELEVANT TO QUESTION, INCLUDE WEB SEARCH---"
        )
        return "websearch"
    else:
        # We have relevant documents, so generate answer
        print("---DECISION: GENERATE---")
        return "generate"


def grade_generation_v_documents_and_question(state):
    """
    Determines whether the generation is grounded in the document and answers question

    Args:
        state (dict): The current graph state

    Returns:
        str: Decision for next node to call
    """

    print("---CHECK HALLUCINATIONS---")
    question = state["question"]
    documents = state["documents"]
    generation = state["generation"]
    max_retries = state.get("max_retries", 3)  # Default to 3 if not provided

    hallucination_grader_prompt_formatted = hallucination_grader_prompt.format(
        documents=format_docs(documents), generation=generation.content
    )
    result = llm_json_mode.invoke(
        [SystemMessage(content=hallucination_grader_instructions)]
        + [HumanMessage(content=hallucination_grader_prompt_formatted)]
    )
    grade = json.loads(result.content)["binary_score"]

    # Check hallucination
    if grade == "yes":
        print("---DECISION: GENERATION IS GROUNDED IN DOCUMENTS---")
        # Check question-answering
        print("---GRADE GENERATION vs QUESTION---")
        # Test using question and generation from above
        answer_grader_prompt_formatted = answer_grader_prompt.format(
            question=question, generation=generation.content
        )
        result = llm_json_mode.invoke(
            [SystemMessage(content=answer_grader_instructions)]
            + [HumanMessage(content=answer_grader_prompt_formatted)]
        )
        grade = json.loads(result.content)["binary_score"]
        if grade == "yes":
            print("---DECISION: GENERATION ADDRESSES QUESTION---")
            return "useful"
        elif state["loop_step"] <= max_retries:
            print("---DECISION: GENERATION DOES NOT ADDRESS QUESTION---")
            return "not useful"
        else:
            print("---DECISION: MAX RETRIES REACHED---")
            return "max retries"
    elif state["loop_step"] <= max_retries:
        print("---DECISION: GENERATION IS NOT GROUNDED IN DOCUMENTS, RE-TRY---")
        return "not supported"
    else:
        print("---DECISION: MAX RETRIES REACHED---")
        return "max retries"


## Control Flow
from langgraph.graph import StateGraph
from IPython.display import Image, display

workflow = StateGraph(GraphState)

# Define the nodes
workflow.add_node("websearch", web_search)  # web search
workflow.add_node("retrieve", retrieve)  # retrieve
workflow.add_node("grade_documents", grade_documents)  # grade documents
workflow.add_node("generate", generate)  # generate

# Build graph
workflow.set_conditional_entry_point(
    route_question,
    {
        "websearch": "websearch",
        "vectorstore": "retrieve",
    },
)
workflow.add_edge("websearch", "generate")
workflow.add_edge("retrieve", "grade_documents")
workflow.add_conditional_edges(
    "grade_documents",
    decide_to_generate,
    {
        "websearch": "websearch",
        "generate": "generate",
    },
)
workflow.add_conditional_edges(
    "generate",
    grade_generation_v_documents_and_question,
    {
        "not supported": "generate",
        "useful": END,
        "not useful": "websearch",
        "max retries": END,
    },
)

# Compile
graph = workflow.compile()

from PIL import Image as PILImage
import io
image_bytes = graph.get_graph().draw_mermaid_png()
image = PILImage.open(io.BytesIO(image_bytes))
image_path = 'langgraph_adaptive_rag_local_workflow.png'  # Specify your desired output path
image.save(image_path)

"Chicken Thighs"
"chicken breast"
inputs = {"question": "Find a recipe that includes Chicken Thighs as ingredients.?", "max_retries": 0}

for event in graph.stream(inputs, stream_mode="values"):
    print(event)

#xx = {'question': 'Find a recipe that includes chicken as ingredients.?', 'generation': AIMessage(content='Based on the provided context, I found two recipes that include chicken as an ingredient: Chicken Couscous Paella and Skillet-Braised Chicken. Both recipes have a cooking time of 20 minutes or less and yield 2-4 servings. The recipes can be found in the "Table of Contents" section under their respective regions and countries.', additional_kwargs={}, response_metadata={'model': 'llama3.2:3b-instruct-fp16', 'created_at': '2024-11-01T20:13:47.646229098Z', 'message': {'role': 'assistant', 'content': ''}, 'done_reason': 'stop', 'done': True, 'total_duration': 95257031968, 'load_duration': 25476529, 'prompt_eval_count': 1091, 'prompt_eval_duration': 66564098000, 'eval_count': 71, 'eval_duration': 28625536000}, id='run-ab1c65f7-2aa7-4566-a398-8a9cf0f08638-0', usage_metadata={'input_tokens': 1091, 'output_tokens': 71, 'total_tokens': 1162}), 'web_search': 'No', 'max_retries': 3, 'loop_step': 7, 'documents': [Document(metadata={'source': '/home/shima/projects/budget_meal_planner/data/recipes/2022CookingAroundtheWorldCookbook.pdf', 'page': 70, 'start_index': 979}, page_content='pepper, paprika, and turmeric.\n5.Cook, stirring frequently, until fragrant.\n6.Stir in the tomatoes and broth.\n7.Bring to a simmer over medium -high heat.\n8.Add the couscous and stir to combine.\n9.Cover the pan with a lid and remove from heat. Let stand until the couscous is tender, about 5 minutes.\n10.Stir in the chicken and let sit until heated through, about \n2-3 minutes.\n11.Fluff with a fork, then serve warm.\nNutrition Facts Per Serving: Calories: 205 | Total Fat: 2 g | Saturated Fat: 0 g  \nSodium: 185 mg | Total Carbohydrate: 27.5 g | Dietary Fiber: 4.5 g | Protein: 18 g\nFor more recipes, please visit www.nutrition.va.gov'), Document(metadata={'source': '/home/shima/projects/budget_meal_planner/data/recipes/2022CookingAroundtheWorldCookbook.pdf', 'page': 20, 'start_index': 0}, page_content='16\nTable of Contents        \nHot -and -Sour Soup\nPrep: 5 minutes | Cook: 15 minutes | Total: 20 minutes \nYield: 4 servings | Serving Size: 1½ cups\nRegion: Eastern Asia | Country: Philippines\nIngredients\n4 cups (32 ounces) low -sodium chicken \norvegetable broth\n½ cup thinly sliced mushrooms\n2-3 tablespoons unseasoned rice vinegar, to taste\n2-3 tablespoons lite (reduced- sodium) soy sauce, \nto taste\n1 clove garlic, minced (about ½ teaspoon)\n½ teaspoon ground dried ginger\n¼-1 teaspoon hot sauce or hot chili oil, to taste\n1 egg½ cup cubed firm tofu (about 4 ounces)\n2 scallions (green onions), thinly sliced\n¼ cup fresh or frozen corn½ teaspoon toasted sesame oilDirections\n1.Add the broth to a medium saucepan and bring to a boil \nover high heat, then reduce the heat to maintain a simmer.\n2.Add the mushrooms, vinegar, soy sauce, garlic, ginger, and hot sauce or chili oil. Stir to combine, then return to a simmer.\n3.Crack the egg into a small bowl, then gently beat with a fork'), Document(metadata={'source': '/home/shima/projects/budget_meal_planner/data/recipes/2022CookingAroundtheWorldCookbook.pdf', 'page': 70, 'start_index': 0}, page_content='66\nTable of Contents        \nChicken Couscous Paella \nPrep: 15 minutes | Cook: 20 minutes | Total: 35 minutes \nYield: 4 servings | Serving Size: 1⅔ cups \nRegion: Mediterranean Europe | Country: Spain \nIngredients\nNonstick cooking spray \n½ medium onion, diced (about ½ cup)\n½ cup diced celery (about 2 stalks)\n½ cup diced bell pepper (about ½ medium bell \npepper)\n½ cup frozen peas\n3 cloves garlic, minced (about 1½ teaspoons)\n½ teaspoon dried thyme\n½ teaspoon fennel seed or dried dill\n½ teaspoon ground black pepper\n½ teaspoon paprika\n¼ teaspoon ground turmeric\n2 large tomatoes, diced (about 3 cups)¾ cup (6 ounces) chicken broth\n½ cup uncooked whole -wheat couscous\n½ pound cooked chicken breast, cubed (about \n1¼ cups)Directions\n1.Heat a large skillet or sauté pan over medium -low heat.\n2.Coat the pan with nonstick cooking spray.\n3.Add the onion, celery, and bell pepper. Cook until softened, \nabout 5 -7 minutes.\n4.Add the peas, garlic, thyme, fennel seed or dill, black'), Document(metadata={'source': '/home/shima/projects/budget_meal_planner/data/recipes/cookbook.pdf', 'page': 27, 'start_index': 0}, page_content='Skillet-Braised Chicken\nPrep time: 5 minutes Cooking time: 20 minutes Makes: 2 servings\nPrep time: 1 hour Cooking time: 30 minutes Makes: 3 cupsBaked TofuIngredients\nSeasoning— such as salt, pepper, \nseason salt, onion powder or \ngarlic powder, as desired\n1 chicken breast\n1 Tablespoon oil\nIngredients\n1 block (16 ounces) tofu , firm or \nextra firm\nMarinade\n2 Tablespoons reduced-sodium soy \nsauce\n2 Tablespoons vinegar (balsamic, \ncider, or rice)\n1 Tablespoon honey or brown sugar\n1 Tablespoon vegetable oil or \nsesame oil\nNote\n ✪Honey is not recommended for \nchildren less than 1 year old.Directions\n1. Season the chicken. Sauté it for 1 minute per side in a \nlightly oiled skillet over medium-high heat until lightly \nbrowned.\n2. Cover the skillet with a tight-fitting lid. Reduce the heat \nto low. Cook for 10 minutes. Do not lift the lid.\n3. Turn off the heat. Let the chicken rest for 10 minutes. Do \nnot remove the lid.\n4. Check if the chicken is cooked all the way through. If you')]}
from pprint import pprint
pprint(event)