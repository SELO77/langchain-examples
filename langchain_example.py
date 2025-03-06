import os
from dotenv import load_dotenv
from langchain.chains import LLMChain
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import SimpleSequentialChain
from typing import Optional

# Load environment variables from .env file
load_dotenv()


class ChatOpenRouter(ChatOpenAI):
    def __init__(
        self,
        # model: str = "qwen/qwq-32b:free", // 추론 모델
        model: str = "moonshotai/moonlight-16b-a3b-instruct:free",
        api_key: Optional[str] = None,
        base_url: str = "https://openrouter.ai/api/v1",
        **kwargs
    ):
        api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        super().__init__(model=model, api_key=api_key, base_url=base_url, **kwargs)


def main():
    # Initialize the LLM
    llm = ChatOpenRouter(
        temperature=0.7,
    )

    # Create a prompt template
    prompt_template = PromptTemplate(
        input_variables=["topic"], template="Write a short paragraph about {topic}."
    )

    # Create a chain
    chain = LLMChain(llm=llm, prompt=prompt_template)

    # Run the chain
    topic = "artificial intelligence"
    result = chain.invoke({"topic": topic})

    # print(f"Topic: {topic}")
    # print(f"Response: {result['text']}")

    # Example of a simple sequential chain
    # First chain generates a title
    first_prompt = PromptTemplate(
        input_variables=["topic"],
        template="Generate a creative title for an article about {topic}.",
    )
    first_chain = LLMChain(llm=llm, prompt=first_prompt)

    # Second chain generates content based on the title
    second_prompt = PromptTemplate(
        input_variables=["title"],
        template="Write the first paragraph of an article with the title: {title}",
    )
    second_chain = LLMChain(llm=llm, prompt=second_prompt)

    # Combine chains
    overall_chain = SimpleSequentialChain(
        chains=[first_chain, second_chain], verbose=True
    )

    # Run the sequential chain
    sequential_result = overall_chain.invoke({"input": topic})
    # print("\nSequential Chain Result:")
    # print(sequential_result)


if __name__ == "__main__":
    main()
