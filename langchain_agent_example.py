import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.agents import AgentType, initialize_agent, load_tools

# Load environment variables
load_dotenv()


def main():
    # Initialize the LLM
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

    # Load tools for the agent
    tools = load_tools(["llm-math", "wikipedia"], llm=llm)

    # Initialize the agent
    agent = initialize_agent(
        tools,
        llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,
        handle_parsing_errors=True,
    )

    # Run the agent with different queries
    queries = [
        "What is the square root of 256?",
        "Who was the first person to walk on the moon and what year did it happen?",
        "If I have 5 apples and give 2 to my friend, then buy 3 more, how many do I have? Calculate step by step.",
    ]

    for query in queries:
        print(f"\nQuery: {query}")
        try:
            response = agent.invoke({"input": query})
            print(f"Response: {response['output']}")
        except Exception as e:
            print(f"Error: {e}")


if __name__ == "__main__":
    main()
