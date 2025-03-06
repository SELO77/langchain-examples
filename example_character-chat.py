import os
from dotenv import load_dotenv
from langchain.chains import LLMChain
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate, ChatPromptTemplate, MessagesPlaceholder
from langchain.memory import ConversationBufferMemory
from typing import Optional, List, Dict

# Load environment variables from .env file
load_dotenv()


class ChatOpenRouter(ChatOpenAI):
    def __init__(
        self,
        model: str = "moonshotai/moonlight-16b-a3b-instruct:free",
        api_key: Optional[str] = None,
        base_url: str = "https://openrouter.ai/api/v1",
        **kwargs,
    ):
        api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        super().__init__(model=model, api_key=api_key, base_url=base_url, **kwargs)


class CharacterChat:
    def __init__(
        self,
        service_prompt: str,
        character_prompt: str,
        user_prompt: str,
        memory_k: int = 3,
    ):
        """
        Initialize the character chat system.

        Args:
            service_prompt: Top-level prompt defining the roleplay
            character_prompt: Prompt for the specific character
            user_prompt: Prompt about the user attributes
            memory_k: Number of previous conversations to include
        """
        self.service_prompt = service_prompt
        self.character_prompt = character_prompt
        self.user_prompt = user_prompt

        # Initialize the LLM
        self.llm = ChatOpenRouter(temperature=0.7)

        # Initialize memory to store conversation history
        self.memory = ConversationBufferMemory(
            memory_key="chat_history", return_messages=True
        )

        # Store the memory_k value for manual history management
        self.memory_k = memory_k

        # Create the chat prompt template
        self.prompt = ChatPromptTemplate.from_messages(
            [
                ("system", self._build_system_prompt()),
                MessagesPlaceholder(variable_name="chat_history"),
                ("human", "{input}"),
            ]
        )

        # Create the conversation chain
        self.chain = LLMChain(
            llm=self.llm, prompt=self.prompt, memory=self.memory, verbose=True
        )

    def _build_system_prompt(self) -> str:
        """Combine the three prompt components into a single system prompt."""
        return f"""
{self.service_prompt}

CHARACTER INFORMATION:
{self.character_prompt}

USER INFORMATION:
{self.user_prompt}
"""

    def chat(self, user_input: str) -> str:
        """Process user input and return the character's response."""
        response = self.chain.invoke({"input": user_input})

        # Manually limit the conversation history if needed
        if self.memory_k > 0 and hasattr(self.memory, "chat_memory"):
            messages = self.memory.chat_memory.messages
            if len(messages) > self.memory_k * 2:  # Each exchange has 2 messages
                # Keep only the most recent k exchanges
                self.memory.chat_memory.messages = messages[-(self.memory_k * 2) :]

        return response["text"]


def main():
    # Example prompts
    service_prompt = """
    You are participating in an immersive roleplay conversation. 
    Respond as the character described below, maintaining their personality, speech patterns, and worldview.
    Keep responses concise and in-character at all times.
    """

    # Sherlock Holmes character
    sherlock_prompt = """
    Name: Sherlock Holmes
    Worldview: Analytical, logical, and observant. Values reason above all else.
    Character Traits: Brilliant detective, socially awkward, blunt, occasionally arrogant, addictive personality.
    Speech Pattern: Precise, formal, uses deductive reasoning, often explains his thought process.
    
    Example Dialog:
    "Elementary, my dear Watson. The mud on his boots clearly indicates he was in the East End this morning."
    "When you have eliminated the impossible, whatever remains, however improbable, must be the truth."
    "The game is afoot!"
    """

    # Tony Stark character
    tony_stark_prompt = """
    Name: Tony Stark (Iron Man)
    Worldview: Futurist, technologist, reformed weapons manufacturer. Believes technology can solve most problems.
    Character Traits: Genius inventor, witty, sarcastic, narcissistic but with a heart of gold, struggles with PTSD.
    Speech Pattern: Fast-talking, uses pop culture references, nicknames people, makes jokes in serious situations.
    
    Example Dialog:
    "Sometimes you gotta run before you can walk."
    "I am Iron Man. The suit and I are one."
    "Genius, billionaire, playboy, philanthropist. That's me in four words."
    """

    # User prompts
    detective_user_prompt = """
    The user is a curious individual interested in mysteries and detective work.
    They may ask you about cases, your methods, or seek your help with puzzles.
    """

    tech_user_prompt = """
    The user is a tech enthusiast interested in futuristic technology and superhero adventures.
    They may ask about your suits, Stark Industries, or the Avengers.
    """

    # Create character chat instances
    sherlock_chat = CharacterChat(
        service_prompt=service_prompt,
        character_prompt=sherlock_prompt,
        user_prompt=detective_user_prompt,
    )

    tony_chat = CharacterChat(
        service_prompt=service_prompt,
        character_prompt=tony_stark_prompt,
        user_prompt=tech_user_prompt,
    )

    # Select character
    print("Select a character to chat with:")
    print("1. Sherlock Holmes")
    print("2. Tony Stark (Iron Man)")

    while True:
        choice = input("Enter 1 or 2: ")
        if choice == "1":
            active_chat = sherlock_chat
            character_name = "Sherlock Holmes"
            break
        elif choice == "2":
            active_chat = tony_chat
            character_name = "Tony Stark"
            break
        else:
            print("Invalid choice. Please enter 1 or 2.")

    # Example conversation
    print(f"\n{character_name} AI Character Chat")
    print("Type 'exit' to end the conversation")
    print("-" * 50)

    while True:
        user_input = input("\nYou: ")
        if user_input.lower() == "exit":
            break

        response = active_chat.chat(user_input)
        print(f"\n{character_name}: {response}")


if __name__ == "__main__":
    main()
