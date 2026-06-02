import os
from typing import List, Dict, Any
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

class ChatbotEngine:
    def __init__(self):
        # We use gemini-2.5-flash by default as it is super fast and low-latency.
        # Alternatively, we can use gemini-2.5-flash or gemini-3.5-flash depending on availability.
        # It reads GEMINI_API_KEY automatically from the environment.
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            temperature=0.7,
            streaming=True
        )
        
        # In-memory session store: session_id -> list of Messages
        self.sessions: Dict[str, List[Any]] = {}

        # Set up a system prompt that guides the AI on how to handle multiple speakers
        self.system_instruction = (
            "You are a helpful, friendly, and highly intelligent voice assistant.\n"
            "You are participating in a conversation where multiple speakers may talk.\n"
            "The user inputs will be formatted as a transcribed dialogue list showing who spoke (e.g., 'SPEAKER_00: Hello', 'SPEAKER_01: Hi').\n"
            "Analyze the conversation, address the speakers correctly by their tag/name if appropriate, "
            "and reply naturally as a single voice assistant. Speak concisely and clearly, as your response "
            "will be read out loud to the users."
        )

        self.prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content=self.system_instruction),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{input_text}")
        ])

        self.chain = self.prompt | self.llm

    def get_session_history(self, session_id: str) -> List[Any]:
        if session_id not in self.sessions:
            self.sessions[session_id] = []
        return self.sessions[session_id]

    def clear_session(self, session_id: str):
        if session_id in self.sessions:
            self.sessions[session_id] = []

    def format_dialogue_input(self, aligned_dialogue: List[Dict[str, Any]]) -> str:
        """
        Formats a list of speaker turns into a single text block for the LLM.
        Example output:
        "SPEAKER_00: How do we start?
        SPEAKER_01: I'm not sure."
        """
        lines = []
        for turn in aligned_dialogue:
            speaker = turn.get("speaker", "Speaker")
            text = turn.get("text", "").strip()
            lines.append(f"{speaker}: {text}")
        return "\n".join(lines)

    async def get_response(self, session_id: str, aligned_dialogue: List[Dict[str, Any]]) -> str:
        """
        Processes the input dialogue, updates history, and returns the LLM's response.
        """
        # Format the dialogue block
        formatted_input = self.format_dialogue_input(aligned_dialogue)
        
        if not formatted_input.strip():
            return "I couldn't hear anyone speaking. Could you please repeat that?"

        # Retrieve conversation history
        history = self.get_session_history(session_id)
        
        # Invoke the chain
        response = await self.chain.ainvoke({
            "history": history,
            "input_text": formatted_input
        })
        
        response_text = response.content
        
        # Update history with the user turns and assistant response
        history.append(HumanMessage(content=formatted_input))
        history.append(AIMessage(content=response_text))
        
        # Prune history to keep last 20 messages to prevent token bloat
        if len(history) > 20:
            self.sessions[session_id] = history[-20:]
            
        return response_text
