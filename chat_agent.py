from dotenv import load_dotenv
import re
from langchain_groq import ChatGroq
from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver
from langchain.agents.middleware import AgentMiddleware, ModelRetryMiddleware, ToolRetryMiddleware
from langchain_core.messages import ToolMessage

from knowledge_retrieval import search_company_docs
from database_retrieval import search_database

# from langchain_core.utils.uuid import uuid7

load_dotenv()

# USER_DATABASE = {
#     "USER_1": {'user_role': 'c-level'},
#     "USER_2": {'user_role': 'engineering'},
#     "USER_3": {'user_role': 'finance'},
#     "USER_4": {'user_role': 'marketing'}
# }

# def get_user_role(user_id):
#     if user_id in USER_DATABASE:
#         return USER_DATABASE[user_id]['user_role']
#     else:
#         return 'general'

class PhoneRedactionMiddleware(AgentMiddleware):
    def wrap_tool_call(self, request, handler):
        # Execute tool
        tool_message = handler(request)
        
        if isinstance(tool_message, ToolMessage) and isinstance(tool_message.content, str):
            # Strict pattern matching (+91 1234-5678)
            specific_pattern = r'\+91\s\d{4}-\d{4}'
            tool_message.content = re.sub(specific_pattern, "[REDACTED PHONE]", tool_message.content)
            
        return tool_message


AGENT_LLM = ChatGroq(model="qwen/qwen3.8-27b", temperature=0.2)

SYSTEM_PROMPT = """
You are a chatbot that answers questions related to a FinTech company.
You have access to two tools - search_company_docs and search_database, decide which to call on basis of question asked by user.
If the user has asked a complex question, divide it into simpler sub-questions you can use to make multiple tool calls if required.
The search_company docs is a RAG tool so you should keep the words close to user's query as it is, even if you split the question.
Always give source of information using the metadata like document name, page number.
If search_database tells you that user doesn't have authorization to access database, say : "You don't have authorization to access database"
Don't invent information, don't hallucinate.
Don't get into an infinite loop. Break off if you don't receive an answer and say : "I couldn't find the answer."
"""

CHECKPOINTER = InMemorySaver()

AGENT = create_agent(
    model = AGENT_LLM,
    tools = [search_company_docs, search_database],
    system_prompt = SYSTEM_PROMPT,
    middleware=[PhoneRedactionMiddleware(), ModelRetryMiddleware(max_retries=2), ToolRetryMiddleware(max_retries=2)],
    checkpointer=CHECKPOINTER
)

def chat(user_question:str, user_role:str, session_thread_id:str)->str:
    
    # user_role = get_user_role(current_user)

    config = {
        'configurable' : {
            'thread_id': session_thread_id,
            'user_role': user_role
        },
        'recursion_limit' : 10
    }

    result = AGENT.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": user_question
                }
            ]
        },
        config=config
    )
    return result['messages'][-1].content

if __name__=="__main__":
    pass

    # print("\n\n")
    # print("=========="*10)
    # print("Welcome to Information Assistant for FinSolve Technologies")
    # print("\n(Type 'exit' to close this application)")
    # print("----------"*10)

    # current_user = 'USER_1'
    # session_thread_id = str(uuid7())
    
    # while True:
    #     try:
    #         print("\nPlease ask your question : ")

    #         user_question = input()

    #         if user_question.lower() in ['exit']:
    #             break
                
    #         response = chat(user_question, current_user, session_thread_id)

    #         print(f"\nASSISTANT : \n{response}\n")
    #         print("----------"*10)

    #     except KeyboardInterrupt:
    #         break