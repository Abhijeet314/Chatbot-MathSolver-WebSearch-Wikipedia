# given any mathematical problem bot will be able to solve or any info will be solved with wikipedia
import streamlit as st
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from langchain.chains import LLMMathChain, LLMChain
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_community.tools import Tool, DuckDuckGoSearchRun
from langchain.agents import initialize_agent, AgentType
from langchain.callbacks import StreamlitCallbackHandler

st.title("Chatbot")

# sidebar
groq_api_key = st.sidebar.text_input("Enter your groq api key", type="password")

if not groq_api_key:
    st.info("please provide the groq api key")
    st.stop()

llm = ChatGroq(groq_api_key=groq_api_key, model="Gemma2-9b-It")

prompt_template = """
You are an agent who is required to solve math questions provided to you and share the detailed explanation
of how you solved it step by step in bullet points, Also if other questions which are not related to math are provided still you need to answer them with your best response
Question: {question}
Answer:
"""

prompt = PromptTemplate(input_variables=['question'], template=prompt_template)

# creating my tools
wiki_wrap = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=300)
wiki_tool = Tool(
    name="wikipedia",
    func=wiki_wrap.run,
    description="Searching from wikipedia about the queries"
)

math_chain = LLMMathChain.from_llm(llm=llm)
math_tool = Tool(
    name="Math tool",
    func=math_chain.run,
    description="Math solver"
)

search = DuckDuckGoSearchRun(name="search1")

# creating my chain
chain = LLMChain(llm=llm, prompt=prompt)

# converted my chain to a tool
reasoning_tool = Tool(
    name="reasoning",
    func=chain.run,
    description="reasoning tool to provide answers"
)

# combining tools with chain through agents
agent = initialize_agent(
    tools=[wiki_tool, math_tool,search, reasoning_tool],
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=False
)

# to manage state
if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "assistant", "content":"Hi I am a Chatbot who can solve math problems, How may I help?"}
    ]

for msg in st.session_state.messages:
    st.chat_message(msg['role']).write(msg['content'])

user_query = st.chat_input(placeholder="Ask you query")

if user_query:
    st.session_state.messages.append({"role": "user", "content": user_query})
    st.chat_message("user").write(user_query)

    with st.chat_message("assistant"):
        st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)
        response = agent.run(st.session_state.messages, callbacks=[st_cb])
        st.session_state.messages.append({"role": "assistant", "content": response})
        st.success(response)
