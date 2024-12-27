# Chatbot Application with Streamlit and LangChain

## Overview
This project demonstrates a chatbot application built using **Streamlit** and **LangChain**. The chatbot is capable of solving mathematical problems with step-by-step explanations, retrieving information from Wikipedia, and answering general queries. It combines multiple tools and models to provide robust responses.

## Features
- **Math Problem Solving**: Solve mathematical queries with detailed step-by-step explanations.
- **Wikipedia Integration**: Fetch concise and accurate information from Wikipedia.
- **General Reasoning**: Handle non-mathematical queries with reasoning capabilities.
- **Search Tool**: Use DuckDuckGo for additional web-based query resolution.

## Prerequisites
Before running the application, ensure you have the following:
- Python 3.8 or higher
- A **Groq API key** for accessing the ChatGroq LLM.

## Setup Instructions

### Step 1: Clone the Repository
```bash
git clone <repository_url>
cd <repository_directory>
```

### Step 2: Install Dependencies
Install the required Python packages using the `requirements.txt` file:
```bash
pip install -r requirements.txt
```

### Step 3: Obtain a Groq API Key
1. Visit the [Groq website](https://www.groq.com) to create an account.
2. Obtain your API key from the dashboard.

### Step 4: Run the Application
1. Launch the Streamlit app:
```bash
streamlit run app.py
```
2. Enter your Groq API key in the sidebar.

### Step 5: Interact with the Chatbot
- Type your queries in the input box provided in the chat interface.
- The chatbot will respond to math, Wikipedia, or general reasoning queries.

## Code Explanation

### 1. **Streamlit Interface**
- **Title and Sidebar**:
  ```python
  st.title("Chatbot")
  groq_api_key = st.sidebar.text_input("Enter your groq api key", type="password")
  ```
  Displays the application title and collects the Groq API key from the user.

- **Chat Interface**:
  ```python
  if "messages" not in st.session_state:
      st.session_state["messages"] = [
          {"role": "assistant", "content":"Hi I am a Chatbot who can solve math problems, How may I help?"}
      ]
  ```
  Maintains the chat history between the user and the assistant.

### 2. **Model Initialization**
- **ChatGroq LLM**:
  ```python
  llm = ChatGroq(groq_api_key=groq_api_key, model="Gemma2-9b-It")
  ```
  Initializes the ChatGroq model for handling queries.

- **Prompt Template**:
  ```python
  prompt_template = """..."""
  ```
  Defines the structure for responses, ensuring step-by-step explanations for math queries.

### 3. **Tools and Chains**
- **Wikipedia Tool**:
  ```python
  wiki_wrap = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=300)
  ```
  Fetches concise results from Wikipedia.

- **Math Tool**:
  ```python
  math_chain = LLMMathChain.from_llm(llm=llm)
  ```
  Provides solutions to mathematical problems.

- **Reasoning Tool**:
  ```python
  reasoning_tool = Tool(name="reasoning", func=chain.run, description="reasoning tool to provide answers")
  ```
  Handles general reasoning queries using an LLMChain.

- **Agent Initialization**:
  ```python
  agent = initialize_agent(tools=[wiki_tool, math_tool, search, reasoning_tool], ...)
  ```
  Combines all tools into a single agent for seamless query resolution.

### 4. **User Interaction**
- **Input Handling**:
  ```python
  user_query = st.chat_input(placeholder="Ask your query")
  ```
  Captures the user's query and appends it to the chat history.

- **Response Generation**:
  ```python
  response = agent.run(st.session_state.messages, callbacks=[st_cb])
  ```
  Generates a response using the agent and displays it in the chat interface.

## How to Integrate
1. **Extend Tools**: Add additional tools to handle more types of queries (e.g., custom APIs or datasets).
2. **Modify Prompts**: Adjust the prompt template to customize the chatbot's behavior.
3. **Deploy**: Deploy the application on platforms like Heroku, AWS, or Google Cloud for broader access.

## Requirements
Ensure the following packages are installed. The `requirements.txt` file includes:
- streamlit
- langchain
- langchain_community
- duckduckgo_search

Install them using:
```bash
pip install -r requirements.txt
```

## Contribution
Feel free to contribute to this project by submitting issues or pull requests. Ensure your code adheres to the project's coding standards and include appropriate documentation.

## License
This project is licensed under the MIT License. See the LICENSE file for details.

