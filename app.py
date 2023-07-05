import streamlit as st
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SequentialChain
from langchain.memory import ConversationBufferMemory
from langchain.utilities import WikipediaAPIWrapper
from api import openai_api_key

st.title('🦜️🔗 Medium GPT')
prompt = st.text_input('Enter the topic name:')

# Prompt Template
title_template = PromptTemplate(
    input_variables=['topic'],
    template='Generate a Medium Blog Title about {topic}'
)
content_template = PromptTemplate(
    input_variables=['title', 'wikipedia_research'],
    template='Generate a Medium Blog Post about this topic in less than 4 paragraphs: {title} ;while leveraging '
             'this wikipedia research: {wikipedia_research}'
)

# Memory
title_memory = ConversationBufferMemory(input_key='topic', memory_key='chat_history')
content_memory = ConversationBufferMemory(input_key='title', memory_key='chat_history')

# Large Language Model
llm = OpenAI(openai_api_key=openai_api_key, temperature=0.9)
title_chain = LLMChain(llm=llm, prompt=title_template, verbose=True, output_key='title', memory=title_memory)
content_chain = LLMChain(llm=llm, prompt=content_template, verbose=True, output_key='content', memory=content_memory)

wiki = WikipediaAPIWrapper()
# # Sequential Chain
# medium_sequential_chain = SequentialChain(chains=[title_chain, content_chain], input_variables=['topic'],
#                                           output_variables=['title', 'content'], verbose=True)

if prompt:
    title = title_chain.run(topic=prompt)
    wiki_research = wiki.run(prompt)
    content = content_chain.run(title=title, wikipedia_research=wiki_research)
    st.write('Title')
    st.write(title)
    st.write('Content')
    st.write(content)

    with st.expander('Title History'):
        st.info(title_memory.buffer)

    with st.expander('Content History'):
        st.info(content_memory.buffer)

    with st.expander('Wikipedia Research History'):
        st.info(wiki_research)