import openai
import tiktoken
import pandas as pd
import numpy as np
import streamlit as st


COMPLETIONS_MODEL = "gpt-3.5-turbo-0301"

person = st.selectbox(
    'Do you want to talk to a fictional or real person?',
    ('Fictional', 'Real'))

if person == "Fictional":
    option = st.selectbox(
        'Who do you want to talk to?',
        (" ",
        'Master Yoda from Star Wars',
        'Sponge Bob',
        'Tony Stark (Iron Man)',
        'Gandalf - Lord of the Rings',
        "Sherlock Holmes",
        "Harry Potter",
        "Atticus Finch - Harper Lee's To Kill a Mockingbird",
        "Darth Vader",
        "Holden Caulfield - J.D. Salinger's The Catcher in the Rye",
        "Frodo Baggins - The Lord of the Rings",
        "Tyrion Lannister - Game of Thrones)",
        "Elizabeth Bennet - Pride and Prejudice",
        "Katniss Everdeen - The Hunger Games",
        "Don Quixote",
        "Holden Caulfield - The Catcher in the Rye",
        "Jay Gatsby - F. Scott Fitzgerald's novel",
        "Hannibal Lecter",
        "Hercule Poirot - the detective in Agatha Christie's novels",
        "Lisbeth Salander - the hacker and vigilante from Stieg Larsson's Millennium series",
        "Willy Wonka - Charlie and the Chocolate Factory",
        "Hermione Granger",
        "Dracula - the vampire from Bram Stoker's novel of the same name"))

else:
    option = st.selectbox(
        'Who do you want to talk to?',
        (" ",
        "Albert Einstein",
        "Steve Jobs",
        "Oprah Winfrey",
        "Mahatma Gandhi",
        "Neil Armstrong",
        "William Shakespeare",
        "Martin Luther King Jr.",
        "Elon Musk",
        "Barack Obama",
        "Mother Teresa",
        "Albert Camus - philosopher and writer",
        "Princess Diana",
        "J.K. Rowling",
        "Stephen Hawking",
        "Nelson Mandela",
        "Malala Yousafzai - Nobel Peace Prize winner",
        "Socrates",
        "Emma Watson"))



st.title(f"Chat with {option}")

# The token limit for gpt-35-turbo is 4096 tokens.
# This limit includes the token count from both the prompt and completion.
MAX_CONTEXT_LEN = 800
SEPARATOR = "\n* "                  # Context separator contains 3 tokens
ENCODING = "cl100k_base"            # Encoding for text-embedding-ada-002

encoding = tiktoken.get_encoding(ENCODING)
separator_len = len(encoding.encode(SEPARATOR))


def construct_prompt(question):
    header = f"Answer the following answer as if you were {option}"

    return header + "\n\n Q: " + question + "\n A:"



def answer(
    query: str,
    show_prompt: bool = False
) -> str:

    prompt = construct_prompt(query)

    if show_prompt:
        print(prompt)

    response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo-0301",
                messages=[{"role": "user", "content": prompt}],
                temperature = 0
            )

    return response.choices[0].message.content.strip(" \n")



def main():
    '''
    test
    '''
    openai.api_key = st.secrets['OPENAI_KEY']




    # Convert document sections to embeddings
    # document_embeddings = compute_doc_embeddings(df)


    st.sidebar.header("Instructions")
    st.sidebar.info(
       '''This is a web application that enables you to ask questions related to a desired context.
       Enter a **query** in the **text box** and **press 'Submit'** to receive
       a **response** from ChatGPT.
       '''
    )


    # Create text area widget to receive a question
    question = st.text_area(f"Taking questions:")

    # Get answer to question...
    if st.button("Submit"):
        with st.spinner("Generating response..."):
            response = answer(question)

        # Create text area widget to provide a response to a question
        st.text_area("Response:", value=response, height=None)

    return


main()

#if __name__ == "__main__":
