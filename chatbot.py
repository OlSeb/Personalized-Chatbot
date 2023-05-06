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
        'Master Yoda',
        'Sponge Bob',
        'Tony Stark (Iron Man)',
        'Gandalf',
        "Sherlock Holmes - fictional detective created by Sir Arthur Conan Doyle",
        "Harry Potter - the protagonist of the Harry Potter series by J.K. Rowling",
        "Gandalf - the wizard from J.R.R. Tolkien's The Lord of the Rings",
        "Atticus Finch - the central character in Harper Lee's To Kill a Mockingbird",
        "Darth Vader - the iconic villain from the Star Wars franchise",
        "Holden Caulfield - the protagonist of J.D. Salinger's The Catcher in the Rye",
        "Frodo Baggins - the protagonist of J.R.R. Tolkien's The Lord of the Rings",
        "Tyrion Lannister - a prominent character from George R.R. Martin's A Song of Ice and Fire series (adapted into the TV series Game of Thrones)",
        "Elizabeth Bennet - the protagonist of Jane Austen's Pride and Prejudice",
        "Katniss Everdeen - the protagonist of Suzanne Collins' The Hunger Games trilogy",
        "Don Quixote - the main character of Miguel de Cervantes' novel Don Quixote",
        "Holden Caulfield - the protagonist of J.D. Salinger's The Catcher in the Rye",
        "Jay Gatsby - the titular character of F. Scott Fitzgerald's The Great Gatsby",
        "Hannibal Lecter - the infamous serial killer and cannibal from Thomas Harris' novels",
        "Hercule Poirot - the detective in Agatha Christie's novels",
        "Lisbeth Salander - the hacker and vigilante from Stieg Larsson's Millennium series",
        "Willy Wonka - the eccentric chocolatier from Roald Dahl's Charlie and the Chocolate Factory",
        "Hermione Granger - one of the main characters in the Harry Potter series by J.K. Rowling",
        "Holden Caulfield - the protagonist of J.D. Salinger's The Catcher in the Rye",
        "Dracula - the vampire from Bram Stoker's novel of the same name"))

else:
    option = st.selectbox(
        'Who do you want to talk to?',
        (" ",
        "Albert Einstein - one of the greatest scientists of all time",
        "Steve Jobs - visionary entrepreneur and founder of Apple Inc.",
        "Oprah Winfrey - media mogul, talk show host, and philanthropist",
        "Mahatma Gandhi - political and spiritual leader of India's independence movement",
        "Neil Armstrong - first person to walk on the moon",
        "William Shakespeare - renowned playwright and poet",
        "Martin Luther King Jr. - civil rights activist and leader",
        "Elon Musk - entrepreneur and CEO of SpaceX and Tesla Inc.",
        "Barack Obama - former US President and Nobel Peace Prize winner",
        "Mother Teresa - Catholic nun and humanitarian",
        "Elon Musk - entrepreneur and CEO of SpaceX and Tesla Inc.",
        "Albert Camus - philosopher and writer",
        "Princess Diana - beloved member of the British royal family",
        "J.K. Rowling - author of the Harry Potter series",
        "Stephen Hawking - theoretical physicist and cosmologist",
        "Nelson Mandela - anti-apartheid activist and former President of South Africa",
        "Malala Yousafzai - advocate for girls' education and Nobel Peace Prize winner",
        "Socrates - ancient Greek philosopher",
        "Emma Watson - actress and UN Women Goodwill Ambassador"))


# option = st.selectbox(
#     'Who do you want to talk to?',
#     ("Albert Einstein - one of the greatest scientists of all time",
#      "Steve Jobs - visionary entrepreneur and founder of Apple Inc.",
#      "Oprah Winfrey - media mogul, talk show host, and philanthropist",
#      "Mahatma Gandhi - political and spiritual leader of India's independence movement",
#      "Neil Armstrong - first person to walk on the moon",
#      "William Shakespeare - renowned playwright and poet",
#      "Martin Luther King Jr. - civil rights activist and leader",
#      "Elon Musk - entrepreneur and CEO of SpaceX and Tesla Inc.",
#      "Barack Obama - former US President and Nobel Peace Prize winner",
#      "Mother Teresa - Catholic nun and humanitarian",
#      "Elon Musk - entrepreneur and CEO of SpaceX and Tesla Inc.",
#      "Albert Camus - philosopher and writer",
#      "Princess Diana - beloved member of the British royal family",
#      "J.K. Rowling - author of the Harry Potter series",
#      "Stephen Hawking - theoretical physicist and cosmologist",
#      "Nelson Mandela - anti-apartheid activist and former President of South Africa",
#      "Malala Yousafzai - advocate for girls' education and Nobel Peace Prize winner",
#      "Socrates - ancient Greek philosopher",
#      "Emma Watson - actress and UN Women Goodwill Ambassador",
#      'Master Yoda',
#      'Sponge Bob',
#      'Tony Stark (Iron Man)',
#      'Gandalf',
#      "Sherlock Holmes - fictional detective created by Sir Arthur Conan Doyle",
#      "Harry Potter - the protagonist of the Harry Potter series by J.K. Rowling",
#      "Gandalf - the wizard from J.R.R. Tolkien's The Lord of the Rings",
#      "Atticus Finch - the central character in Harper Lee's To Kill a Mockingbird",
#      "Darth Vader - the iconic villain from the Star Wars franchise",
#      "Holden Caulfield - the protagonist of J.D. Salinger's The Catcher in the Rye",
#      "Frodo Baggins - the protagonist of J.R.R. Tolkien's The Lord of the Rings",
#      "Tyrion Lannister - a prominent character from George R.R. Martin's A Song of Ice and Fire series (adapted into the TV series Game of Thrones)",
#      "Elizabeth Bennet - the protagonist of Jane Austen's Pride and Prejudice",
#      "Katniss Everdeen - the protagonist of Suzanne Collins' The Hunger Games trilogy",
#      "Don Quixote - the main character of Miguel de Cervantes' novel Don Quixote",
#      "Holden Caulfield - the protagonist of J.D. Salinger's The Catcher in the Rye",
#      "Jay Gatsby - the titular character of F. Scott Fitzgerald's The Great Gatsby",
#      "Hannibal Lecter - the infamous serial killer and cannibal from Thomas Harris' novels",
#      "Hercule Poirot - the detective in Agatha Christie's novels",
#      "Lisbeth Salander - the hacker and vigilante from Stieg Larsson's Millennium series",
#      "Willy Wonka - the eccentric chocolatier from Roald Dahl's Charlie and the Chocolate Factory",
#      "Hermione Granger - one of the main characters in the Harry Potter series by J.K. Rowling",
#      "Holden Caulfield - the protagonist of J.D. Salinger's The Catcher in the Rye",
#      "Dracula - the vampire from Bram Stoker's novel of the same name"))
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
    This function gets the user input, passes it to the ChatGPT function and
    displays the response
    '''
    # When deploying over the web, UNCOMMENT. Your OpenAI API key will be in streamlit Secrets (under Settings)
    # openai.api_key = st.secrets['OPENAI_KEY']

    # When deploying locally, provide your openai key (DON"T provide your key over GitHub. Use the method above):
    openai.api_key = "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"



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
