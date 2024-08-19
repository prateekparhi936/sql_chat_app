import streamlit as st
import google.generativeai as genai


# Configure Streamlit page
st.set_page_config(page_title='GenAI Web App', page_icon=None)

# Helper function to generate content
def generate_content(prompt):
    try:
        response = model.generate_content(prompt)
        if response and response.text:
            return response.text
        else:
            return "The response was blocked or could not be generated. Please try again with a different input."
    except Exception as e:
        return f"An error occurred: {e}"

# Helper function to clear conversation history
def clear_history():
    st.session_state.conversation_history = []


# Initialize conversation history if not already done
if 'conversation_history' not in st.session_state:
    st.session_state.conversation_history = []

# Main page
st.sidebar.title("GenAI Web App")
use_case = st.sidebar.radio("Select a Use Case", ["Home", "Data Model to SQL", "Text to SQL", "SQL Script Upload"])

if use_case == "Home":
    st.write("# Welcome to the GenAI Web App!")
    st.write("### Please select a use case from the sidebar on LEFT.")
    st.session_state.conversation_history = []

elif use_case == "Data Model to SQL":
    # Configure Google Generative AI with your API key
    
    # Configure Google Generative AI with your API key
    genai.configure(api_key=st.secrets['GOOGLE_API_KEY'])

    model = genai.GenerativeModel('gemini-pro')

    def upload_to_gemini(file, mime_type=None):
        """Uploads the given file to Gemini."""
        file = genai.upload_file(file, mime_type=mime_type)
        st.write(f"Uploaded file '{file.display_name}' as: {file.uri}")
        return file

    # Streamlit interface
    st.title("ER Diagram to BigQuery Script Generator")

    # Slider inputs for generation config
    temperature = st.slider("Temperature", min_value=0.0, max_value=1.0, value=1.0, step=0.1)
    top_p = st.slider("Top-p", min_value=0.0, max_value=1.0, value=0.95, step=0.05)
    top_k = st.slider("Top-k", min_value=0, max_value=100, value=64, step=1)
    max_output_tokens = st.slider("Max Output Tokens", min_value=1, max_value=8192, value=8192, step=1)

    # Create the model
    generation_config = {
        "temperature": temperature,
        "top_p": top_p,
        "top_k": top_k,
        "max_output_tokens": max_output_tokens,
        "response_mime_type": "text/plain",
    }

    model = genai.GenerativeModel(
        model_name="gemini-1.5-flash",
        generation_config=generation_config,
    )

    # Initialize chat history
    if "history" not in st.session_state:
        st.session_state.history = []

    if "chat_session" not in st.session_state:
        st.session_state.chat_session = None

    # File uploader for initial ER diagram
    uploaded_file = st.file_uploader("Upload an Entity Relationship Diagram", type=["png", "jpg", "jpeg"])

    if uploaded_file is not None:
        mime_type = uploaded_file.type
        # Upload file to Gemini
        with open(uploaded_file.name, "wb") as f:
            f.write(uploaded_file.getbuffer())
        gemini_file = upload_to_gemini(uploaded_file.name, mime_type=mime_type)

        # Start a chat session with the uploaded file
        st.session_state.chat_session = model.start_chat(
            history=[
                {
                    "role": "user",
                    "parts": [
                        gemini_file,
                        "You are an intelligent system. I am going to provide an Entity relationship diagram. You need to read that diagram and develop a BigQuery script using that. Use temp tables to store the joined table results. In one temp table do not have more than 3 tables joined. Create a separate table to store more than 3 tables join. Subsequently use the temp tables created and form the logic. and make sure to exclude ''' in the beginning and end.",
                    ],
                }
            ]
        )

    # Function to send and receive messages
    def chat_with_ai(user_message):

        message_content = user_message


        # Add user message to history
        st.session_state.history.append({"role": "user", "content": message_content})

        # Check if chat session is initialized
        if st.session_state.chat_session is None:
            st.error("Chat session is not initialized. Please upload an ER diagram to start.")
            return

        # Send message to AI and get response
        response = st.session_state.chat_session.send_message(message_content)

        # print(response)

        # Add AI response to history
        st.session_state.history.append({"role": "assistant", "content": response.parts[0].text})

    # User input section
    # user_input = st.text_input("Enter your message to the AI:")
    st.text_area("Ask your question in natural language", height=200)
    # uploaded_image = st.file_uploader("Or upload an image:", type=["png", "jpg", "jpeg"], key="chat_image_uploader")

    if st.button("Send"):
        if user_input:
            # print(user_input)
            chat_with_ai(user_input)

    if st.button("Clear Chat"):
        st.session_state.history = []
        st.session_state.chat_session = None

    # Display chat history
    for message in st.session_state.history:
        if message["role"] == "user":
            st.write(f"**You:** {message['content']}")
        else:
            # st.write(f"**AI:** {message['content']}")
            st.write(f"**AI:** {message['content']}")

elif use_case == "Text to SQL":
    # Setting web app page name and selecting wide layout (optional)
    # Streamlit interface
    st.title("Text to SQL and schema insights")

    # Setting column sizes for schema input and question input
    col1, col2 = st.columns((1.5, 1.5))

    # Left side: Textbox for Database Schema input
    with col1:
        st.markdown("### :scroll: Input Database Schema")
        schema_input = st.text_area("Provide your database schema here", height=300)

    # Right side: Textbox for Natural Language Questions
    with col2:
        st.markdown("### :question: Ask Questions About the Schema")
        question_input = st.text_area("Ask your question in natural language", height=300)
        submit = st.button("Get Response")

    # Google API key configuration
    
    # Configure Google Generative AI with your API key
    genai.configure(api_key=st.secrets['GOOGLE_API_KEY'])
    model = genai.GenerativeModel('gemini-pro')
    supportive_info2 = ["""Based on the SQL query code, create an example input dataframe before the SQL query code is applied and the output dataframe after the SQL query is applied. """]
    supportive_info3 = [""" Explain the SQL query in 2 lines as what's being done in detail without any example output."""]

    # Initialize conversation history
    if 'conversation_history' not in st.session_state:
        st.session_state.conversation_history = []

    # Helper function to generate content
    def generate_content(prompt):
        return model.generate_content(prompt).text

    # Handle submit button click
    if submit:
        with st.spinner("Generating response..."):
            if not schema_input.strip():
                st.error("No schema provided. Please provide a database schema to ask questions from.")
            elif not question_input.strip():
                st.error("No question provided. Please ask a question based on the provided schema.")
            else:
                # Construct the prompt using the schema and the question
                prompt = f"Schema:\n{schema_input}\n\nQuestion:\n{question_input}"
                response = generate_content(prompt)
                # response2=model.generate_content([supportive_info2[0], response.text])
                # response3=model.generate_content([supportive_info3[0], response.text])

                # If the response is unclear or not relevant, ask the user to be more specific
                if "not relevant" in response.lower() or "unclear" in response.lower():
                    st.warning("The question seems unclear or not relevant to the schema. Please be more specific.")
                else:
                    st.write("##### Generated SQL Query:")
                    st.code(response)

                    st.write("##### A Sample Of Expected Ouput :")
                    response2=model.generate_content([supportive_info2[0], response])
                    st.write(response2.text)

                    st.write("##### Explanation of the SQL Query code generated :")
                    response3=model.generate_content([supportive_info3[0], response])
                    st.write(response3.text)

                    # Save the conversation
                    st.session_state.conversation_history.append({
                        'schema': schema_input,
                        'query': question_input,
                        'response': response,
                        # 'sample_output': response2,
                        # 'explanation': response3
                    })

    # Display conversation history
    if st.session_state.conversation_history:
        st.write("#### :blue[Conversation History:]")
        for i, history in enumerate(st.session_state.conversation_history):
            st.write(f"**Query {i + 1}:** {history['query']}")
            st.code(history['response'])
            # st.write(history['sample_output'])
            # st.write(history['explanation'])
            st.write("---")

    # Allow users to ask follow-up questions based on previous responses
    follow_up_input = st.text_area('Ask a follow-up question based on the previous response')
    follow_up_submit = st.button("Ask Follow-Up")

    if follow_up_submit:
        with st.spinner("Generating follow-up response..."):
            if not st.session_state.conversation_history:
                st.error("No previous conversation found. Please ask an initial question first.")
            else:
                last_response = st.session_state.conversation_history[-1]['response']
                follow_up_prompt = f"Previous Response:\n{last_response}\n\nFollow-up Question:\n{follow_up_input}"
                follow_up_response = generate_content(follow_up_prompt)

                st.write("##### Follow-Up Response:")
                st.code(follow_up_response)

                # Save the follow-up question and response
                st.session_state.conversation_history.append({
                    'schema': schema_input,
                    'query': follow_up_input,
                    'response': follow_up_response,
                })

elif use_case == "SQL Script Upload":
    st.write("## :blue[SQL Script Upload]")

    # # User input for text prompt
    # query_input = st.text_area('Please enter your prompt using simple English')
    query_input=None
    sql_file = st.file_uploader("Or upload an SQL script", type=["sql"])
    submit = st.button("Get AI-response")

    # Google API key configuration
    
    # genai.configure(api_key=GOOGLE_API_KEY)
    genai.configure(api_key=st.secrets['GOOGLE_API_KEY'])
    model = genai.GenerativeModel('gemini-pro')

    # Supportive contexts for the generative model
    supportive_info1 = ["You have been provided with a BigQuery SQL script. "
                        "Your task is to modify the script based on the user's instructions. "
                        "The modifications may include fixing syntax errors, optimizing the query for performance, or making contextual changes and additions as specified by the user. "
                        "Ensure that the updated script is functional and follows BigQuery SQL best practices."
                        "nstructions for the Model:Error Fixing: Identify and correct any syntax errors in the provided SQL script. "
                        "Ensure the script is compatible with BigQuery SQL syntax and functions."
                        "Optimization: Analyze the query for performance bottlenecks, such as inefficient joins, unnecessary subqueries, or missing indexes. "
                        "Refactor the code to improve execution speed and resource usage.Contextual Changes: Based on the user's input, make the necessary additions or alterations to the script. "
                        "This may include adding new fields, applying different filters, or changing aggregation logic."
                        "Validation: Ensure the final script is executable in BigQuery without errors and returns the expected results.Output: Provide the updated SQL script along with brief explanations of the changes made.make sure to exclude ''' in the beginning and end."]
    #supportive_info2 = ["Based on the SQL query code, create an example input dataframe before the SQL query code is applied and the output dataframe after the SQL query is applied."]
    #supportive_info3 = ["Explain the SQL query in detail without any example output."]

    # Initialize conversation history
    if 'conversation_history' not in st.session_state:
        st.session_state.conversation_history = []

    # Helper function to generate content
    def generate_content(prompt):
        return model.generate_content(prompt).text

    # If submit is clicked
    if submit:
        with st.spinner("Generating.."):
            if sql_file:
                sql_script = sql_file.read().decode("utf-8")
                query_input = sql_script

            st.write("##### 1. The Generated SQL Query Code:")
            response = generate_content([supportive_info1[0], query_input])
            st.code(response)

            # st.write("##### 2. A Sample Of Expected Output with 3 rows of dummy data:")
            # response2 = generate_content([supportive_info2[0], response])
            # st.write(response2)

            # st.write("##### 3. Explanation of the SQL Query code generated:")
            # response3 = generate_content([supportive_info3[0], response])
            # st.write(response3)

            # Save the conversation
            st.session_state.conversation_history.append({
                'query': query_input,
                'response': response
                # 'sample_output': response2,
                # 'explanation': response3
            })

    # Display conversation history
    if st.session_state.conversation_history:
        st.write("#### :blue[Conversation History:]")
        for i, history in enumerate(st.session_state.conversation_history):
            st.write(f"**Query {i + 1}:** {history['query']}")
            st.code(history['response'])
            # st.write(history['sample_output'])
            # st.write(history['explanation'])
            st.write("---")

    # Allow users to ask follow-up questions based on previous responses
    follow_up_input = st.text_area('Ask a follow-up question')
    follow_up_submit = st.button("Ask Follow-Up")

    if follow_up_submit:
        with st.spinner("Generating follow-up response.."):
            last_response = st.session_state.conversation_history[-1]['response']
            follow_up_response = generate_content([last_response, follow_up_input])

            st.write("##### Follow-Up Response:")
            st.write(follow_up_response)

            # Save the follow-up question and response
            st.session_state.conversation_history.append({
                'query': follow_up_input,
                'response': follow_up_response,
                # 'sample_output': '',
                # 'explanation': ''
            })
    if st.button("Clear Chat"):
        clear_history()
