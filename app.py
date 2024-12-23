import streamlit as st
from openai import OpenAI
import pandas as pd
import plotly.express as px

st.title("🥗 Customer Review Sentiment Analyzer")
st.markdown("This app analyzes the sentiment of customer reviews to gain insights into their opinions.")

# OpenAI API Key input
openai_api_key = st.sidebar.text_input(
    "Enter your OpenAI API Key", 
    type="password", 
    help="You can find your API key at https://platform.openai.com/account/api-keys"
)

def classify_sentiment_openai(review_text):
    """
    Classify the sentiment of a customer review using OpenAI's GPT-4o model.
    Parameters:
        review_text (str): The customer review text to be classified.
    Returns:
        str: The sentiment classification of the review as a single word, "positive", "negative", or "neutral".
    """
    client = OpenAI(api_key=openai_api_key)
    prompt = f'''
        Classify the following customer review. 
        State your answer
        as a single word, "positive", 
        "negative" or "neutral":

        {review_text}
        '''

    completion = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {
                "role": "user",
                "content": prompt
            }
        ]
    ) 

    return completion.choices[0].message.content


# CSV file uploader
uploaded_file = st.file_uploader(
    "Upload a CSV file with restaurant reviews", 
    type=["csv"])

# Once the user uploads a csv file:
if uploaded_file is not None: 
    # Read the file
    reviews_df = pd.read_csv(uploaded_file)

    # Check if the data has a text column
    text_columns = reviews_df.select_dtypes(include="object").columns

    if len(text_columns) == 0:
        st.error("No text columns found in the uploaded file.")

    # Show a dropdown menu to select the review column
    review_column = st.selectbox(
        "Select the column with the customer reviews",
        text_columns
    )

    # Analyze the sentiment of the selected column
    reviews_df["sentiment"] = reviews_df[review_column].apply(classify_sentiment_openai)
    
    # Display the sentiment distribution in metrics in 3 columns: Positive, Negative, Neutral
    # Make the strings in the sentiment column titled
    reviews_df["sentiment"] = reviews_df["sentiment"].str.title()
    sentiment_counts = reviews_df["sentiment"].value_counts()
    # st.write(reviews_df)
    # st.write(sentiment_counts)

    # Create 3 columns to display the 3 metrics
    col1, col2, col3 = st.columns(3)

    with col1:
        # Show the number of positive reviews and the percentage
        positive_count = sentiment_counts.get("Positive", 0)
        st.metric("Positive", 
                  positive_count, 
                  f"{positive_count / len(reviews_df) * 100:.2f}%")
    
    with col2:
        # Show the number of neutral reviews and the percentage
        neutral_count = sentiment_counts.get("Neutral", 0)
        st.metric("Neutral", 
                  neutral_count, 
                  f"{neutral_count / len(reviews_df) * 100:.2f}%")
    
    with col3:
        # Show the number of negative reviews and the percentage
        negative_count = sentiment_counts.get("Negative", 0)
        st.metric("Negative", 
                  negative_count, 
                  f"{negative_count / len(reviews_df) * 100:.2f}%")
        
    # Display pie chart
    fig = px.pie(
        values=sentiment_counts.values, 
        names=sentiment_counts.index, 
        title='Sentiment Distribution'
    )
    st.plotly_chart(fig)