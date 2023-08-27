import streamlit as st
import preprocessor,helper
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import classification_report
from sklearn.decomposition import PCA



# Download the vader_lexicon resource
import nltk
nltk.download('vader_lexicon')


# Load the VADER sentiment analyzer
sia = SentimentIntensityAnalyzer()

st.sidebar.title("Whatsapp Chat Analyzer")

uploaded_file = st.sidebar.file_uploader("Choose a file")
if uploaded_file is not None:
    bytes_data = uploaded_file.getvalue()
    data = bytes_data.decode("utf-8")
    df = preprocessor.preprocess(data)

    # Calculate additional statistics
    df['num_messages'] = 1
    df['words'] = df['message'].apply(lambda msg: len(msg.split()))
    df['num_media_messages'] = df['message'].apply(lambda msg: 1 if msg == '<Media omitted>\n' else 0)
    df['num_links'] = df['message'].apply(lambda msg: len(helper.extract.find_urls(msg)))

    # fetch unique users
    user_list = df['user'].unique().tolist()
    user_list.remove('group_notification')
    user_list.sort()
    user_list.insert(0,"Overall")

    selected_user = st.sidebar.selectbox("Show analysis wrt",user_list)

    if st.sidebar.button("Show Analysis"):

        # Stats Area
        num_messages, words, num_media_messages, num_links = helper.fetch_stats(selected_user,df)
        st.title("Top Statistics")
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.header("Total Messages")
            st.title(num_messages)
        with col2:
            st.header("Total Words")
            st.title(words)
        with col3:
            st.header("Media Shared")
            st.title(num_media_messages)
        with col4:
            st.header("Links Shared")
            st.title(num_links)

            # PCA Visualization
            if st.sidebar.checkbox("Perform PCA Visualization"):
                sentiment_scores = []
                for message in df['message']:
                    sentiment = sia.polarity_scores(message)
                    sentiment_scores.append(sentiment['compound'])

                df['sentiment_score'] = sentiment_scores

                X = df[['sentiment_score', 'num_messages', 'words', 'num_media_messages', 'num_links']]

                # Perform PCA
                pca = PCA(n_components=2)
                X_pca = pca.fit_transform(X)

                df_pca = pd.DataFrame(X_pca, columns=['PC1', 'PC2'])

                st.title("PCA Visualization")
                st.write("2D Scatter Plot of PCA Components")
                st.write(df_pca)

                # Scatter plot using Matplotlib
                plt.figure(figsize=(10, 6))
                plt.scatter(df_pca['PC1'], df_pca['PC2'], c='blue', alpha=0.5)
                plt.xlabel("Principal Component 1")
                plt.ylabel("Principal Component 2")
                plt.title("PCA Visualization")
                st.pyplot(plt)

            # Sentiment Analysis using Naive Bayes Classifier
            if st.sidebar.checkbox("Perform Sentiment Analysis using Naive Bayes"):
                sentiment_scores = []
                for message in df['message']:
                    sentiment = sia.polarity_scores(message)
                    sentiment_scores.append(sentiment['compound'])

                df['sentiment_score'] = sentiment_scores
                df['sentiment'] = df['sentiment_score'].apply(lambda score: 'Positive' if score >= 0 else 'Negative')

                X = df['message']
                y = df['sentiment']

                # Vectorize the text data using CountVectorizer
                vectorizer = CountVectorizer()
                X_vectorized = vectorizer.fit_transform(X)

                # Split the data into training and testing sets
                X_train, X_test, y_train, y_test = train_test_split(X_vectorized, y, test_size=0.2, random_state=42)

                # Train the Naive Bayes Classifier
                nb_classifier = MultinomialNB()
                nb_classifier.fit(X_train, y_train)

                # Predict sentiment on test data
                y_pred = nb_classifier.predict(X_test)

                # Evaluate and display classification report
                report = classification_report(y_test, y_pred)

                st.title("Sentiment Analysis using Naive Bayes")
                st.write("Classification Report:")
                st.write(report)

            # Sentiment Analysis using SVM
            if st.sidebar.checkbox("Perform Sentiment Analysis using SVM"):
                sentiment_scores = []
                for message in df['message']:
                    sentiment = sia.polarity_scores(message)
                    sentiment_scores.append(sentiment['compound'])

                df['sentiment_score'] = sentiment_scores

                X = df[['sentiment_score']]
                y = df['sentiment_score'].apply(lambda score: 'Positive' if score >= 0 else 'Negative')

                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

                svm_model = SVC(kernel='linear', random_state=42)
                svm_model.fit(X_train, y_train)

                y_pred = svm_model.predict(X_test)
                accuracy = accuracy_score(y_test, y_pred)

                st.title("Sentiment Analysis using SVM")
                st.write(f"Accuracy: {accuracy:.2f}")

            # Sentiment Analysis
            if st.sidebar.checkbox("Perform Sentiment Analysis"):
                sentiment_scores = []
                for message in df['message']:
                    sentiment = sia.polarity_scores(message)
                    sentiment_scores.append(sentiment)

                sentiment_df = pd.DataFrame(sentiment_scores)
                df_with_sentiment = pd.concat([df, sentiment_df], axis=1)

                st.title("Sentiment Analysis")
                st.dataframe(df_with_sentiment)

                # Decision Tree Classification
                if st.sidebar.checkbox("Perform Decision Tree Classification"):
                    X = df_with_sentiment[['compound']]
                    y = df_with_sentiment['compound'].apply(lambda x: 'Positive' if x >= 0 else 'Negative')

                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

                    max_depth = st.sidebar.number_input("Max Depth of Tree", min_value=1, max_value=10, value=3)
                    dt_classifier = DecisionTreeClassifier(max_depth=max_depth, random_state=42)
                    dt_classifier.fit(X_train, y_train)

                    y_pred = dt_classifier.predict(X_test)
                    accuracy = accuracy_score(y_test, y_pred)

                    st.title("Decision Tree Classification Results")
                    st.write(f"Accuracy: {accuracy:.2f}")

                # KNN Classification
                if st.sidebar.checkbox("Perform KNN Classification"):
                    X = df_with_sentiment[['compound']]
                    y = df_with_sentiment['compound'].apply(lambda x: 'Positive' if x >= 0 else 'Negative')

                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

                    k = st.sidebar.number_input("Number of Neighbors", min_value=1, max_value=10, value=5)
                    knn = KNeighborsClassifier(n_neighbors=k)
                    knn.fit(X_train, y_train)

                    y_pred = knn.predict(X_test)
                    accuracy = accuracy_score(y_test, y_pred)

                    st.title("KNN Classification Results")
                    st.write(f"Accuracy: {accuracy:.2f}")

                # K-Means Clustering
                if st.sidebar.checkbox("Perform K-Means Clustering"):
                    X = df_with_sentiment[['compound']]
                    scaler = StandardScaler()
                    X_scaled = scaler.fit_transform(X)

                    num_clusters = st.sidebar.number_input("Number of Clusters", min_value=2, max_value=10, value=3)
                    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
                    df_with_sentiment['cluster'] = kmeans.fit_predict(X_scaled)

                    st.title("K-Means Clustering Results")
                    st.dataframe(df_with_sentiment[['message', 'compound', 'cluster']])

                # Classification using Sentiment Scores
                if st.sidebar.checkbox("Perform Sentiment-Based Classification"):
                    # Assign labels based on compound sentiment score
                    df_with_sentiment['sentiment_label'] = df_with_sentiment['compound'].apply(lambda x: 'Positive' if x >= 0 else 'Negative')

                    # Split data for classification
                    X = df_with_sentiment[['compound']]
                    y = df_with_sentiment['sentiment_label']
                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

                    # Train Logistic Regression model
                    clf = LogisticRegression()
                    clf.fit(X_train, y_train)

                    # Evaluate and display accuracy
                    y_pred = clf.predict(X_test)
                    accuracy = accuracy_score(y_test, y_pred)
                    st.title("Sentiment-Based Classification Results")
                    st.write(f"Accuracy: {accuracy:.2f}")



        # monthly timeline
        st.title("Monthly Timeline")
        timeline = helper.monthly_timeline(selected_user,df)
        fig,ax = plt.subplots()
        ax.plot(timeline['time'], timeline['message'],color='green')
        plt.xticks(rotation='vertical')
        st.pyplot(fig)

        # daily timeline
        st.title("Daily Timeline")
        daily_timeline = helper.daily_timeline(selected_user, df)
        fig, ax = plt.subplots()
        ax.plot(daily_timeline['only_date'], daily_timeline['message'], color='black')
        plt.xticks(rotation='vertical')
        st.pyplot(fig)

        # activity map
        st.title('Activity Map')
        col1,col2 = st.columns(2)

        with col1:
            st.header("Most busy day")
            busy_day = helper.week_activity_map(selected_user,df)
            fig,ax = plt.subplots()
            ax.bar(busy_day.index,busy_day.values,color='purple')
            plt.xticks(rotation='vertical')
            st.pyplot(fig)

        with col2:
            st.header("Most busy month")
            busy_month = helper.month_activity_map(selected_user, df)
            fig, ax = plt.subplots()
            ax.bar(busy_month.index, busy_month.values,color='orange')
            plt.xticks(rotation='vertical')
            st.pyplot(fig)

        st.title("Weekly Activity Map")
        user_heatmap = helper.activity_heatmap(selected_user,df)
        fig,ax = plt.subplots()
        ax = sns.heatmap(user_heatmap)
        st.pyplot(fig)

        # finding the busiest users in the group(Group level)
        if selected_user == 'Overall':
            st.title('Most Busy Users')
            x,new_df = helper.most_busy_users(df)
            fig, ax = plt.subplots()

            col1, col2 = st.columns(2)

            with col1:
                ax.bar(x.index, x.values,color='red')
                plt.xticks(rotation='vertical')
                st.pyplot(fig)
            with col2:
                st.dataframe(new_df)

        # WordCloud
        st.title("Wordcloud")
        df_wc = helper.create_wordcloud(selected_user,df)
        fig,ax = plt.subplots()
        ax.imshow(df_wc)
        st.pyplot(fig)

        # most common words
        most_common_df = helper.most_common_words(selected_user,df)

        fig,ax = plt.subplots()

        ax.barh(most_common_df[0],most_common_df[1])
        plt.xticks(rotation='vertical')

        st.title('Most commmon words')
        st.pyplot(fig)

        # emoji analysis
        emoji_df = helper.emoji_helper(selected_user,df)
        st.title("Emoji Analysis")

        col1,col2 = st.columns(2)

        with col1:
            st.dataframe(emoji_df)
        with col2:
            fig,ax = plt.subplots()
            ax.pie(emoji_df[1].head(),labels=emoji_df[0].head(),autopct="%0.2f")
            st.pyplot(fig)