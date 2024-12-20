import streamlit as st
import pandas as pd
import seaborn as sns

def main():
    st.set_page_config(page_title="Air Quality Analysis App", layout="wide")

    # Sidebar for navigation
    st.sidebar.title("Navigation")
    options = st.sidebar.radio("Select a section:", [
        "Welcome",
        "Data Overview",
        "Exploratory Data Analysis (EDA)",
        "Modeling and Prediction"
    ])

    if options == "Welcome":
        welcome_section()
    elif options == "Data Overview":
        data_overview_section()
    elif options == "Exploratory Data Analysis (EDA)":
        eda_section()
    elif options == "Modeling and Prediction":
        modeling_prediction_section()

def welcome_section():
    st.title("Welcome to the Air Quality Analysis App!")
    st.markdown("""
        This web application is designed to help you explore and analyze air quality data.
        The app allows you to:
        - View an overview of the dataset.
        - Perform Exploratory Data Analysis (EDA) to visualize key trends and distributions.
        - Build and make predictions using machine learning models.
        
        The app is powered by a trained machine learning model that predicts PM2.5 levels 
        based on various environmental factors such as temperature, humidity, and other air pollutants.
        
        Get started by navigating through the sections.
    """)

def data_overview_section():
    st.title("Data Overview")
    st.write("This section provides an overview of the dataset.")

    # Load dataset
    data = pd.read_csv("air_quality.csv")

    # Display the dataset
    st.write("### Dataset")
    st.dataframe(data)

    # Display basic statistics
    st.write("### Basic Statistics")
    st.write(data.describe())

    # Display dataset shape
    st.write("### Dataset Shape")
    st.write(f"Rows: {data.shape[0]}, Columns: {data.shape[1]}")

    # Display column names
    st.write("### Column Names")
    st.write(data.columns)

    # Display data types
    st.write("### Data Types")
    st.write(data.dtypes)

    # Check for missing values
    st.write("### Missing Values")
    missing_values = data.isnull().sum()
    st.write(missing_values[missing_values > 0])

    # Check for duplicate rows
    st.write("### Duplicate Rows")
    duplicates = data.duplicated().sum()
    st.write(f"Number of duplicate rows: {duplicates}")

    # Display unique values for each column as a table
    st.write("### Unique Values per Column")
    unique_values = pd.DataFrame({
        "Column Name": data.columns,
        "Unique Values": [data[col].nunique() for col in data.columns]
    })
    st.dataframe(unique_values)

    # Summary of categorical variables
    categorical_cols = data.select_dtypes(include=['object']).columns
    if len(categorical_cols) > 0:
        st.write("### Summary of Categorical Variables")
        for col in categorical_cols:
            st.write(f"**{col}**")
            st.write(data[col].value_counts())
    else:
        st.write("No categorical variables in the dataset.")

def eda_section():
    st.title("Exploratory Data Analysis (EDA)")
    st.write("Explore the dataset visually.")

    # Load dataset
    data = pd.read_csv("air_quality_numerical.csv")

    # Sidebar for navigation
    plot_type = st.sidebar.selectbox(
        "Select a plot type:",
        ["Correlation Heatmap", 
         "Histograms for All Numerical Variables", 
         "Bar Plots for Categorical Variables", 
         "Pair Plot", 
         "Grouped Bar Chart by Wind Direction", 
         "Station Comparison", 
         "Pie Chart of Pollutant Distribution"]
    )

    # Correlation heatmap
    if plot_type == "Correlation Heatmap":
        st.write("### Correlation Heatmap")
        fig, ax = plt.subplots(figsize=(20, 12))
        sns.heatmap(data.corr(), annot=True, cmap="coolwarm", ax=ax)
        st.pyplot(fig)

    # Histograms for all numerical variables
    elif plot_type == "Histograms for All Numerical Variables":
        st.write("### Histograms for Numerical Variables")
        numerical_cols = ['PM2.5', 'PM10', 'SO2', 'NO2', 'CO', 'O3', 'TEMP', 'PRES', 'DEWP', 'RAIN', 'WSPM']
        for col in numerical_cols:
            st.write(f"#### {col} Distribution")
            fig, ax = plt.subplots(figsize=(8, 5))
            sns.histplot(data[col], kde=True, bins=30, color='blue', ax=ax)
            ax.set_title(f"Distribution of {col}")
            st.pyplot(fig)

    # Bar plot for categorical variables
    elif plot_type == "Bar Plots for Categorical Variables":
        st.write("### Bar Plots for Categorical Variables")
        categorical_cols = ['wd', 'station']
        for col in categorical_cols:
            st.write(f"#### Distribution of {col}")
            fig, ax = plt.subplots(figsize=(8, 5))
            data[col].value_counts().plot(kind='bar', color='orange', ax=ax)
            ax.set_title(f"Distribution of {col}")
            st.pyplot(fig)

    # Pair plot for numerical variables
    elif plot_type == "Pair Plot":
        st.write("### Pair Plot of Numerical Variables")
        numerical_cols = ['PM2.5', 'PM10', 'SO2', 'NO2', 'CO', 'O3', 'TEMP', 'PRES', 'DEWP', 'RAIN', 'WSPM']
        sns.pairplot(data[numerical_cols], diag_kind='kde', corner=True)
        st.pyplot()

    # Grouped bar chart by wind direction
    elif plot_type == "Grouped Bar Chart by Wind Direction":
        st.write("### Average Pollutant Levels by Wind Direction")
        grouped_data = data.groupby('wd')[['PM2.5', 'PM10', 'SO2', 'NO2', 'CO', 'O3']].mean()
        fig, ax = plt.subplots(figsize=(12, 8))
        grouped_data.plot(kind='bar', ax=ax)
        ax.set_title("Average Pollutant Levels by Wind Direction")
        st.pyplot(fig)

    # Compare stations
    elif plot_type == "Station Comparison":
        st.write("### Pollutant Levels by Monitoring Station")
        numerical_cols = ['PM2.5', 'PM10', 'SO2', 'NO2', 'CO', 'O3', 'TEMP', 'PRES', 'DEWP', 'RAIN']
        station_avg = data.groupby('station')[numerical_cols].mean()
        fig, ax = plt.subplots(figsize=(15, 10))
        station_avg.plot(kind='bar', stacked=True, ax=ax)
        ax.set_title("Pollutant Levels by Monitoring Station")
        st.pyplot(fig)

    # Pie chart of pollutant distribution
    elif plot_type == "Pie Chart of Pollutant Distribution":
        st.write("### Pie Chart of Pollutant Distribution")
        elements = ['PM2.5', 'PM10', 'SO2', 'NO2', 'CO', 'O3', 'TEMP', 'PRES', 'DEWP', 'RAIN']
        element_counts = data[elements].sum()
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.pie(element_counts, labels=elements, autopct='%1.1f%%', startangle=140, colors=plt.cm.tab10.colors)
        ax.set_title("Pollutant Distribution")
        st.pyplot(fig)

def modeling_prediction_section():
    st.title("Modeling and Prediction")
    st.write("Make predictions based on the trained model.")

    # Load dataset
    data = pd.read_csv("air_quality.csv")

    # Load trained model
    with open("model.pkl", "rb") as file:
        model = joblib.load(file)

    # Input features
    st.write("### Enter Feature Values for Prediction")
    inputs = {}
    for col in ["PM10", "SO2", "NO2", "CO", "O3", "TEMP", "PRES", "DEWP", "RAIN", "wd", "WSPM", "station"]:
        inputs[col] = st.number_input(f"Enter {col}:", value=0)

    # Prediction
    if st.button("Predict"):
        features = pd.DataFrame([inputs.values()], columns=inputs.keys())
        prediction = model.predict(features)
        st.write(f"### Predicted PM2.5 Level: {prediction[0]:.2f}")

if __name__ == "__main__":
    main()
