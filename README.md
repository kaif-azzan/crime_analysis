**Crime Analysis**

This project focuses on analyzing the San Francisco crime dataset to uncover hidden patterns in criminal activities and build predictive capabilities to assist in understanding crime trends better.

**🔍 Project Overview**

Data Analysis
Explored and analyzed the San Francisco crime dataset to extract meaningful insights. Identified patterns in crime frequency, locations, and types to better understand how crime varies across different areas and times.

Machine Learning Model
Developed a machine learning model that performs two key tasks:

Predicts the category of crime based on input features such as crime descriptions and contextual parameters.

Estimates the likelihood of a crime occurring, given certain conditions (like time, location, and type of incident).

Natural Language Processing (NLP)
Utilized NLP techniques to process and understand the textual crime descriptions, helping improve model accuracy and interpretability.

Class Balancing
Addressed the imbalance in dataset classes by applying under-sampling and over-sampling strategies, ensuring fair representation of different crime categories for better predictive performance.

Interactive User Interface
Designed and developed an intuitive, user-friendly UI that features:

Map-based visualization of crime hotspots, making it easy to identify high-risk areas.

Dynamic graphs and charts to visualize crime trends over time, category distributions, and more.

Real-time interaction to explore data and prediction outputs seamlessly.

**🛠️ Tech Stack**

Python (for data processing and ML)

Scikit-learn, XGBoost (for machine learning models)

NLTK / SpaCy (for NLP)

Streamlit / React (for building the interactive UI)

Map libraries (like Folium or Leaflet.js) for map visualizations

Pandas, NumPy, Matplotlib, Seaborn (for data analysis and plotting)


**Instructions to run the project**

🧩 Exploratory Analysis & Model Training
To perform data analysis and train the model, download the .ipynb notebook file and execute it cell by cell in your Colab environment. All graphs, maps, and the trained model will be generated and saved within the environment. You can choose to download these outputs or leave them as is.

💻 Running the Streamlit Application
To run the user interface, copy the content folder, app.py, and requirements.txt into a single directory. Install the required dependencies listed in the requirements.txt file, and then run the application using Streamlit.

📂 Submitting PDF or CSV Datasets
If you wish to submit your own dataset, please ensure the following:
  For PDF files, use a format similar to the sample provided in the submit folder.
  For CSV files, make sure the file size is less than 10 MB to avoid exceeding resource limits.
