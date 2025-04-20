**Instruction to run the files**

This program was developed and has only been tested on Windows. I cannot vouch for its execution on other Operating Systems.
After cloning this repository, follow the below steps:

Running streamlit application on localhost

1. Ensure that 'python' and 'pip' are installed.

2. Open a terminal/command prompt/powershell and change the directory to the cloned repository.

3. Run "pip install -r requirements.txt".
	Note: This process may take up to 15 mins to complete as several packages/dependencies are installed during this.

4. Run "streamlit run app.py".

Running 'data_analysis' notebook on Google Colab

1. After logging and opening Colab, you will be asked to open a new notebook.
   Go to 'Upload' and upload the 'data_analysis' file in the cloned repository.

2. Once open, click on the first cell and hit 'Ctrl + F10' to run all the cell blocks
	Note: You will see some warning/error regarding 'pip' and prompted to restart the session. Restart it and if it keeps coming up,
	do it around 3 - 4 times and ignore warnings hereafter. Do not forget to re-run from the first cell after restarting the session.

3. If you wish to run the streamlit application within in Colab itself (i.e. host it from Colab itself), 
   navigate to the following block of code: 

	"!wget -q -O - https://loca.lt/mytunnelpassword"

   In the output of this code block, you will find a code in the following format: (XX.XX.XX.XX)
   Copy this code and follow the next step.

	Note: You will be prompted to enter a confirmation statement. Click on the empty box waiting for 
        a text prompt, and type 'y'.

4. Once you have run the entire notebook, the streamlit application should already be hosted locally. 
   You can navigate to the following code block:

	"!streamlit run demo.py & npx localtunnel --port 8501"

   In the output, you will find a "Local URL" (eg:- http://localhost:8501), open this URL.

5. You will be prompted to paste a code on this page as it is a tunnel page. The code that was copied from 
   Step 3 shall be pasted here. This will redirect you to the streamlit application where you can run the 
   application for as long as you want. 


**Project Description**

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

**Training the model and generating Maps**

If you wish to train the model and genetate the diagrams just run the .ipynb file and the diagrams and model will be saved on your local which you can later download onto your machine.


**Instructions to run the project**

🧩 Exploratory Analysis & Model Training
To perform data analysis and train the model, download the .ipynb notebook file and execute it cell by cell in your Colab environment. All graphs, maps, and the trained model will be generated and saved within the environment. You can choose to download these outputs or leave them as is.

💻 Running the Streamlit Application
To run the user interface, copy the content folder, app.py, and requirements.txt into a single directory. Install the required dependencies listed in the requirements.txt file, and then run the application using Streamlit.

**📂 Submitting PDF or CSV Datasets**

If you wish to submit your own dataset, please ensure the following:
  For PDF files, use a format similar to the sample provided in the submit folder.
  For CSV files, make sure the file size is less than 10 MB to avoid exceeding resource limits.
