import streamlit as st
import joblib
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import re
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
import fitz#
import xgboost as xgb 
import xgboost  
import os


# ================
# PATH CONFIGURATION (ADDED)
# ================

def get_content_path(filename):
    # Works in both local and Docker environments
    base_path = os.path.join(os.getcwd(), "content")
    return os.path.join(base_path, filename)


# ================
# CONFIGURATION
# ================
severity_mapping = {
    "NON-CRIMINAL": "Severity 1",
    "SUSPICIOUS OCCURRENCE": "Severity 1",
    "MISSING PERSON": "Severity 1",
    "RUNAWAY": "Severity 1",
    "RECOVERED VEHICLE": "Severity 1",
    "WARRANTS": "Severity 2",
    "OTHER OFFENSES": "Severity 2",
    "VANDALISM": "Severity 2",
    "TRESPASS": "Severity 2",
    "DISORDERLY CONDUCT": "Severity 2",
    "BAD CHECKS": "Severity 2",
    "LARCENY/THEFT": "Severity 3",
    "VEHICLE THEFT": "Severity 3",
    "FORGERY/COUNTERFEITING": "Severity 3",
    "DRUG/NARCOTIC": "Severity 3",
    "STOLEN PROPERTY": "Severity 3",
    "FRAUD": "Severity 3",
    "BRIBERY": "Severity 3",
    "EMBEZZLEMENT": "Severity 3",
    "ROBBERY": "Severity 4",
    "WEAPON LAWS": "Severity 4",
    "BURGLARY": "Severity 4",
    "EXTORTION": "Severity 4",
    "KIDNAPPING": "Severity 5",
    "ARSON": "Severity 5"
}

st.set_page_config(page_title="Crime Analytics Suite", layout="wide")

# Custom CSS
st.markdown("""
<style>
    .main > div {
        padding: 2rem 3rem;
    }
    .stSubheader {
        padding-top: 2rem !important;
    }
    .visualization-container {
        margin: 2rem 0;
        border-radius: 15px;
        background: #ffffff;
        padding: 1.5rem;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .full-width-viz {
        width: 100%;
        height: 700px;
        border: none;
    }
    .user-message {
        background-color: #e3f2fd;
        border-radius: 15px;
        padding: 10px;
        margin: 5px 0;
    }
    .assistant-message {
        background-color: #f5f5f5;
        border-radius: 15px;
        padding: 10px;
        margin: 5px 0;
    }
    [data-testid="stChatMessage"] {
        padding: 15px;
    }
</style>
""", unsafe_allow_html=True)

# ================
# HELPER FUNCTIONS
# ================
def load_html_file(filename):
    try:
        with open(filename, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        import streamlit as st
        st.error(f"File missing: {filename}")
        return "<div style='color:red'>Content not available</div>"

def load_svg_file(filename):
    with open(filename, "r", encoding="utf-8") as f:
        return f.read()
    
#================
#GPT FUNCTION
#================

gog='AIzaSyB1QKiTXJOkjcbfdVmIxq6A39S0ZSXG7_c'
genai.configure(api_key=gog)
model = genai.GenerativeModel('gemini-2.0-flash')

def get_csv_text(doc):
    text = ""
    df = pd.read_csv(doc)
    text = df.to_string(index=False)
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=5)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    if not text_chunks:
        raise ValueError("No text chunks found. Cannot create FAISS index.")

    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=gog)
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")


def get_conversational_chain():
    prompt_template = """
    I will providing you a dataset or a dataframe you need to answer all the questions as detailed as possible also you need to think about the probelem and the context before answering anything
    i need you to first think about the question and the context and to output even what you thought and how it makes sense then finally i want you to output the final answer.
    if the answer is not in
    provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
    Context:\n {context}\n
    Question: \n{question}\n
    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.3, google_api_key=gog)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)

    return chain

def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=gog)
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)

    docs = new_db.similarity_search(user_question)

    if not docs:
        return "answer is not available in the context."

    chain = get_conversational_chain()
    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)

    return response.get("output_text", "No answer generated.")



# ================
# DASHBOARD SECTION
# ================
def show_dashboard():
    st.title("📈 City X Crime Analysis")

    with st.container():
        st.header("⏳ Temporal Patterns")

        st.subheader("Annual Crime Trends")
        with st.expander("View Annual Trends"):
            st.components.v1.html(load_html_file(get_content_path("yearly_crime_count.svg")), height=600)


        with st.expander("View Monthly Trends"):
            st.components.v1.html(load_html_file(get_content_path("number_of_crimes_pm.svg")), height=600)

        with st.expander("View Weekly Trends"):
            st.components.v1.html(load_html_file(get_content_path("number_of_crimes_DOW.svg")), height=600)

        st.subheader("Hourly Patterns by Severity")
        with st.expander("Explore Hourly Trends"):
            st.components.v1.html(load_html_file(get_content_path('crime_per_hour_per_severity.html')), height=800)

        st.subheader("Monthly Patterns by Severity")
        with st.expander("Explore Monthly Trends"):
            st.components.v1.html(load_html_file(get_content_path('criem_per_month_per_severity.html')), height=800)


        st.subheader("Crime per month per month")
        with st.expander("Crime per month by severity"):
            st.components.v1.html(load_html_file(get_content_path("crime_per_month_per_crime.html")), height=800)




    st.divider()

    with st.container():
        st.header("⚠️ Crime Severity Analysis")

        st.subheader("Severity Distribution")
        with st.expander("View Severity Distribution"):
            st.components.v1.html(load_html_file(get_content_path("number_of_crimes_per_severity.svg")), height=600)

        st.subheader("Category Distribution")
        with st.expander("View Category Distribution"):
            st.components.v1.html(load_html_file(get_content_path("category_distribution.svg")), height=600)

        st.subheader("Resolution Rates")
        with st.expander("Analyze Resolution Rates"):
            tabs = st.tabs(["Severity 1", "Severity 2", "Severity 3", "Severity 4", "Severity 5"])
            with tabs[0]:
                st.components.v1.html(load_html_file(get_content_path("Res_for severity1.svg")), height=450)
            with tabs[1]:
                st.components.v1.html(load_html_file(get_content_path("Res_for severity2.svg")), height=450)
            with tabs[2]:
                st.components.v1.html(load_html_file(get_content_path("Res_for severity3.svg")), height=450)
            with tabs[3]:
                st.components.v1.html(load_html_file(get_content_path("Res_for severity4.svg")), height=450)
            with tabs[4]:
                st.components.v1.html(load_html_file(get_content_path("Res_for severity5.svg")), height=450)

    st.divider()

    with st.container():
        st.header("📍 Geographical Analysis")

        st.subheader("Crime Timelapse througout years")
        with st.expander("Explore Crime Density"):
            st.components.v1.html(load_html_file(get_content_path("crime_hots.html")), height=800)

        st.subheader("Crime Cluster")
        with st.expander("Crime exploration"):
            st.components.v1.html(load_html_file(get_content_path("crime_cluster.html")), height=800)

        st.subheader("Neighbourhood safety")
        with st.expander("Analyze Safe vs Unsafe"):
            tabs = st.tabs(["Safe Neighbourhood", "Unsafe Neighbourhood"])
            with tabs[0]:
                st.components.v1.html(load_html_file(get_content_path("safe_neighborhoods_map.html")), height=450)
            with tabs[1]:
                st.components.v1.html(load_html_file(get_content_path("unsafe_neighborhoods_map.html")), height=450)

    st.divider()

    with st.container():
        st.header("🔍 Crime Type Analysis")

        st.subheader("Top Crime Types")
        crime_types = {
            "Theft": get_content_path("theft_map.html"),
            "Vandalism": get_content_path("vandalism_map.html"),
            "Vehicle Theft": get_content_path("vehicle_robbery_map.html"),
            'Other offenses': get_content_path("other_offence_map.html"),
            'Non criminal': get_content_path("non_criminal_map.html")
        }
        for crime, path in crime_types.items():
            with st.expander(f"View {crime} Patterns"):
                st.components.v1.html(load_html_file(path), height=600)


        st.subheader("Least Crime Types")
        crime_types = {
            "Arson": get_content_path("34_arson_map.html"),
            "Embezzelment": get_content_path("33_embezzlement_map.html"),
            "Bad-check": get_content_path("30_bad_behavior_map.html"),
            'Bribery': get_content_path("31_bribery_map.html"),
            'Extrotion': get_content_path("32_extortion_map.html")
        }
        for crime, path in crime_types.items():
            with st.expander(f"View {crime} Patterns"):
                st.components.v1.html(load_html_file(path), height=600)

    st.divider()

    with st.container():
        st.header("👮 Police Department Performance")

        st.subheader(" Top 3 District Workload")
        districts = {
            "Southern": get_content_path("southern.html"),
            "Mission": get_content_path("mission.html"),
            "Northern": get_content_path("northern.html")
        }
        for district, path in districts.items():
            with st.expander(f"View {district} District Activity"):
                st.components.v1.html(load_html_file(path), height=600)
        st.subheader('Least Workload')
        with st.expander("View Richmond District Activity"):
           st.components.v1.html(load_html_file(get_content_path("richmond.html")), height=600)
# ================
# PDF PROCESSING
# ================
@st.cache_resource
def load_resources():
    return {
        "model": joblib.load(get_content_path("crime_classifier.pkl")),
        "vectorizer": joblib.load(get_content_path("tfidf_vectorizer.pkl")),
        "label_encoder": joblib.load(get_content_path("label_encoder.pkl"))
    }
def load_vectorizer():
   return joblib.load(get_content_path('tfidf_vectorizer.pkl'))
def load_model():
   return joblib.load(get_content_path('crime_classifier.pkl'))
def load_label_encoder():
  return joblib.load(get_content_path('label_encoder.pkl'))


def show_pdf_processing():
    st.title("📄 PDF Report Processing")
    st.write("Upload a crime report PDF to extract details and predict the crime category.")
    vectorizer = load_vectorizer()
    temp=load_model()
    label_encoder=load_label_encoder()
    # Initialize form data in session state
    if 'form_data' not in st.session_state:
        st.session_state.form_data = {
            "Report Number": "",
            "Date & Time": "",
            "Reporting Officer": "",
            "Incident Location": "",
            "Coordinates": "",
            "Police District": "",
            "Resolution": "",
            "Suspect Description": "",
            "Victim Information": "",
            "Detailed Description": "",
            "Category": "",
            "Severity": ""
        }

    uploaded_file = st.file_uploader("Upload PDF", type=["pdf"])

    if uploaded_file is not None:
        # Extract text from PDF
        with fitz.open(stream=uploaded_file.read(), filetype="pdf") as doc:
            text = "\n".join([page.get_text() for page in doc])

        # Parse crime report data
        dete = {}
        lines = text.splitlines()

        # Enhanced extraction with regex
        def extract_info(pattern, key):
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                dete[key] = match.group(1).strip()

        # Extract fields with improved regex patterns
        #for above for text extrcation the code is
        extract_info(r"Report Number:\s*(.+)", "Report Number")
        extract_info(r"Date & Time:\s*(.+)", "Date & Time")
        extract_info(r"Reporting Officer:\s*(.+)", "Reporting Officer")
        extract_info(r"Incident Location:\s*(.+)", "Incident Location")
        extract_info(r"Coordinates:\s*(.+)", "Coordinates")
        extract_info(r"Police District:\s*(.+)", "Police District")
        extract_info(r"Resolution:\s*(.+)", "Resolution")
        extract_info(r"Suspect Description:\s*(.+)", "Suspect Description")
        extract_info(r"Victim Information:\s*(.+)", "Victim Information")

        # Extract multi-line detailed description
        desc_match = re.search(r"Detailed Description:\s*\n([^\n]+(?:\n[^\n]+)?)(?=\n\s*\w+:|$)", text, flags=re.MULTILINE)
        if desc_match:
          dete["Detailed Description"] = desc_match.group(1).strip()

        # Update form data with extracted values (only if found)
        for key in st.session_state.form_data:
            if key in dete and dete[key]:
                st.session_state.form_data[key] = dete[key]

        # Update form data with extracted values (only if found)
        for key in st.session_state.form_data:
            if key in dete and dete[key]:
                st.session_state.form_data[key] = dete[key]

        # Auto-process description if available
        if dete.get("Detailed Description"):
            # Predict category and severity
            description_tfidf = vectorizer.transform([dete["Detailed Description"]])
            predicted_category_encoded = temp.predict(description_tfidf)
            predicted_category = label_encoder.inverse_transform(predicted_category_encoded)[0]

            # Update form fields with predictions
            st.session_state.form_data.update({
                "Category": predicted_category,
                "Severity": severity_mapping.get(predicted_category, "Unknown")
            })



    # Editable form
    with st.form("crime_report_form"):
        col1, col2 = st.columns(2)

        with col1:
            st.session_state.form_data["Report Number"] = st.text_input(
                "Report Number *",
                value=st.session_state.form_data["Report Number"]
            )
            st.session_state.form_data["Date & Time"] = st.text_input(
                "Date & Time",
                value=st.session_state.form_data["Date & Time"]
            )
            st.session_state.form_data["Reporting Officer"] = st.text_input(
                "Reporting Officer",
                value=st.session_state.form_data["Reporting Officer"]
            )
            st.session_state.form_data["Incident Location"] = st.text_input(
                "Incident Location",
                value=st.session_state.form_data["Incident Location"]
            )
            st.session_state.form_data["Coordinates"] = st.text_input(
                "Coordinates",
                value=st.session_state.form_data["Coordinates"]
            )

        with col2:
            st.session_state.form_data["Police District"] = st.text_input(
                "Police District",
                value=st.session_state.form_data["Police District"]
            )
            st.session_state.form_data["Resolution"] = st.selectbox(
                "Resolution",
                options=["Open", "Closed", "Pending", "Referred"],
                index=["Open", "Closed", "Pending", "Referred"].index(
                    st.session_state.form_data["Resolution"]
                ) if st.session_state.form_data["Resolution"] in ["Open", "Closed", "Pending", "Referred"] else 0
            )
            st.session_state.form_data["Suspect Description"] = st.text_area(
                "Suspect Description",
                value=st.session_state.form_data["Suspect Description"]
            )
            st.session_state.form_data["Detailed Description"]=st.text_area(
                "Detailed Description",
                value=st.session_state.form_data['Detailed Description']
            )
            st.session_state.form_data["Victim Information"] = st.text_area(
                "Victim Information",
                value=st.session_state.form_data["Victim Information"]
            )
            st.session_state.form_data["Category"] = st.text_input(
                "Category",
                value=st.session_state.form_data["Category"]
            )
            st.session_state.form_data["Severity"] = st.text_input(
                "Severity",
                value=st.session_state.form_data["Severity"]
            )

        # Form submission
        if st.form_submit_button("Submit Report"):
            if not st.session_state.form_data["Report Number"]:
                st.error("Report Number is required!")
            else:
                st.success(f"Report {st.session_state.form_data['Report Number']} submitted successfully!")
                st.json(st.session_state.form_data)

# ================
# RISK PREDICTION
# ================
@st.cache_resource
def load_prob_model():
    try:
        return joblib.load(get_content_path("Probability_classifier.pkl"))
    except Exception as e:
        st.error(f"Model loading failed: {str(e)}")
        return None

def show_probability():
    st.title("🔮 Number of Crime Risk Predictor")

    category_map = {
        "ARSON": 0, "BAD CHECKS": 1, "BRIBERY": 2, "BURGLARY": 3,
        "DISORDERLY CONDUCT": 4, "DRUG/NARCOTIC": 5, "EMBEZZLEMENT": 6,
        "EXTORTION": 7, "FORGERY/COUNTERFEITING": 8, "FRAUD": 9,
        "KIDNAPPING": 10, "LARCENY/THEFT": 11, "MISSING PERSON": 12,
        "NON-CRIMINAL": 13, "OTHER OFFENSES": 14, "RECOVERED VEHICLE": 15,
        "ROBBERY": 16, "RUNAWAY": 17, "STOLEN PROPERTY": 18,
        "SUSPICIOUS OCC": 19, "TRESPASS": 20, "VANDALISM": 21,
        "VEHICLE THEFT": 22, "WARRANTS": 23, "WEAPON LAWS": 24
    }

    district_map = {
        "BAYVIEW": 0, "CENTRAL": 1, "INGLESIDE": 2, "MISSION": 3,
        "NORTHERN": 4, "PARK": 5, "RICHMOND": 6, "SOUTHERN": 7,
        "TARAVAL": 8, "TENDERLOIN": 9
    }

    model = load_prob_model()
    if not model:
        st.stop()

    with st.form("prediction_form"):
        col1, col2 = st.columns([1, 1])

        with col1:
            crime_type = st.selectbox("Crime Type*", options=list(category_map.keys()))
            district = st.selectbox("Police District*", options=list(district_map.keys()))
            hour = st.slider("Hour of Day*", 0, 23, 12)

        with col2:
            latitude = st.number_input("Latitude*", value=37.7749, format="%.6f")
            longitude = st.number_input("Longitude*", value=-122.4194, format="%.6f")
            day = st.selectbox("Day of Week*",
                             ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"])

        if st.form_submit_button("Calculate Risk"):
            try:
                features = pd.DataFrame([[
                    district_map[district],
                    ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"].index(day),
                    hour,
                    latitude,
                    longitude,
                    category_map[crime_type]
                ]], columns=['PdDistrict_encode', 'DayOfWeek_encode', 'hour',
                            'latitude', 'longitude', 'Category_encoded'])

                probability = model.predict(features)[0]

                st.subheader("Prediction Results")
                col1, col2 = st.columns(2)
                col1.metric("Crime Probability",probability*100)

                if probability < 0.3:
                    col2.success("Low Risk Area")
                elif probability < 0.7:
                    col2.warning("Moderate Risk Area")
                else:
                    col2.error("High Risk Area")

            except Exception as e:
                st.error(f"Prediction error: {str(e)}")

def show_chatcsv():
    st.title("📊 CSV Chat Assistant")

    # Initialize chat history
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # File upload section
    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

    # Process CSV when file is uploaded
    if uploaded_file is not None:
        with st.spinner("Processing your CSV..."):
            try:
                # Process CSV and create vector store
                text = get_csv_text(uploaded_file)
                text_chunks = get_text_chunks(text)
                get_vector_store(text_chunks)
                st.success("CSV processed successfully! Ask me anything about the data.")
            except Exception as e:
                st.error(f"Error processing CSV: {str(e)}")
                return

    # Display chat messages
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input
    if prompt := st.chat_input("Ask a question about your CSV data"):
        if uploaded_file is None:
            st.error("Please upload a CSV file first!")
            return

        # Add user message to chat history
        st.session_state.chat_history.append({"role": "user", "content": prompt})

        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate response
        with st.spinner("Thinking..."):
            try:
                response = user_input(prompt)

                # Format response with markdown
                formatted_response = f"""
                <div style='
                    background-color: #f0f2f6;
                    padding: 15px;
                    border-radius: 10px;
                    margin: 10px 0;
                '>
                    {response}
                </div>
                """
            except Exception as e:
                response = f"Error generating response: {str(e)}"
                formatted_response = f"<div style='color: red'>{response}</div>"

        # Add assistant response to chat history
        st.session_state.chat_history.append({"role": "assistant", "content": response})

        # Display assistant response
        with st.chat_message("assistant"):
            st.markdown(formatted_response, unsafe_allow_html=True)


# ================
# MAIN APP
# ================
def main():
    st.sidebar.title("Crime Analytics Suite")
    page = st.sidebar.radio("Navigation", [
        "Crime Dashboard",
        "Report Processor",
        "Risk Predictor",
        "chatcsv"
    ])

    if page == "Crime Dashboard":
        show_dashboard()
    elif page == "Report Processor":
        show_pdf_processing()
    elif page == "Risk Predictor":
        show_probability()
    elif page == "chatcsv":
        show_chatcsv()

if __name__ == "__main__":
    main()
