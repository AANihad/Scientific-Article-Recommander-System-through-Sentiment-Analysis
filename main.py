# import concurrent.futures
import pdftotext
import streamlit as st
import preprocessingFun, Citation, sentimentAnalysis
from Info import getAcc, getF1

import spacy
import unicodedata
import pandas as pd
import tempfile

nlp = spacy.load('en_core_web_sm')
nlp.add_pipe('sentencizer', before="parser" ) # updated
available_models = ["SciBERT", "SiEBERT", "BERT Large", "BERT Base", "XLNet Large", "XLNet Base"]

page_bg = """
<style>
[data-testid="stAppViewContainer"]{
background-color: #f5f5f5;
opacity: 1;
background-size: 13px 13px;
background-image: repeating-linear-gradient(45deg, #e5e5e5 0, #e5e5e5 1.3px, #f5f5f5 0, #f5f5f5 50%);
}
[data-testid="stHeader"]{
    opacity: 0.75;
    background-image: repeating-linear-gradient(45deg, #e5e5e5 0, #e5e5e5 1.3px, #f5f5f5 0, #f5f5f5 50%);
}
</style>
"""


def main():
    st.set_page_config(layout="wide")
    st.markdown(page_bg, unsafe_allow_html=True)
    st.title("Scientific Article Recommander System:")
    st.subheader("Using fine-tuned transformers on scientific citations")

    with st.sidebar:
        st.subheader("File:")
        uploaded_file = st.file_uploader("Choose a pdf file")

        st.subheader("Model:")
        mName = st.selectbox(
            'Choose the model you want to use for the recommendation',
            available_models
        )
        placeholder = st.empty()
        btn = placeholder.button('Sort References by sentiment', disabled=False, key='1')

        col1, col2 = st.columns(2)
        col1.metric("Accuracy", getAcc(mName),"")
        col2.metric("Macro-F1", getF1(mName), "")
    
    if uploaded_file is not None:
        with st.spinner("Reading File, please wait..."): 
            # To read file as bytes:   
            temp_file_1 = tempfile.NamedTemporaryFile(delete=False,suffix='.pdf')
            temp_file_1.write(uploaded_file.getbuffer()) 
            with open(temp_file_1.name, "rb") as f:
                pdf = pdftotext.PDF(f)

            

    if btn:
        originalCitations = []
        citations=[]
        if uploaded_file is not None:
            with st.spinner("Extracting References, please wait..."):
                # Get the references    
                references = preprocessingFun.extractRefs(temp_file_1.name)
                # Normalize the text
                content = str(unicodedata.normalize('NFKD', "\n".join(pdf)).encode('ASCII', 'ignore')).replace("\\n", "\n")
                # Clean the text
                text = preprocessingFun.clean_string(content.rsplit('References', 1)[0])
                doc = nlp(text)

            my_expander = st.expander(label="Extracted references")
            for i in references:
                my_expander.markdown("- " + i.getText())

            with st.spinner("Loading Model ..."):
                tokenizer, model = sentimentAnalysis.loadModel(mName)

            with st.spinner("Proccessing Text ..."):
                for sent in doc.sents:
                    if preprocessingFun.containsCitation(sent.text):
                        try:
                            citationText = sent.text
                            citationKeys = preprocessingFun.findKeys(preprocessingFun.containsCitation(sent.text))
                            citationReferences = preprocessingFun.findRef(citationKeys, references)
                            predictions = sentimentAnalysis.predict(citationText, model, tokenizer, mName)
                            cit = Citation.Citation(citationText, citationKeys, citationReferences, predictions)
                            originalCitations.append(citationText)
                            citations.append(cit)
                        except RuntimeError:
                            pass

                        
            for r in references[:]:
                for c in citations:
                    if c.refersTo(r):
                        r.addCitation(c)
                if (len(r.getCitations()) == 0):
                    references.remove(r)
                else:
                    r.calculateSentiments()


            df = pd.DataFrame([x.as_dict() for x in references])
            df = df.sort_values('Avg Pos', ascending=False)
            st.dataframe(df)

            st.markdown(" - Avg Pos: Average positive score of all citations citing this reference")
            st.markdown(" - Avg Obj: Average objective score of all citations citing this reference")
            st.markdown(" - Avg Neg: Average negative score of all citations citing this reference")
            st.markdown(" - Citation Count by sentiment: Respectively number of Positive, Objective and negative citations citing this reference")

        else:
            st.warning('Please Load a file first', icon="⚠️")


if __name__ == "__main__":
    main()