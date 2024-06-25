import streamlit as st
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.summarize import load_summarize_chain
from transformers.models.t5.tokenization_t5 import T5Tokenizer
from transformers.models.t5.modeling_t5 import T5ForConditionalGeneration
from youtube_transcript_api import YouTubeTranscriptApi
import base64
import speech_recognition as sr
import google.generativeai as genai

def file_preprocessing(file):
    loader = PyPDFLoader(file)
    pages = loader.load_and_split()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=50)
    texts = text_splitter.split_documents(pages)
    final_texts = "".join([text.page_content for text in texts])
    return final_texts

def fetch_youtube_transcript(video_url):
    video = video_url.split("v=")[-1]
    transcript = YouTubeTranscriptApi.get_transcript(video)
    text = " ".join([item["text"] for item in transcript])
    return text

# def transcribe_audio(audio_filepath):
#     r = sr.Recognizer()
#     with sr.AudioFile(audio_filepath) as source:
#         audio = r.record(source)
#     text = r.recognize_google(audio)
#     return text

# from pydub import AudioSegment

# def transcribe_audio(audio_filepath):
#     # Load the audio file
#     audio = AudioSegment.from_file(audio_filepath)

#     # Convert the audio to WAV format
#     audio = audio.set_frame_rate(16000).set_channels(1)

#     # Save the converted audio to a temporary WAV file
#     temp_wav_filepath = "temp_audio.wav"
#     audio.export(temp_wav_filepath, format="wav")

#     # Transcribe the audio using SpeechRecognition
#     r = sr.Recognizer()
#     with sr.AudioFile(temp_wav_filepath) as source:
#         try:
#             audio_data = r.record(source)
#             text = r.recognize_google(audio_data)
#             return text
#         except sr.UnknownValueError:
#             st.error("Audio file could not be transcribed. Please ensure it contains recognizable speech.")
#             return ""
#         except sr.RequestError as e:
#             st.error(f"Could not request results from Google Speech Recognition service; {e}")
#             return ""

def transcribe_and_summarize_audio(audio_filepath):  
    genai.configure(api_key="AIzaSyD4lD8pcLQaSaoBiQQMUHRCoOTekeejUlU")

    audio_file = genai.upload_file(path=audio_filepath)
    model = genai.GenerativeModel("models/gemini-1.5-pro-latest")

    response = model.generate_content(
        [
            "Please summarize the following audio.",
            audio_file
        ]
    )
    return response.text

def llm_pipeline(input_text):
    checkpoint = "LaMini-Flan-T5-248M"
    tokenizer = T5Tokenizer.from_pretrained(checkpoint)
    model = T5ForConditionalGeneration.from_pretrained(checkpoint)

    input_ids = tokenizer.encode(input_text)
    output_ids = model.generate(input_ids, max_length=500, min_length=50, num_beams=4, early_stopping=True)[0]
    summary = tokenizer.decode(output_ids, skip_special_tokens=True)
    return summary

@st.cache_data
def displayPDF(file):
    with open(file, "rb") as f:
        base64_pdf = base64.b64encode(f.read()).decode('utf-8')
    pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="600" type="application/pdf"></iframe>'
    st.markdown(pdf_display, unsafe_allow_html=True)

def main():
    st.set_page_config(layout="wide")
    st.title("Document Summarization App using Language Model")

    uploaded_file = st.file_uploader("Upload your PDF file", type=['pdf'])
    youtube_url = st.text_input("Enter a YouTube video URL")
    audio_file = st.file_uploader("Upload an audio file", type=['wav', 'mp3'])

    if uploaded_file is not None:
        if st.button("Summarize PDF"):
            col1, col2 = st.columns(2)
            filepath = "data/" + uploaded_file.name
            # with open(filepath, "wb") as temp_file:
            #     temp_file.write(uploaded_file.read())

            with col1:
                st.info("Uploaded File")
                displayPDF(filepath)

            with col2:
                summary = llm_pipeline(file_preprocessing(filepath))
                st.info("Summarization Complete")
                st.success(summary)
    elif youtube_url:
        if st.button("Summarize YouTube Video"):
            col1, col2 = st.columns(2)

            with col1:
                st.info("YouTube Video URL")
                st.write(youtube_url)

            with col2:
                summary = llm_pipeline(fetch_youtube_transcript(youtube_url))
                st.info("Summarization Complete")
                st.success(summary)
                
    elif audio_file is not None:
        
        if st.button("Summarize Audio"):
            col1, col2 = st.columns(2)

            with col1:
                st.info("Uploaded Audio File")
                audio_filepath = "data/" + audio_file.name
                # with open(audio_filepath, "wb") as temp_audio_file:
                #     temp_audio_file.write(audio_file.read())
                st.audio(audio_filepath, format="audio/wav")

            with col2:
                summary = transcribe_and_summarize_audio(audio_filepath)
                st.info("Summarization Complete")
                st.success(summary)

    # elif audio_file is not None:
    #     if st.button("Summarize Audio"):
    #         col1, col2 = st.columns(2)

    #         with col1:
    #             st.info("Uploaded Audio File")
    #             audio_filepath = "data/" + audio_file.name
    #             with open(audio_filepath, "wb") as temp_audio_file:
    #                 temp_audio_file.write(audio_file.read())
    #             st.audio(audio_filepath, format="audio/wav")

    #         with col2:
    #             text = transcribe_audio(audio_filepath)
    #             summary = llm_pipeline(text)
    #             st.info("Summarization Complete")
    #             st.success(summary)

if __name__ == "__main__":
    main()