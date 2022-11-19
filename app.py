import streamlit as st
import streamlit.components.v1 as com
import  streamlit_vertical_slider  as svs
import numpy as np
import scipy as sc
import pandas as pd
from math import ceil,floor
import plotly.express as px
import matplotlib.pyplot as plt
from glob import glob
import librosa
from librosa.effects import pitch_shift  
import librosa.display
from numpy import fft
import xlsxwriter
import io
from io import BytesIO 
from scipy.io.wavfile import write
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import time as ti
from scipy import signal
from Functions import Functions


st.set_page_config(page_title="Equalizer", page_icon=":bar_chart:",layout="wide")

st.set_option('deprecation.showPyplotGlobalUse', False)

reduce_header_height_style = """
    <style>
        div.block-container {padding-top:0rem;}
        header {visibility: hidden;}
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
    </style>
"""
st.markdown(reduce_header_height_style, unsafe_allow_html=True)

st.write(
    '<style>div.row-widget.stRadio > div{flex-direction:row;justify-content: left;} </style>', unsafe_allow_html=True)

def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

local_css("style.css")
# Initialization of session state
if 'sliderValues' not in st.session_state:
    st.session_state['sliderValues'] = []
if 'groups' not in st.session_state:
    st.session_state['groups'] = []
if 'audio' not in st.session_state:
    st.session_state['audio'] =[]
if 'sampleRare' not in st.session_state:
    st.session_state['sampleRare'] = []
if 'spectrum' not in st.session_state:
    st.session_state['spectrum'] = []
if 'fft_frequency' not in st.session_state:
    st.session_state['fft_frequency'] = []
if  'Uniform_Range_Default' not in st.session_state:
    st.session_state['Uniform_Range_Default']=[]
if  'spectrum_inv' not in st.session_state:
    st.session_state['spectrum_inv']=[]
if  'spectrum_inv_uniform' not in st.session_state:
    st.session_state['spectrum_inv_uniform']=[]
if 'start' not in st.session_state:
    st.session_state['start']=0
if 'size1' not in st.session_state:
    st.session_state['size1']=0
if  'spectrum_inv_uniform' not in st.session_state:
    st.session_state['spectrum_inv_uniform']=[]
if "pause_play_flag" not in st.session_state:
    st.session_state.pause_play_flag = False
if 'i' not in st.session_state:
    st.session_state['i']=0
if 'flag' not in st.session_state:
    st.session_state['flag'] = 0
if 'startSize' not in st.session_state:
    st.session_state['startSize'] = 0    


select_col,graph=st.columns((1,4))
with select_col:
     Mode_Selection=st.selectbox(
     'Equalizer',
     ('Uniform Range', 'Vowels', 'Musical Instruments','Voice Changer'))
     spec_visibality=st.checkbox("Spectrogram")
     if spec_visibality:
        Functions.plotSpectrogram(st.session_state['audio'],st.session_state['sampleRare'],'Input')
        Functions.plotSpectrogram(st.session_state['spectrum_inv'],st.session_state['sampleRare'],'Output')

ranges={
    "uniform":[],
    "music": [[0,279],[280, 1000],[1000, 7000]],
    "vocals":[[1895,7805],[100,1000],[55,460]]
}
if Mode_Selection=='Uniform Range' :
    sliders_number = 10
    rangesKey="uniform"
    # lst_final=[]
    text=["1st","2nd","3rd","4th","5th","6th","7th","8th","9th","10th"]
    flag=1
if Mode_Selection=='Musical Instruments' :
    sliders_number = 3
    Drums =[0,279]
    English_horn = [280, 1000]
    Glockenspeil =[1000, 7000]
    rangesKey="music"

    # lst_final_music=[Drums,English_horn,Glockenspeil]
    text=["Drums","English_horn","Glockenspeil"]
    flag=1
      
if Mode_Selection=='Vowels' :
    sliders_number = 3

    lst_sh=[1895,7805]
    lst_y=[100,1000]
    rangesKey="vocals"
    # lst_final=[lst_sh,lst_y]
    text=["SH","Y","Z"]
    flag=1

if Mode_Selection=='Voice Changer' :
    sliders_number = 1 
    flag= 1
    text=["Female to male"]    
with select_col:
        upload_file= st.file_uploader("Upload your File",type='wav')
if not upload_file:
    st.session_state['audio'],st.session_state['sampleRare']=librosa.load("audio\hello-female-friendly-professional.wav")
else:
    st.session_state['audio'],st.session_state['sampleRare']= librosa.load(upload_file)
if  flag==1:
    audio_trim,_ = librosa.effects.trim(st.session_state['audio'], top_db=30)
    st.session_state['audio']=audio_trim
    #play audio
    with select_col:
        st.write('Original Audio')
        if upload_file:
            st.audio(upload_file, format='audio/wav')
        else:
            input_bytes=Functions.convertToAudio(st.session_state['sampleRare'],st.session_state['audio'])
            st.audio(input_bytes, format='audio/wav')
    with graph:
      pause_btn= st.button("▷")
    # transform to fourier 
    list_freq_domain,frequncies, magnitude,phase, number_samples = Functions.fourier_transformation(st.session_state['audio'], st.session_state['sampleRare'])    
    freq_axis_list, amplitude_axis_list,bin_max_frequency_value = Functions.bins_separation(frequncies, magnitude, sliders_number)
    spTry,freqTry, magTry,phaseTry, number_samplesTry = Functions.fourier_transformation(st.session_state['audio'][0:2000], st.session_state['sampleRare'])
    valueSlider = Functions.Sliders_generation(sliders_number,text)
    value=valueSlider[0]
    if Mode_Selection=="Voice Changer":
       st.session_state['spectrum_inv']=pitch_shift(st.session_state['audio'] , sr= st.session_state['sampleRare'] , n_steps=value*20)
    else: 
        if Mode_Selection=="Uniform Range":
          Modified_signal=Functions.frequencyFunction(valueSlider, amplitude_axis_list) 
          st.session_state['spectrum_inv']=Functions.inverse(Modified_signal,phase) 
       
        else:
            Modified_signaltry=Functions.final_func(spTry,freqTry,ranges[rangesKey],valueSlider)
            fig_trans=px.line(x=freqTry, y=np.abs(Modified_signaltry)).update_layout(yaxis_title='Amp',xaxis_title='HZ')
            st.plotly_chart(fig_trans)
            Modified_signal=Functions.final_func(list_freq_domain,frequncies,ranges[rangesKey],valueSlider)
            # elif Mode_Selection== "Musical Instruments":
            #    Modified_signal=Functions.final_func(list_freq_domain,frequncies,lst_final_music,valueSlider)    
            st.session_state['spectrum_inv']=np.fft.irfft(Modified_signal)     
    #convert to audio
    result_bytes = Functions.convertToAudio(st.session_state['sampleRare'], st.session_state['spectrum_inv'])
    with graph:
        Functions.plotShow(st.session_state['audio'], st.session_state['spectrum_inv'], pause_btn,valueSlider,st.session_state['sampleRare'])
    with select_col:
        st.write('Modified Audio')
        st.audio(result_bytes, format='audio/wav')