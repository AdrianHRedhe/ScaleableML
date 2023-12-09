import gradio as gr
import jiwer
from pytube import Search
from lyricsgenius import Genius
from transformers import pipeline
from evaluate import load
import librosa
import numpy as np
from math import ceil
import os

def Search_Genius(Artist, Songtitle):
    token = os.environ['Genius_API_Token']
    genius = Genius(token)
    song = genius.search_song(Songtitle, Artist)

    if song == None:
        return None
    
    lyrics = song.lyrics
    return lyrics

def Search_Youtube(Artist, Songtitle):
    s = Search(f'{Songtitle} {Artist} ')
    video = s.results[0]
    stream = video.streams.filter(only_audio=True).first()
    
    # Download the soundfile
    print('Before downloading / loading')
    target_sample_rate = 16000
    y, sr = librosa.load(stream.download(),sr=target_sample_rate)
    print('After downloading / loading')

    return y, sr

    # Kolla om sr istället för ggr 1000.
def Get_Song_Windows(sound_file, sample_rate):
    # Max size of the windows to go into whisper.
    window_size = 30 * sample_rate

    # Create n splits for inference
    song_length = sound_file.size
    n_splits = ceil(song_length/window_size)
    
    lowerBoundSplits = [i*window_size for i in range(n_splits)]
    upperBoundSplits = [min((i+1)*window_size,song_length) for i in range(n_splits)]

    song_windows = []

    for i in range(n_splits):
        lb = lowerBoundSplits[i]
        ub = upperBoundSplits[i]
        song_windows.append(sound_file[lb:ub])

    return song_windows

def get_foreground(y, sr, margin):
    S_full, phase = librosa.magphase(librosa.stft(y))
    S_filter = librosa.decompose.nn_filter(S_full, aggregate=np.median, metric='cosine', width=int(librosa.time_to_frames(2, sr=sr)))
    S_filter = np.minimum(S_full, S_filter)

    # We can use a margin to reduce bleed between the vocals and instrumentation masks.
    mask = librosa.util.softmask(S_full - S_filter, margin * S_filter, power=2)

    # Once we have the mask, simply multiply it with the input spectrum to separate the components
    S_foreground = mask * S_full
    y_foreground = librosa.istft(S_foreground * phase)

    return y_foreground

def transcribe(song_windows):
    print('Transcribe Time')
    transcriber = pipeline(model='adrianhr/whisper-small-sv-v4')
    print(f'n_song_windows = {len(song_windows)}')
    transcribed_splits = transcriber(song_windows)
    print(f'\n\nsplits:{transcribed_splits}\n\n')
    transcribed_splits_list = [e['text'] for e in transcribed_splits]
    transcribed_lyrics = ' '.join(transcribed_splits_list)
    return transcribed_lyrics

def clean_lyrics_and_calculate_wer(lyrics,transcribed_lyrics):
    # Clean the transcribed_lyrics 
    # We want to turn the punctuation into blanklines like the original
    transformation_condition = str.maketrans('.,!?','\n'*4)
    cleaned_transcribed_lyrics = transcribed_lyrics.translate(transformation_condition)

    # Clean Lyrics in the same way
    split_lyrics = lyrics.split('\n')
    split_lyrics_without_metadata = [row for row in split_lyrics if not '[' in row]
    split_lyrics_without_blanklines = [row for row in split_lyrics_without_metadata if len(row) != 0]
    joined_lyrics = '\n'.join(split_lyrics_without_blanklines)
    cleaned_lyrics = joined_lyrics.strip('.,!?')

    # Calculate WER
    wer_metric = load("wer")
    wer = wer_metric.compute(references=[cleaned_lyrics], predictions=[cleaned_transcribed_lyrics])

    return cleaned_lyrics, cleaned_transcribed_lyrics, wer

def get_teaser_audio(sound_file_foreground, sample_rate):
    lb = 20 * sample_rate
    ub = 30 * sample_rate
    sample = sound_file_foreground[lb:ub]
    return gr.Audio(value=(sample_rate, sample))

def on_failure_return_this():
    y,sr = librosa.load('audio.wav')
    audio_to_send_back = (sr,y)
    return audio_to_send_back, 'The Song is not available on Youtube or on Genius'

def greet(Artist, Songtitle):
    sound_file, sample_rate = Search_Youtube(Artist, Songtitle)
    # Optional but can improve the amount of text we get back
    # Try to mask instruments
    sound_file_foreground = get_foreground(sound_file, sample_rate, margin=3)
    song_windows_foreground = Get_Song_Windows(sound_file_foreground, sample_rate)

    transcribed_lyrics = transcribe(song_windows_foreground)
    lyrics = Search_Genius(Artist, Songtitle)

    if lyrics != None:
        cleaned_lyrics, cleaned_transcribed_lyrics, wer = clean_lyrics_and_calculate_wer(lyrics,transcribed_lyrics)
    else:
        cleaned_lyrics = 'No song with that title and artist was found on Genius'
        cleaned_transcribed_lyrics = transcribed_lyrics
        wer = 'Undetermined'

    sample_audio = get_teaser_audio(sound_file_foreground, sample_rate)

    return sample_audio , cleaned_lyrics, cleaned_transcribed_lyrics, wer

# Check WER between the two and output both.

iface = gr.Interface(fn=greet,inputs=["text","text"], outputs=[gr.Audio(),'text','text','text'])
iface.launch()