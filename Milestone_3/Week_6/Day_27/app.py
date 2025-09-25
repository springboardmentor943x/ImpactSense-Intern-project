import streamlit as st
import time


st.title('This is a Title')
st.header('This is a Header')
st.subheader('This is a Subheader')
st.text('This is a standard text message.')

st.markdown('# Markdown Title\nSome **bold** and _italic_ text.')
st.code('print("Hello, Streamlit!")', language='python')
st.latex('e^{i\\pi} + 1 = 0')
st.write('Write can display various data types including text, numbers, and dataframes')

if st.button('Click me'):
    st.write('Button clicked!')

if st.checkbox('Check me'):
    st.write('Checkbox is checked')

choice = st.radio('Choose one:', ['Option 1', 'Option 2'])
st.write(f'You chose {choice}')

option = st.selectbox('Select:', ['A', 'B', 'C'])
st.write(f'Selected {option}')

options = st.multiselect('Select multiple:', ['A', 'B', 'C'])
st.write(f'Selected {options}')

val = st.slider('Slide me', 0, 100, 25)
st.write(f'Slider value {val}')

text = st.text_input('Enter text:')
st.write(f'Your input: {text}')

num = st.number_input('Enter a number:', 0, 100)
st.write(f'Number: {num}')

date = st.date_input('Pick a date')
st.write(f'Date selected: {date}')

time_val = st.time_input('Pick a time')
st.write(f'Time selected: {time_val}')

file = st.file_uploader('Upload file')
if file:
    st.write(f'Filename: {file.name}')

img = st.camera_input('Take a picture')
if img:
    st.image(img)

color = st.color_picker('Pick a color')
st.write(f'Color: {color}')

col1, col2 = st.columns(2)
col1.write('Column 1')
col2.write('Column 2')

with st.expander('Expand me'):
    st.write('Hidden text here')

with st.sidebar:
    st.write('Sidebar content here')

st.balloons()

import time
with st.spinner('Loading...'):
    time.sleep(2)
st.write('Done!')

progress = st.progress(0)
for i in range(100):
    progress.progress(i + 1)
st.write('Progress complete')

st.metric(label='Temperature', value='70 °F', delta='-5 °F')