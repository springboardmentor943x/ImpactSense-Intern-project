FROM python:3.11-slim

WORKDIR /app

COPY app.py ./
COPY requirements.txt ./

RUN pip install -r requirements.txt

EXPOSE 9001

CMD ["streamlit", "run", "app.py", "--server.port=9001", "--server.address=0.0.0.0"]