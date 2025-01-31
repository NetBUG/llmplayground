FROM base_image

ENV HF_HOME=/app/cache
WORKDIR /app

COPY req*.txt /app/

RUN python3 -m pip install --no-cache-dir -r /app/requirements.txt && \
    python3 -m spacy download en_core_web_sm
# RUN apt install -y unzip wget && wget http://nb3.me/model330k.zip && unzip model330k.zip -d models
COPY . .

EXPOSE 80/tcp

ENTRYPOINT ["python3", "app.py", "--port", "80", "--device", "cuda:0"]
