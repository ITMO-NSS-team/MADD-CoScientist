FROM python:3.12

WORKDIR /app

COPY . .

ENV POETRY_VIRTUALENVS_CREATE=false

RUN apt-get update && \
    apt-get install -y build-essential vim curl git && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* && \
    pip install poetry && \
    export PATH="/root/.local/bin:$PATH" && \
    poetry --no-interaction --no-ansi --no-root install && \
    poetry run pip install --no-deps git+https://github.com/aimclub/ProtoLLM.git@main && \
    rm -rf /root/.cache/pip

CMD ["streamlit", "run", "ChemCoScientist/streamlit_app.py"]
