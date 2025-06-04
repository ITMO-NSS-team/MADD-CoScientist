FROM python:3.10

WORKDIR /app

COPY . .

RUN apt-get update && \
    apt-get install -y build-essential vim && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install --no-deps ProtoLLM==0.1.2 && \
    rm -rf /root/.cache/pip && \
    echo '#!/bin/bash\n\
cd streamlit_app\n\
python -c "from streamlit_endpoints import load_papers; load_papers()"\n\
streamlit run app.py\n\
' > /app/run.sh && \
    chmod +x /app/run.sh

CMD ["/app/run.sh"]