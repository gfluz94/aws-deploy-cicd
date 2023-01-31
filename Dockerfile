FROM public.ecr.aws/lambda/python:3.9

COPY serve serve

RUN mv serve/app.py .

COPY ./output/default_classifier.pkl models/

COPY requirements.txt .

RUN python3.9 -m pip install -r requirements.txt -t .

CMD ["app.handler"]