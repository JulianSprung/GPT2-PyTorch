FROM python:3.12.2-slim

ENV POETRY_HOME="/opt/poetry" \
    POETRY_VIRTUALENVS_CREATE=false 

# Add Poetry to PATH
ENV PATH="/opt/poetry/bin:$PATH"

# Install poetry
RUN apt-get update && apt-get install -y --no-install-recommends wget
RUN wget --https-only -O - https://install.python-poetry.org | python

WORKDIR /app

# Copy requirements and install dependecies
COPY ./pyproject.toml ./poetry.lock* /app/
RUN poetry install --only main

# Copy the code
COPY src/ /app/
CMD [ "python" , "main.py" ]