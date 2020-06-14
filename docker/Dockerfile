FROM infrarift/ubuntu16-py36:latest

WORKDIR /app

# Create the environment
COPY environment.yml .

RUN conda env create -f environment.yml

# Make RUN commands use the new environment:
SHELL ["conda", "run", "-n", "atenv", "/bin/bash", "-c"]
