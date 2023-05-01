# Use base image
FROM python:3.10.11-bullseye

# Prepare environment
RUN apt update -y \
&&  apt upgrade -y

# Add dependencies
COPY requirements.txt /req.txt
RUN pip3 install -r /req.txt
RUN mkdir /repo

# Set dir to repo
WORKDIR /repo

# Run the main command
CMD python /repo/all_download.py \
&&  python /repo/main.py
