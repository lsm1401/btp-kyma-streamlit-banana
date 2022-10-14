#Reference: https://docs.streamlit.io/knowledge-base/tutorials/deploy/docker

# STEP 1: Install Python base image
FROM python:3.9-slim

# Step 2: Add requirements.txt file 
COPY requirements.txt /requirements.txt
#
# Step 3:  Install required pyhton dependencies from requirements file
RUN pip install -r requirements.txt

# Step 4: Copy source code in the current directory to the container
ADD . /app

COPY app.py ./app.py
COPY ripeness.h5 ./ripeness.h5

# Step 5: Install git 
RUN apt-get update && apt-get install -y \
    build-essential \
    software-properties-common \
    git \
    && rm -rf /var/lib/apt/lists/*

# If you have git repo, you can clone your code that lives in a remote repo to WORKDIR
# RUN git clone https://github.com/streamlit/streamlit-example.git .

# Step 6: Set working directory to previously added app directory
WORKDIR /app

# # Step 7: Expose the port Flask is running on
EXPOSE 8501

# Step 8: Run the application
ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
