# Use the official Python slim image
FROM python:3.10-slim

# 1) Set a working directory
WORKDIR /app

# 2) Copy your requirements and install
COPY requirements.txt .
RUN pip install --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt

# 3) Copy the rest of your code
COPY . .

# 4) Expose the port Streamlit runs on
EXPOSE 8501

# 5) Default command to run your app
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
