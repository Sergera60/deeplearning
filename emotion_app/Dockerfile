# -------------------------------------------------------
# 1. Base image (Python + slim Debian)
# -------------------------------------------------------
FROM python:3.10-slim

# -------------------------------------------------------
# 2. Set working directory inside the container
# -------------------------------------------------------
WORKDIR /app

# -------------------------------------------------------
# 3. Install system dependencies (for numpy, opencv, Pillow, etc.)
# -------------------------------------------------------
RUN apt-get update && apt-get install -y \
        build-essential \
        cmake \
        libgl1 \
        libglx-mesa0 \
        libglib2.0-0 \
        && rm -rf /var/lib/apt/lists/*

# -------------------------------------------------------
# 4. Copy requirements.txt first (to use Docker cache)
# -------------------------------------------------------
COPY requirements.txt .

# -------------------------------------------------------
# 5. Install Python dependencies
# -------------------------------------------------------
RUN pip install --no-cache-dir -r requirements.txt

# -------------------------------------------------------
# 6. Copy the rest of the project
# -------------------------------------------------------
COPY . .

# -------------------------------------------------------
# 7. Expose Streamlit port
# -------------------------------------------------------
EXPOSE 8501

# -------------------------------------------------------
# 8. Streamlit runs on 0.0.0.0 inside container
# -------------------------------------------------------
ENTRYPOINT ["streamlit", "run", "app.py", "--server.address=0.0.0.0"]
