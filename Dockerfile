# Use an official Python runtime with Jupyter support
FROM jupyter/scipy-notebook:latest

# Set the working directory in the container
WORKDIR /app

# Copy the machine learning code and dataset into the container at /app
COPY MLops.ipynb /app/
COPY Social_Network_Ads.csv /app/

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir scikit-learn pandas

# Expose the port for Jupyter Notebook
EXPOSE 8888

# Command to run Jupyter Notebook
CMD ["jupyter", "notebook", "--ip='*'", "--port=8888", "--no-browser", "--allow-root"]
