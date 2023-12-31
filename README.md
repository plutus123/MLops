# MLops


### Repository Structure

/data: Contains the dataset used for training the machine learning model.

/MLop: Contains the main Python script for training the model (MLops.ipynb).

/tests: Contains unit tests for the machine learning code.

Dockerfile: Contains instructions for Docker containerization.

README.md: The main documentation file.


### Step 1 :Testing Process

The Random Forest Classifier has been chosen for its high accuracy, precision, recall, F1 score, and low log loss.

Various hyperparameters were tuned to achieve optimal results.

### Results

Accuracy Score: 0.925

Precision Score: 0.875

Recall Score: 0.9333

F1 Score: 0.9032

Log Loss: 0.3516



### Step 2 :Docker Containerization

In this step, we'll containerize the machine learning model along with its dependencies using Docker. This will allow for consistent and reproducible deployment across different environments.

## Dockerfile

The `Dockerfile` contains instructions to build the Docker image. It specifies the base image, sets the working directory, copies necessary files, installs required packages, and defines the command to run the application.

```Dockerfile
# Use an official Python runtime as a parent image
FROM python:3.8-slim

# Set the working directory in the container
WORKDIR /app

# Copy the machine learning code and dataset into the container at /app
COPY MLops.ipynb /app/
COPY Social_Network_Ads.csv /app/

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir scikit-learn pandas

# Command to run the machine learning script
CMD ["python", "MLops.ipynb"]
```
Building the Docker Image

To build the Docker image, execute the following command in your terminal:


```
docker build -t vishwasmishra/mlops_image:v1 .
```
This command instructs Docker to build an image based on the instructions in the Dockerfile and tag it as vishwasmishra/mlops_image:v1.

Running the Docker Image

You can run the Docker image with the following command:

```
docker run vishwasmishra/mlops_image:v1
```
This command starts a container using the vishwasmishra/mlops_image:v1 image, which will execute the machine learning script specified in the Dockerfile.

Pushing the Docker Image
If you want to share your Docker image, you can push it to a container registry like Docker Hub. To do so, follow these steps:

1.Log in to Docker Hub using the command docker login.

2.Tag the image with your Docker Hub username:

```
docker tag vishwasmishra/mlops_image:v1 vishwasmishra/mlops_image:v1
```

Push the image to Docker Hub:
```
docker push vishwasmishra/mlops_image:v1
```

Please remember to replace `vishwasmishra/mlops_image:v1` with your actual Docker Hub repository and image name. Additionally, make sure that the Dockerfile and the required files (`MLops.ipynb` and `Social_Network_Ads.csv`) are in the same directory as your README.md file.


## Step 3: Automated Testing

Automated testing has been set up to ensure the reliability of the machine learning code.

### Unit Tests

The repository includes unit tests to verify the functionality of the machine learning code. These tests cover critical components and edge cases.

To run the unit tests locally, use the following command:

```
python -m unittest tests/test_mlops.py
```

Version Control

The code in this repository is version-controlled using Git and hosted on GitHub.


