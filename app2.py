from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression,ElasticNet
from sklearn.metrics import mean_squared_error
from sklearn.datasets import fetch_california_housing
import pandas as pd
import mlflow
from mlflow.models import infer_signature
import mlflow.sklearn
from urllib.parse import urlparse

import dagshub
dagshub.init(repo_owner='hetbhagatji09', repo_name='Mlflow-experiments', mlflow=True)

# Load the dataset
california = fetch_california_housing()
# Split the data into training and testing sets
X = california.data
y = california.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
alpha=0.5
l1_ratio=0.5
# Create and train the linear regression model
with mlflow.start_run():
    model = ElasticNet(alpha=alpha,l1_ratio=l1_ratio)
    model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test)
    mlflow.log_param("alpha",alpha)
    mlflow.log_param("l1_ratio",l1_ratio)
    
    
    # Evaluate the model
    mse = mean_squared_error(y_test, y_pred)
    print(f"Mean Squared Error: {mse}")
    mlflow.log_metric("mse", mse)
    # mlflow.log_artifact(X)
    
    remote_server_uri="https://dagshub.com/hetbhagatji09/Mlflow-experiments.mlflow"
     #now mlruns folder directly created in dagshub uri
    mlflow.set_tracking_uri(remote_server_uri)
   
    
    #"model" is a artifacts path  where model is stored as pkl file 
    mlflow.sklearn.log_model(
                model, "model", registered_model_name="ElasticnetWineModel"
            )