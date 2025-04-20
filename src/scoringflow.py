from metaflow import FlowSpec, step, Flow
import pandas as pd
import mlflow
import mlflow.sklearn

class Lab6TestFlow(FlowSpec):

    @step
    def start(self):
        # Load test data and feature names from the most recent training flow
        run = Flow('Lab6TrainFlow').latest_run
        self.X_test = run['end'].task.data.X_test
        self.y_test = run['end'].task.data.y_test
        self.feature_names = run['end'].task.data.feature_names

        print("Loaded test data with shape:", self.X_test.shape)

        self.next(self.load_model)

    @step
    def load_model(self):
        # Load registered model from MLflow (Production stage)
        mlflow.set_tracking_uri("https://my-lab5-service-374799462061.us-west2.run.app")
        model_uri = "models:/lab6-best-model/1"
        self.model = mlflow.sklearn.load_model(model_uri)

        print("Loaded model from MLflow:", model_uri)

        self.next(self.predict)

    @step
    def predict(self):
        # Make predictions
        predictions = self.model.predict(self.X_test)
        self.predictions = predictions.tolist()
        self.actuals = self.y_test.tolist()

        # Print sample results
        print("Sample predictions (Predicted vs Actual):")
        for i in range(min(10, len(self.predictions))):
            print(f"{i+1}. Predicted: {self.predictions[i]:.2f}, Actual: {self.actuals[i]:.2f}")

        self.next(self.end)

    @step
    def end(self):
        print("Scoring completed.")

if __name__=='__main__':
    Lab6TestFlow()
