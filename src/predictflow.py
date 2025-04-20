from metaflow import FlowSpec, step, Flow, Parameter, JSONType
import pandas as pd

class RegressorPredictFlow(FlowSpec):
    vector = Parameter('vector', type=JSONType, required=True)

    @step
    def start(self):
        run = Flow('Lab6TrainFlow').latest_run 
        self.train_run_id = run.pathspec 
        self.model = run['end'].task.data.model
        self.feature_names = run['end'].task.data.feature_names
        
        print("Input vector", self.vector)
        print("Expected feature order:", self.feature_names)

        self.X_input = pd.DataFrame([self.vector], columns=self.feature_names)

        # Next step: Done
        self.next(self.end)

    @step
    def end(self):
        print('Model', self.model)

        print("Prediction input:")
        print(self.X_input)

        prediction = self.model.predict(self.X_input)[0]
        print("Predicted price:", prediction)
        
if __name__=='__main__':
    RegressorPredictFlow()
