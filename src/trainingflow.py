from metaflow import FlowSpec, step, Flow, Parameter, JSONType
import pandas as pd


class Lab6TrainFlow(FlowSpec):

    @step
    def start(self):
        import seaborn as sns
        from sklearn.model_selection import train_test_split

        # Ingest raw data
        df = sns.load_dataset('diamonds')
        y = df["price"]
        X = df.drop(columns="price")

        # Feature transformations
        X = pd.get_dummies(X, columns=['cut', 'color', 'clarity'], drop_first=True)
        X_important = X[["y", "carat", "x", "z", "clarity_SI2", "clarity_I1", "color_J", "clarity_SI1", "color_I", "clarity_VVS2", "depth", "color_H"]]
        self.feature_names = list(X_important.columns)
        
        self.X_full, self.X_test, self.y_full, self.y_test = train_test_split(X_important, y, test_size=0.2, shuffle=True, random_state=42)
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(self.X_full, self.y_full, test_size=0.25, shuffle=True, random_state=42)
        print("Data loaded successfully")

        # Next step: model training
        self.next(self.train_rf, self.train_gbm, self.train_dt)


    # Train RFRegressor
    @step
    def train_rf(self):
        from sklearn.ensemble import RandomForestRegressor
    
        self.model = RandomForestRegressor(n_estimators=50, max_depth=10, random_state=42)
        self.model.fit(self.X_train, self.y_train)

        # Next step: Choose Model
        self.next(self.choose_model)


    # Train GBM
    @step
    def train_gbm(self):
        from sklearn.ensemble import GradientBoostingRegressor
    
        self.model = GradientBoostingRegressor()
        self.model.fit(self.X_train, self.y_train)

        # Next step: Choose Model
        self.next(self.choose_model)

    # Train DT
    @step
    def train_dt(self):
        from sklearn.tree import DecisionTreeRegressor
    
        self.model = DecisionTreeRegressor()
        self.model.fit(self.X_train, self.y_train)

        # Next step: Choose Model
        self.next(self.choose_model)


    # Choose the best model
    @step
    def choose_model(self, inputs):
        import mlflow
        mlflow.set_tracking_uri('https://my-lab5-service-374799462061.us-west2.run.app')
        mlflow.set_experiment('lab6-diamond-dataset')

        # returns coefficient of determination (R² score)
        def score(inp):
            r2 = inp.model.score(inp.X_test, inp.y_test)
            return inp.model, r2

        self.X_test = inputs[0].X_test
        self.y_test = inputs[0].y_test

        self.results = sorted(map(score, inputs), key=lambda x: -x[1])
        self.model = self.results[0][0]
        self.feature_names = inputs[0].feature_names

        # Register and save model using the MLFlow model registry.
        with mlflow.start_run():
            mlflow.sklearn.log_model(self.model, artifact_path = 'lab6_models', registered_model_name="lab6-best-model")
            mlflow.log_metric("r2_score", self.results[0][1])
            mlflow.end_run()

        # Next step: Done
        self.next(self.end)


    @step
    def end(self):
        self.X_test = self.X_test
        self.y_test = self.y_test
        self.feature_names = self.feature_names
        self.model = self.model
    
        print('Scores:')
        print('\n'.join('%s %f' % res for res in self.results))
        print('Model:', self.model)


if __name__=='__main__':
    Lab6TrainFlow()
