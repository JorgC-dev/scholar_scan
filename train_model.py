import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd



class KnnModelStudent:

    def get_data(self,path):
        # Read CSV to get the data
        df = pd.read_csv(path)
        print(type(df))
        return df

    def train_model(self, df):

        #Create a scaler
        # scaler = StandardScaler()

        #we make a copy from de Df
        new_df = df.copy()

        family = ['Low', 'Middle','High']
        features_original = ['gpa', 'family', 'study_hours']
        features_col = ['gpa', 'family_encoded','study_hours']

        encoder = OrdinalEncoder(categories=[family])

        # we apply the encoder to dataframe column 'family'
        new_df['family_encoded'] = encoder.fit_transform(new_df[['family']])

        # Get features and caracteristics 
        X = new_df[features_col].values
        y = new_df['class_attendance'].values

        # To ensure, we print the dataframe and its properties
        print(type(X))
        print(X)
        print(X.shape)
        print(type(y))
        print(y.shape)
        print(y)


        # Split the data in train and test
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=24)

        # Train the model (Kneightbir clasifier)
        model = KNeighborsClassifier(n_neighbors=3)
        model.fit(X_train, y_train)

        # Train the model ()

        # Test the accuracy and model Score
        # first, predict the x_test
        y_predict = model.predict(X_test)

        # Evaluate the model
        accuracy = accuracy_score(y_test, y_predict)
        score = model.score(X_test,y_test)

        # print info model
        print("Model Score: ",score)
        print("Model acccuracy: ",accuracy)

        return model,encoder,features_original 

    def predict_risk(self, gpa, f_workload,s_hours, encoder,model, columns):
        # Make sure the format is correct
        gpa = float(gpa)
        s_hours = int(s_hours)

        # Create list
        data = [gpa, f_workload, s_hours]

        # Create dataframe 
        df = pd.DataFrame([data], columns=columns)
        
        # Tranform the column 'family workload' to encoder
        df['family_encoded'] = encoder.fit_transform(df[['family']])

        # drop 'family' column
        df.drop('family', axis=1, inplace=True)

        # get the names of columns
        columns  = df.columns.tolist()

        # get the values  from 
        X_new = df[columns].values

        # Predict the risk/class
        y_pred = model.predict(X_new)

        # print(y_pred)
        # print(type(y_pred))

        return y_pred[0]