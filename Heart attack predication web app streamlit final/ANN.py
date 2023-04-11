
import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
from PIL import Image
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Dropout



@st.cache(allow_output_mutation=True)

def load_data(file):
    data = pd.read_csv(file)
    return data

# Check shape
def check_shape(data):
   
# Display image
    image = Image.open('/Users/shreyanthhg/Desktop/Heart attack predication web app streamlit final/heart.jpeg')
    st.image(image, caption='Heart prediction', use_column_width=True)
    st.write(data.head(10))
    if st.button('check shape'):
        st.write("Shape of the dataset:", data.shape)
    
    

# Describe
def describe_data(data):
    st.write("Descriptive statistics of the dataset:")
    st.write(data.head(10))
    if st.button('Describe'):
        st.write(data.describe())
    
def missing(data):
    count=data.isnull().sum().sort_values(ascending=False)
    perc=data.isnull().mean().sort_values(ascending=False)
    total = pd.concat([count,perc],axis=1,keys=['missing value count','missing value %'])
    st.write(data.head(10))
    if st.button('Check Missing Value'):
        st.dataframe(total)
        
        
def visualize_data(data):
    st.set_option('deprecation.showPyplotGlobalUse', False)
    #st.beta_set_page_config(layout="wide")

    st.title("Data Visualization with Streamlit")

    # Load dataset
    #data = st.cache(pd.read_csv)("/Users/shreyanthhg/Desktop/Heart attack predication web app streamlit copy/heart_failure_clinical_records_dataset.csv")

    # Display dataset
    st.write("### Dataset")
    st.dataframe(data)

    # Uni-variate analysis
    st.sidebar.subheader("Uni-variate Analysis")
    uni_variate_cols = data.columns
    selected_uni_variate_col = st.sidebar.selectbox("Select a column", uni_variate_cols)

    if st.sidebar.button("Plot"):
        st.write("### Uni-variate Analysis")
        sns.histplot(data[selected_uni_variate_col])
        st.pyplot()

    # Bi-variate analysis
    st.sidebar.subheader("Bi-variate Analysis")
    bi_variate_cols = data.columns
    selected_bi_variate_cols = st.sidebar.multiselect("Select two columns", bi_variate_cols)

    if st.sidebar.button("Plot", key="plot_button1"):
        st.write("### Bi-variate Analysis")
        sns.jointplot(x=data[selected_bi_variate_cols[0]], y=data[selected_bi_variate_cols[1]])
        st.pyplot()

    # Multi-variate analysis
    st.sidebar.subheader("Multi-variate Analysis")
    multi_variate_cols = data.columns
    selected_multi_variate_cols = st.sidebar.multiselect("Select columns", multi_variate_cols)

    if st.sidebar.button("Plot", key="plot_button2"):
        st.write("### Multi-variate Analysis")
        sns.pairplot(data[selected_multi_variate_cols])
        st.pyplot()
        

# Missing value imputation
def missing_value_imputation(data):
    st.write("Missing value imputation:")
    st.write(data.head(10))
    
    for col in data.columns:
        if data[col].dtype == 'int64':
            if st.button(f'Fill {col} with mean'):
                data[col].fillna(data[col].mean(), inplace=True)
        
        elif data[col].dtype == 'float64':
            if st.button(f'Fill {col} with median'):
                data[col].fillna(data[col].median(), inplace=True)
                
        elif data[col].dtype == 'object':
            if st.button(f'Fill {col} with mode'):
                data[col].fillna(data[col].mode()[0], inplace=True)
                
    
# Label encoding
def label_encoding(data):
    st.write("Label encoding:")
    st.write(data.head(10))
    le = LabelEncoder()
    data['DEATH_EVENT'] = le.fit_transform(data.iloc[:, -1])
    if st.button('Label encoder'):
        st.write(data.head(10))

# Standardization
def standardization(data):
    st.write("Standardization:")
    st.write(data.head(10))
    if st.button('Standardize'):
        scaler = StandardScaler()
        data.iloc[:, :-1] = scaler.fit_transform(data.iloc[:, :-1])
        st.write(data.head(10))


# Train test split
def train_test(data):
    st.write("Train test split:")
    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    st.write(data.head(10))
    if st.button('Split'):
        st.write("Training set shape:", X_train.shape, y_train.shape)
        st.write("Testing set shape:", X_test.shape, y_test.shape)
    return X_train,y_train,X_test,y_test   

# Model building using ANN
def ann_model(X_train, y_train):
    model = Sequential()
    
    st.write("Building ANN model:")
        
    model.add(Dense(10, activation='relu', input_dim=X_train.shape[1]))
    model.add(Dense(units = 8, kernel_initializer = 'uniform', activation = 'relu'))
    model.add(Dropout(0.25))
    model.add(Dense(units = 4, kernel_initializer = 'uniform', activation = 'relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=100, batch_size=32, verbose=0)
    st.write("ANN model built")
    return model

# Prediction
def prediction(model, X_test):
    st.write("Making predictions:")
    y_pred = model.predict(X_test)
    st.write("Predictions:", y_pred)
    return y_pred

def accuracy_score(y_pred, y_test):
    correct = 0
    total = len(y_test)
    for i in range(total):
        if y_pred[i] == y_test.iloc[i]:
            correct += 1
    a=(correct / total) * 100
    return a

def Heart_prediction(model,input_data):
    
    # changing the input_data to numpy array
    input_data_as_numpy_array = pd.DataFrame(input_data)
    
    # reshape the array as we are predicting for one instance
    input_data_reshaped = input_data_as_numpy_array.transpose()    
    st.write(input_data_reshaped)
    predict = prediction(model,input_data_reshaped)
    

    if (predict[0] == 0):
      return 'The results says the person is safe'
    else:
      return 'The person is likely to have heart attack'
  
        
    
# Main function
data= None

def main():
   
    st.title("Data Preprocessing and Model Building for Heart Attack prediction using Streamlit")
    st.sidebar.title("Select Options")
    
    # Load data
    data = load_data("/Users/shreyanthhg/Desktop/Heart attack predication web app streamlit final/heart_failure_clinical_records_dataset.csv")
    data=pd.DataFrame(data)
    # Sidebar options
    option = st.sidebar.selectbox("Select option", ["Check Shape", "Describe","Missing Value Check","Visualization", "Missing Value Imputation", 
                                                    "Label Encoding", "Standardization", "Train Test Split",
                                                    "Model Building", "Prediction","Input"])

    
    if option == "Check Shape":
        check_shape(data)

    elif option == "Describe":
        describe_data(data)
        
    elif option == "Missing Value Check":
        missing(data)
        
    elif option == "Visualization":
        st.subheader("Data Visualization")
        visualize_data(data)

    elif option == "Missing Value Imputation":
        missing_value_imputation(data)

    elif option == "Label Encoding":
        label_encoding(data)

    elif option == "Standardization":
        standardization(data)

    elif option == "Train Test Split":
        train_test(data)

    elif option == "Model Building":
        X_train,y_train,X_test,y_test = train_test(data)
        model = ann_model(X_train, y_train)
        

    elif option == "Prediction":
        
        X_train,y_train,X_test,y_test = train_test(data)
        
        model = ann_model(X_train, y_train)
        y_pred=prediction(model, X_test)
        y_pred = np.where(y_pred > 0.5, 1, 0)
        st.write("Predictions after thresholding:", y_pred)
        accuracy=accuracy_score(y_pred, y_test)
        st.write("Accuracy:", accuracy)
        
    elif option=='Input':
        
        st.title('Heart Attack Prediction Web App')
        
        #building model 
        X_train,y_train,X_test,y_test = train_test(data)
        model = ann_model(X_train, y_train)
        
        # getting the input data from the user
        
        age = st.text_input('Enter Age')
        anaemia = st.text_input('Anaemia')
        creatinine_phosphokinase = st.text_input('Creatinine_phosphokinase')
        diabetes = st.text_input('Diabetes ')
        ejection_fraction = st.text_input('Ejection_fraction')
        high_blood_pressure = st.text_input('high_blood_pressure')
        platelets = st.text_input('platelets value')
        serum_creatinine = st.text_input('serum_creatinine')
        serum_sodium = st.text_input('serum_sodium')
        sex = st.text_input('sex')
        smoking = st.text_input('smoking')
        time = st.text_input('time')
        
        age = float(age)
        anaemia = float(anaemia)
        creatinine_phosphokinase = float(creatinine_phosphokinase)
        diabetes = float(diabetes)
        ejection_fraction = float(ejection_fraction)
        high_blood_pressure = float(high_blood_pressure)
        platelets = float(platelets)
        serum_creatinine = float(serum_creatinine)
        serum_sodium = float(serum_sodium)
        sex = float(sex)
        smoking = float(smoking)
        time = float(time)

        
        # code for Prediction
        heart = ''
        
        # creating a button for Prediction
        input_data=[age,anaemia,creatinine_phosphokinase,diabetes,ejection_fraction,high_blood_pressure,platelets,serum_creatinine,serum_sodium,sex,smoking,time]
        if st.button('Heart Attack Prediction Test Result'):
            heart = Heart_prediction(model,input_data)
        
        
        st.success(heart)
        
        

if __name__ == '__main__':
    main()

