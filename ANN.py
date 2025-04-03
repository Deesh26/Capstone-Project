#importing and definnig libraries

import pandas as pd
import numpy as np
import random
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from tensorflow import keras

from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.utils import to_categorical

import matplotlib.pyplot as plt




#loading the dataset from the excel file
excel_file = "sensor_data_training.xlsx"
df = pd.read_excel(excel_file)
#print(df.columns)

#checking for any missing values in the dataset
df.dropna(inplace=True)

# Feature selection (independent variables)
X = df[["Temperature (°C)", "SpO₂ (%)", "Blood Pressure (mmHg)", "Heart Rate (BPM)", "Age"]].values

# Define labels (classification based on predefined thresholds)
def classify_patient(temp, spo2, bp, hr, age):
    if ( 39 <= temp <= 35) or spo2 < 90 or bp > 180 or hr > 100 or age > 65  :  # Critical condition
        return 0  # Class A (Emergency)
    elif ((38 <= temp <= 39) or (36 <= temp <= 37)) or (90 <= spo2 <= 94) or (140 <= bp <= 180) or (85<= hr <=99 ) or (50<= age <= 65):  # Moderate
        return 1  # Class B (Urgent)
    else:  # Normal readings
        return 2  # Class C (Non-Urgent)
    
# Apply classification function
y = np.array([classify_patient(row[0], row[1], row[2], row[3], row[4]) for row in X])
y = to_categorical(y, num_classes=3)

# Split dataset into training (80%) and testing (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize data for better model performance
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

print("Data processing complete")

# Build MLP model(Multi-layer Perceptron)

#three layers to the MLP model (input, hidden,output)
#sequential creates a stack of layers or connections where each layer feeds its output into the next layer
# Dense(32, activation='relu', input_shape=(3,))----means there are 32 neurons/nodes processing data
#relu is a policy to add non-linearity so that the model can learn complex patterns
#input shape is nothing but the features of our program(temp, spo2,blood pressure)
#dropout(0.2) means the model ramdomly closes some neurons/nodes to prevent overfitting
#and the model is forced to not rely on the same amount of neurons all the time.
#activation='softmax' means that the raw output obtained is converted into probabalities(3 classes).
model = Sequential([
    Dense(32, activation='relu', input_shape=(5,)),  # First hidden layer
    Dropout(0.2),  # Dropout for regularization
    Dense(16, activation='relu'),  # Second hidden layer
    Dropout(0.2),
    Dense(3, activation='softmax')  # Output layer (3 classes)
])

#adam(adaptive moment estimation) is a optimizer which remembers past gradients and adjusts learning rate
#loss=categorical_crossentropy=reduces the loss value over time by adjusting model weights and
#compares the predicted probabalities(from softmax) with the true (actual) labels of stable , critical etc...
# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train, epochs=50, batch_size=8, validation_data=(X_test, y_test))

print("Model training complete!")


# Evaluate the model on test data
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {test_acc:.4f}")

# Plot training & validation 
plt.figure(1, figsize=(10, 5))
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.title("Model Accuracy Over Epochs")
plt.show()

# Plot training & validation loss

plt.figure(2, figsize=(10, 5))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.title("Model Loss Over Epochs")
plt.show()


#------------patient prediction-----------#
patient_file = "sensor_data.xlsx"
patient_df= pd.read_excel(patient_file)

# Function to fill random values for missing/zero BP and Age
def fill_random_bp_and_age(df):
    # Define ranges
    systolic_min, systolic_max = 90, 120
    age_min, age_max = 20, 25

    # Fill missing or zero BP
    df['Blood Pressure(mmHg)'] = df['Blood Pressure(mmHg)'].apply(
        lambda x: random.randint(systolic_min, systolic_max) if pd.isna(x) or x == 0 else x
    )

    # Fill missing or zero Age
    df['Age'] = df['Age'].apply(
        lambda x: random.randint(age_min, age_max) if pd.isna(x) or x == 0 else x
    )

    return df

# Apply the function before cleaning
patient_df = fill_random_bp_and_age(patient_df)
#print(df.columns)
# Remove rows with invalid or zero values
patient_df = patient_df[(patient_df != 0).all(axis=1)]

# Prediction function
def predict_patient_severity(temp, spo2, bp, hr, age):
    new_data = np.array([[temp, spo2, bp, hr, age]])
    new_data = scaler.transform(new_data)
    prediction = model.predict(new_data)
    predicted_class = np.argmax(prediction)

    classes = {
        0: ("Emergency (Class A)", 1, "0-5 minutes"),
        1: ("Urgent (Class B)", 2, "10-20 minutes"),
        2: ("Non-Urgent (Class C)", 3, "30+ minutes")
    }
    return classes[predicted_class]

# Apply prediction to each patient
severities = []
priorities = []
wait_times = []

for index, row in patient_df.iterrows():
    temp = row['Temperature (°C)']
    spo2 = row['SpO₂ (%)']
    bp = row['Blood Pressure(mmHg)']
    hr = row['Heart Rate (BPM)']
    age = row['Age']

    severity, priority, wait_time = predict_patient_severity(temp, spo2, bp, hr, age)
    
    severities.append(severity)
    priorities.append(priority)
    wait_times.append(wait_time)

# Add results to the DataFrame
patient_df['Severity'] = severities
patient_df['Priority'] = priorities
patient_df['Estimated Wait Time'] = wait_times

# Sort patients by priority (1 = highest)
priority_list = patient_df.sort_values(by='Priority')

# Display the priority list
print("\n PRIORITY LIST:")
print(priority_list[['Temperature (°C)', 'SpO₂ (%)', 'Blood Pressure(mmHg)','Heart Rate (BPM)','Age', 'Severity', 'Priority', 'Estimated Wait Time']])

# Save the priority list back to Excel
output_file = "patient_priority_list.xlsx"
priority_list.to_excel(output_file, index=False)
print(f"\n Priority list saved to '{output_file}'")










