import pandas as pd 
import numpy as np 
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import StandardScaler 
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.metrics import accuracy_score 
data = { 
"Study Hours": [1, 2.5, 6, 2, 6, 4, 8, 9, 6, 8], 
"Attendance (%)": [18, 60, 90, 70, 85, 75, 85, 90, 85, 87], 
"Previous Score": [5, 18, 65, 40, 78, 85, 90, 85, 90, 80], 
"Final Exam Pass (1=Pass, 0=Fail)": [0, 0, 1, 0, 1, 1, 1, 1, 1, 1] 
} 
df = pd.DataFrame(data) 
X = df[["Study Hours", "Attendance (%)", "Previous Score"]] 
y = df["Final Exam Pass (1=Pass, 0=Fail)"] 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2) 
scaler = StandardScaler() 
X_train = scaler.fit_transform(X_train) 
X_test = scaler.transform(X_test) 
k = 3 
knn = KNeighborsClassifier(n_neighbors=k) 
knn.fit(X_train, y_train) 
y_pred = knn.predict(X_test) 
accuracy = accuracy_score(y_test, y_pred) 
print(f"KNN Model Accuracy: {accuracy: .6f}") 
new_student = np.array([[10, 70, 60]]) 
new_student_scaled = scaler.transform(new_student) 
prediction = knn.predict(new_student_scaled) 
result = "Pass" if prediction[0] == 1 else "Fail" 
print(f"Prediction for new student: {result}")
