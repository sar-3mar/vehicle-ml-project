import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder, OneHotEncoder, MinMaxScaler
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import category_encoders as ce

df = pd.read_csv("vehicles.csv", nrows=1000)
df = df[['price','year','odometer','fuel','transmission','manufacturer','condition']]
df = df.dropna()

categorical_features = ['fuel','transmission','manufacturer','condition']

print(" Original data:")
print(df.head(20))

# Label Encoding
df_le = df.copy()
le = LabelEncoder()
for col in categorical_features:
    df_le[col] = le.fit_transform(df_le[col])

print("\n after Label Encoding :")
print(df_le.head(20))

# Ordinal Encoding
df_oe = df.copy()
oe = OrdinalEncoder()
df_oe[categorical_features] = oe.fit_transform(df_oe[categorical_features])

print("\n after Ordinal Encoding :")
print(df_oe.head(20))

# One-Hot Encoding
df_ohe = df.copy()
ohe = OneHotEncoder(sparse_output=False, drop='first')
encoded = ohe.fit_transform(df_ohe[categorical_features])
df_ohe = pd.concat([df_ohe.drop(columns=categorical_features),
                    pd.DataFrame(encoded, columns=ohe.get_feature_names_out(categorical_features))],
                   axis=1)

print("\n after One-Hot Encoding :")
print(df_ohe.head(20))


# Count Encoding
df_count = df.copy()
ce_count = ce.CountEncoder()
df_count[categorical_features] = ce_count.fit_transform(df_count[categorical_features])

print("\n after Count Encoding :")
print(df_count.head(20))

# Target Encoding
df_target = df.copy()
ce_target = ce.TargetEncoder(cols=categorical_features)
df_target[categorical_features] = ce_target.fit_transform(df_target[categorical_features], df_target['price'])

print("\n after Target Encoding :")
print(df_target.head(20))

# Normalization 
scaler = MinMaxScaler()
df_le[['price','year','odometer']] = scaler.fit_transform(df_le[['price','year','odometer']])

# Splitting dataset 
X = df_le.drop('price', axis=1)
y = df_le['price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Linear Regression
lr = LinearRegression()
lr.fit(X_train, y_train)
print("\n Linear Regression R² Score:", lr.score(X_test, y_test))

# Logistic Regression
y_class = (y > y.median()).astype(int)
X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(X, y_class, test_size=0.2, random_state=42)

log_reg = LogisticRegression(max_iter=1000)
log_reg.fit(X_train_c, y_train_c)
y_pred_log = log_reg.predict(X_test_c)
print("\n Logistic Regression Accuracy:", accuracy_score(y_test_c, y_pred_log))
print("Confusion Matrix:\n", confusion_matrix(y_test_c, y_pred_log))
print("Classification Report:\n", classification_report(y_test_c, y_pred_log))


# KNN Regression
knn_reg = KNeighborsRegressor(n_neighbors=5)
knn_reg.fit(X_train, y_train)
print("\n KNN Regression R² Score:", knn_reg.score(X_test, y_test))

# KNN Classification
knn_clf = KNeighborsClassifier(n_neighbors=5)
knn_clf.fit(X_train_c, y_train_c)
y_pred_knn = knn_clf.predict(X_test_c)
print("\n KNN Classification Accuracy:", accuracy_score(y_test_c, y_pred_knn))
print("Confusion Matrix:\n", confusion_matrix(y_test_c, y_pred_knn))
print("Classification Report:\n", classification_report(y_test_c, y_pred_knn))




# /////////////////////////////////////////////////////////
# import pandas as pd

# df = pd.read_csv("vehicles.csv", nrows=20)  
# print(df.head(20).to_string())

