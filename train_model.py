import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score
import joblib


def train_backpropagation_model():
    print("1. Loading dataset...")
    # Load the dataset we generated in Step 1
    df = pd.read_csv('dataset_siswa_training.csv')

    # 2. Feature Selection
    # We only want the grades as inputs (Features).
    # We drop ID, Name, DOB, School, Final Average, and the Label itself.
    X = df.drop(columns=['id_siswa', 'nama', 'tanggal_lahir', 'asal_sekolah',
                         'rata_rata_akhir', 'status_eligible'])

    # Y is our Target/Label (0 = Not Eligible, 1 = Eligible)
    y = df['status_eligible']

    print(f"Total input features per student: {X.shape[1]} (5 subjects x 6 semesters)")

    # 3. Train-Test Split
    # Split the data: 80% for training the ANN, 20% for testing its accuracy
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 4. Data Scaling (Crucial for Neural Networks!)
    # Neural Networks perform much better when all input values are normalized (scaled).
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print("2. Training the Backpropagation (MLP) Model...")
    # 5. Build the Backpropagation Neural Network
    # hidden_layer_sizes=(64, 32) means 2 hidden layers with 64 and 32 neurons.
    # max_iter=1000 ensures the network has enough epochs to converge.
    mlp_model = MLPClassifier(hidden_layer_sizes=(64, 32),
                              activation='relu',
                              solver='adam',
                              max_iter=1000,
                              random_state=42)

    # Train the model
    mlp_model.fit(X_train_scaled, y_train)

    print("3. Evaluating the Model...")
    # 6. Test the model's accuracy on the 20% unseen data
    y_pred = mlp_model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)

    print(f"✅ Model Accuracy: {accuracy * 100:.2f}%\n")
    print("Classification Report:")
    print(classification_report(y_test, y_pred, target_names=['Not Eligible', 'Eligible']))

    # 7. Save the Model and the Scaler
    print("4. Saving the model and scaler to disk...")
    joblib.dump(mlp_model, 'admission_mlp_model.pkl')
    joblib.dump(scaler, 'admission_scaler.pkl')
    print("✅ Successfully saved 'admission_mlp_model.pkl' and 'admission_scaler.pkl'")


if __name__ == "__main__":
    train_backpropagation_model()