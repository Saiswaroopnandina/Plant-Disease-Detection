
    import streamlit as st
    import cv2
    import numpy as np
    import joblib
    from PIL import Image

    st.title("🌿 Plant Leaf Disease Detection (SVM)")

    clf = joblib.load("plant_disease_model.pkl")
    le = joblib.load("label_encoder.pkl")

    uploaded_file = st.file_uploader("Upload a leaf image...", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        img = Image.open(uploaded_file)
        st.image(img, caption="Uploaded Image", use_column_width=True)

        img = img.resize((128, 128))
        img_array = np.array(img).flatten().reshape(1, -1)

        prediction = clf.predict(img_array)[0]
        prob = clf.predict_proba(img_array).max()

        st.success(f"Predicted Disease: {le.inverse_transform([prediction])[0]} ({prob*100:.2f}%)")

    import os
    import cv2
    import numpy as np
    from sklearn.svm import SVC
    from sklearn.preprocessing import LabelEncoder
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report
    import joblib

    def load_dataset(data_dir, img_size=(128, 128)):
        X, y = [], []
        class_names = os.listdir(data_dir)
        for label in class_names:
            folder_path = os.path.join(data_dir, label)
            for file in os.listdir(folder_path):
                try:
                    img_path = os.path.join(folder_path, file)
                     img = cv2.imread(img_path)
                    img = cv2.resize(img, img_size)
                    X.append(img.flatten())
                    y.append(label)
                except:
                    continue
        return np.array(X), np.array(y), class_names

    def train_model(data_dir):
        X, y, class_names = load_dataset(data_dir)
        le = LabelEncoder()
        y_enc = le.fit_transform(y)

        X_train, X_test, y_train, y_test = train_test_split(X, y_enc, test_size=0.2, random_state=42)

        clf = SVC(kernel='linear', probability=True)
        clf.fit(X_train, y_train)

        y_pred = clf.predict(X_test)
        print(classification_report(y_test, y_pred, target_names=le.classes_))

        joblib.dump(clf, "plant_disease_model.pkl")
        joblib.dump(le, "label_encoder.pkl")
        print("Model and label encoder saved.")

    if __name__ == "__main__":
        train_model("dataset")
