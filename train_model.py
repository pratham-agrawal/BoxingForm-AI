import os
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import TensorBoard
from sklearn.metrics import multilabel_confusion_matrix, accuracy_score

def load_dataset(folder_path, label):
    sequences, labels = [], []
    for file in os.listdir(folder_path):
        if os.path.isfile(os.path.join(folder_path, file)):
            print(os.path.join(folder_path, file))
            res = np.load(os.path.join(folder_path, file))
            sequences.append(res)
            labels.append(label)
    return sequences, labels

def create_model():
    label_map = {'normal': 0, 'good jab': 1, 'good guard': 2}
    control_seq, control_labels = load_dataset('training_data/H_Control', 0)
    jab_seq, jab_labels = load_dataset('training_data/H_Jab', 1)
    guard_seq, guard_labels = load_dataset('training_data/H_Guard', 2)

    sequences = control_seq + jab_seq + guard_seq
    labels = control_labels + jab_labels + guard_labels

    X = np.array(sequences)
    y = to_categorical(labels).astype(int)
    print(y)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

    log_dir = os.path.join('Logs')
    tb_callback = TensorBoard(log_dir=log_dir)

    print(X.shape)

    SEQUENCE_LENGTH = 15
    model = Sequential()
    model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(SEQUENCE_LENGTH, 92)))
    model.add(LSTM(128, return_sequences=True, activation='relu'))
    model.add(LSTM(64, return_sequences=False, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(3, activation='softmax')) #change to num categories in future
    model.compile(optimizer='Adam',loss='categorical_crossentropy',metrics=['categorical_accuracy'])
    model.fit(X_train, y_train, epochs=30, callbacks=[tb_callback])

    print(model.summary())
    model.save('form_correct.h5')
    # model.load_weights('correct.h5')

    yhat = model.predict(X_test)
    for i in range(len(yhat)):
        print(np.argmax(yhat[i]))
        print(np.argmax(y_test[i]))

    ytrue = np.argmax(y_test, axis=1).tolist()
    yhat = np.argmax(yhat, axis=1).tolist()
    print(multilabel_confusion_matrix(ytrue, yhat))
    print(accuracy_score(ytrue, yhat))


def main():
    create_model()


if __name__ == "__main__":
    main()