import math
import os
import numpy as np
import pandas as pd
import tensorflow as tf
import wfdb
from matplotlib import pyplot as plt
from ecgdetectors import Detectors
from sklearn.preprocessing import MinMaxScaler, label_binarize
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline
from tensorflow.keras import layers, Model, optimizers
from tensorflow.keras.layers import *
from tensorflow.keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.metrics import confusion_matrix, f1_score, precision_recall_curve, roc_curve, auc
import seaborn as sns


def load_one_record(nameStr, sampfrom, sampto):
    """
    one record for example file (100.dat) has nearly 30min*60sec*360hz=648000 samples in total
    (actually 30:05.556min length 650000 samples)
     For mitbit dataset, channel “0” is "MLII","1" is "V5". So set channels=[0,1] could read both
    """
    record = wfdb.rdrecord(os.path.join(rootdir, nameStr), sampfrom=sampfrom, sampto=sampto, channels=[0])
    return record


def load_one_annotation(nameStr, sampfrom, sampto):
    # annotation = wfdb.rdann(record_name=nameStr, extension='atr', pn_dir='mitdb', sampfrom=sampfrom, sampto=sampto, shift_samps=True)
    annotation = wfdb.rdann(os.path.join(rootdir, nameStr), 'atr', sampfrom=sampfrom, sampto=sampto, shift_samps=True)
    return annotation


def assign_labels_to_beats(r_peaks, annotation_samples, annotation_symbols):
    labels = []
    annotation_index = 0
    for r in r_peaks:
        # If the annotation index has reached the end of the annotation list
        if annotation_index >= len(annotation_samples) - 1:
            # Check if annotation is in acceptable list, else label 'N'
            label = annotation_symbols[annotation_index] if annotation_symbols[
                                                                annotation_index] in acceptable_annotations else 'N'
            labels.append(label)
            continue
        # If the current R peak is past the current annotation but before the next one
        if r > annotation_samples[annotation_index] and r <= annotation_samples[annotation_index + 1]:
            # Check if annotation is in acceptable list, else label 'N'
            label = annotation_symbols[annotation_index] if annotation_symbols[
                                                                annotation_index] in acceptable_annotations else 'N'
            labels.append(label)
        # If the current R peak is past the current annotation and the next one
        elif r > annotation_samples[annotation_index + 1]:
            while r > annotation_samples[annotation_index + 1]:
                annotation_index += 1
                if annotation_index >= len(annotation_samples) - 1:
                    break
            # Check if annotation is in acceptable list, else label 'N'
            label = annotation_symbols[annotation_index] if annotation_symbols[
                                                                annotation_index] in acceptable_annotations else 'N'
            labels.append(label)
        else:
            labels.append('N')
    return labels


def deal_data_to_CSV(rootdir):
    f = 360
    detectors = Detectors(sampling_frequency=f)
    files = os.listdir(rootdir)
    name_list = []
    type_counts = {}
    window_size = 259

    segmented_beats = []
    segmented_labels = []

    for file in files:
        if file[0:3] not in name_list:
            if file[0] not in ['1', '2', '3', '4', '5', '6', '7', '8', '9', '0']:
                continue
            # print(file[0:3])
            name_list.append(file[0:3])

    print('begin!')

    for name in name_list:
        record = load_one_record(name, 0, 650000)
        annotation = load_one_annotation(name, 0, 650000)

        # wfdb.plot_wfdb(record=record, annotation=annotation, title='MIT-BIH Record '+name,
        #                plot_sym=True, time_units='seconds', figsize=(15, 8))
        # wfdb.plot_wfdb(record=record, annotation=annotation, title='MIT-BIH Record '+name,
        #                plot_sym=True, time_units='samples', figsize=(15, 8))

        signal_normalized = MinMaxScaler(feature_range=(0, 1)).fit_transform(record.p_signal)
        # signal_normalized = (record.p_signal - np.min(record.p_signal)) / (np.max(record.p_signal) - np.min(record.p_signal))

        record.p_signal = signal_normalized

        # wfdb.plot_wfdb(record=record, annotation=annotation, title='MIT-BIH Record ' + name,
        #                plot_sym=True, time_units='samples', figsize=(15, 8))

        # Detect R peaks
        r_peaks = detectors.pan_tompkins_detector(unfiltered_ecg=record.p_signal.transpose(), MWA_name="cumulative")
        # r_peaks = detectors.pan_tompkins_detector(unfiltered_ecg=record.p_signal.transpose())
        # print(name, f'has', len(r_peaks), f'bests')

        for r_peak in r_peaks:
            # Ensuring the window stays within the array bounds
            start_index = max(r_peak - 89, 0)  # One less before the peak
            end_index = min(r_peak + 170, len(record.p_signal))  # Keeping 170 after the peak

            # Segment the signal using R peak location and the window size
            segment = record.p_signal[start_index:end_index]
            # convert shape (259,1) to (259,)
            segment = segment.flatten()

            # If the segment size is less than window size, pad it with zeroes
            if len(segment) < window_size:
                pad = np.zeros(window_size - len(segment))
                segment = np.append(segment, pad)

            # make sure 259 samples per windows
            assert len(segment) == window_size, f"Segment length {len(segment)} != window size {window_size}"

            segmented_beats.append(segment)

        segmented_label = assign_labels_to_beats(r_peaks, annotation.sample, annotation.symbol)

        # print(name, f'has', len(segmented_label), f'atr')

        for symbol in segmented_label:
            if symbol in type_counts:
                type_counts[symbol] += 1
            else:
                type_counts[symbol] = 1

        sorted_types = sorted(type_counts.items(), key=lambda d: d[1], reverse=True)

        segmented_labels += segmented_label

    print(f'Total ', len(segmented_beats), f'bests per 259 samples')
    print(f'Total ', len(segmented_labels), f'atr per 1 symbol')
    print(type_counts)

    # total 110084 beats and atrs
    # original distribution is {'N': 78831, 'A': 2535, 'V': 6361, '/': 7044, 'L': 8072, 'R': 7241}
    segmented_data = pd.DataFrame(segmented_beats)
    segmented_label = pd.DataFrame(segmented_labels)
    segmented_data.to_csv('X_mitdb_MLII.csv', index=False)
    segmented_label.to_csv('Y_mitdb_MLII.csv', index=False)
    print("save as csv finished")


class ECA_module(Layer):
    def __init__(self, gamma=2, b=1, **kwargs):
        super(ECA_module, self).__init__(**kwargs)
        self.gamma = gamma
        self.b = b
        self.conv = None

    def build(self, input_shape):
        N, L, C = input_shape
        t = int(abs((math.log(C, 2) + self.b) / self.gamma))
        k = t if t % 2 else t + 1
        self.conv = tf.keras.layers.Conv1D(1, k, padding='same', use_bias=False)
        print(k)
        super(ECA_module, self).build(input_shape)

    def call(self, x):
        N, L, C = x.shape
        y = tf.reduce_mean(x, axis=1, keepdims=True)

        y = tf.keras.layers.Reshape((1, C))(y)

        y = self.conv(y)
        y = tf.keras.activations.sigmoid(y)
        return x * y

    def get_config(self):
        config = super(ECA_module, self).get_config()
        config.update({
            'gamma': self.gamma,
            'b': self.b
        })
        return config


def Layer(layer_in, Filter):
    # x = ZeroPadding1D()(layer_in)
    x = Conv1D(Filter, kernel_size=7)(layer_in)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    return x


def residual_block(layer_in, n_filters):
    if n_filters == 64:
        shortcut = layer_in
        # conv1
        x = Conv1D(filters=n_filters, kernel_size=1)(layer_in)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        # conv2
        conv2 = Conv1D(filters=n_filters, kernel_size=1)(x)
        x = layers.BatchNormalization()(conv2)
        x = layers.Activation('relu')(x)
        # conv3
        # x = ZeroPadding1D(padding=4)(x)
        conv3 = Conv1D(filters=n_filters, kernel_size=1)(x)
        x = layers.BatchNormalization()(conv3)
        x = layers.add([x, shortcut])
        x = layers.Activation('relu')(x)
        x = ECA_module()(x)
        return x
    if n_filters == 128:
        shortcut = layer_in
        # conv1
        x = Conv1D(filters=n_filters, kernel_size=1)(layer_in)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        # conv2
        conv2 = Conv1D(filters=n_filters, kernel_size=1)(x)
        x = layers.BatchNormalization()(conv2)
        x = layers.Activation('relu')(x)
        # conv3
        # x = ZeroPadding1D(padding=1)(x)
        conv3 = Conv1D(filters=n_filters, kernel_size=1)(x)
        x = layers.BatchNormalization()(conv3)
        shortcut = Conv1D(filters=n_filters, kernel_size=1)(shortcut)
        x = layers.add([x, shortcut])
        x = layers.Activation('relu')(x)
        x = ECA_module()(x)
        return x


def ECANet(input_shape, num_classes):
    inputs = Input(shape=input_shape)

    x = Conv1D(64, kernel_size=15, padding='same')(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = ZeroPadding1D(padding=10)(x)
    maxp1 = MaxPooling1D(pool_size=3, strides=2, padding="same")(x)

    r1 = residual_block(maxp1, 64)

    x = Layer(r1, 64)
    x = ZeroPadding1D()(x)
    maxp2 = MaxPooling1D(pool_size=3, strides=2, padding="same")(x)

    r2 = residual_block(maxp2, 64)

    x = Layer(r2, 128)
    x = ZeroPadding1D()(x)
    maxp3 = MaxPooling1D(pool_size=3, strides=2, padding="same")(x)

    r3 = residual_block(maxp3, 128)

    x = Layer(r3, 256)
    x = ZeroPadding1D(padding=3)(x)
    x = Concatenate(axis=-1)([r3, x])

    # Global Average pooling and the Fully connected Layer
    x = GlobalMaxPooling1D(keepdims=False)(x)
    x = Flatten()(x)
    x = Dense(512, activation='relu')(x)
    x = Dense(512, activation='relu')(x)

    # output layer
    outputs = Dense(num_classes, activation='softmax')(x)

    # Create the model
    model = Model(inputs=inputs, outputs=outputs)

    return model


def ShowHistory(history):
    # Training Accuracy and Validation accuracy graph
    plt.figure(figsize=(10, 5))
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title("Model Accuracy")
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(['Training Accuracy', 'Validation accuracy'], loc='lower right')
    plt.show()

    # Loss
    plt.figure(figsize=(10, 5))
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title("Model Loss")
    plt.xlabel('Epoch')
    plt.ylabel('loss')
    plt.legend(['Training loss', 'Validation loss'], loc='upper right')
    plt.show()


def evaluateModel(model, X_test, y_test):
    loss, accuracy = model.evaluate(X_test, y_test)
    print('Test loss:', loss)
    print('Test accuracy:', accuracy)

    y_pred = model.predict(X_test)
    # Convert the predictions to binary format
    y_pred_labels = np.argmax(y_pred, axis=1)

    # Get the class labels for the true values
    y_true = np.argmax(y_test, axis=1)

    # Compute the confusion matrix
    confusion = confusion_matrix(y_true, y_pred_labels)
    print("confusion_matrix: ")
    print(confusion)

    # Plot confusion matrix as an image
    plt.figure(figsize=(10, 7))
    sns.heatmap(confusion, annot=True, cmap='YlGnBu', fmt='g')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()

    # Compute F1 score for each class separately
    f1 = f1_score(y_true, y_pred_labels, average=None)
    for i, score in enumerate(f1):
        print(f"F1 score for class {i}: {score}")
    # # Compute the F1 score
    # f1 = f1_score(y_true, y_pred_labels,
    #               average='micro')  # average should be one of {None, 'micro', 'macro', 'weighted', 'samples'}
    # print("f1_score: " + str(f1))

    # Binarize the output (one-hot encoding)
    y_true_bin = label_binarize(y_true, classes=[0, 1, 2, 3, 4, 5])
    y_pred_bin = label_binarize(y_pred_labels, classes=[0, 1, 2, 3, 4, 5])

    n_classes = y_true_bin.shape[1]

    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_pred_bin[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_true_bin.ravel(), y_pred_bin.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    # Compute Precision-Recall curve and area for each class
    precision = dict()
    recall = dict()
    pr_auc = dict()
    for i in range(n_classes):
        precision[i], recall[i], _ = precision_recall_curve(y_true_bin[:, i], y_pred_bin[:, i])
        pr_auc[i] = auc(recall[i], precision[i])

    # Compute micro-average Precision-Recall curve and area
    precision["micro"], recall["micro"], _ = precision_recall_curve(y_true_bin.ravel(), y_pred_bin.ravel())
    pr_auc["micro"] = auc(recall["micro"], precision["micro"])

    # Plot ROC curve for each class
    plt.figure(figsize=(8, 8))
    for i in range(n_classes):
        plt.plot(fpr[i], tpr[i], label='ROC curve of class {0} (area = {1:0.2f})'
                                       ''.format(i, roc_auc[i]))

    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic Curve')
    plt.legend(loc="lower right")
    plt.show()

    # Plot Precision-Recall curve for each class
    plt.figure(figsize=(8, 8))
    for i in range(n_classes):
        plt.plot(recall[i], precision[i], label='Precision-Recall curve of class {0} (area = {1:0.2f})'
                                                ''.format(i, pr_auc[i]))

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc="lower right")
    plt.show()

    # Compute sensitivity and specificity
    sensitivity = tpr[1]
    specificity = 1 - fpr[1]
    print("Sensitivity: " + str(sensitivity))
    print("Specificity: " + str(specificity))


if __name__ == '__main__':
    rootdir = r'D:\ECGData\mit-bih-arrhythmia-database-1.0.0\mit-bih-arrhythmia-database-1.0.0'
    acceptable_annotations = ['N', 'L', 'R', 'V', 'A', '/']
    # N: Non-ectopic peaks
    # L: Left bundle branch block beat
    # R: Right bundle branch block beat
    # V: Premature ventricular contraction
    # A: Atrial premature contraction
    # /: Paced beat

    '''
    This code is required for the first run
    '''
    # deal_data_to_CSV(rootdir)
    '''
    This code is required for the first run
    '''

    # Read the data from the CSV files
    X_df = pd.read_csv('X_mitdb_MLII.csv')
    y_df = pd.read_csv('Y_mitdb_MLII.csv')

    # Convert the dataframes to numpy arrays
    segmented_beats = X_df.values
    segmented_labels = y_df.values
    print(segmented_beats.shape, segmented_labels.shape)

    entire_X = np.array(segmented_beats)  # float64 (110084,259)
    entire_y = np.array(segmented_labels)  # str (110084,1)

    smote = SMOTE()
    # Set your target sample sizes
    num_beats_per_class = 10000
    # Setup up samplers
    over_sampler = SMOTE(sampling_strategy={key: num_beats_per_class for key in ['L', 'R', 'V', 'A', '/']})
    under_sampler = RandomUnderSampler(sampling_strategy={key: num_beats_per_class for key in ['N']})
    # Define pipeline
    pipeline = Pipeline([('O', over_sampler), ('U', under_sampler)])
    # Apply the samplers through the pipeline
    X_resampled, y_resampled = pipeline.fit_resample(entire_X, entire_y)

    print(X_resampled.shape, y_resampled.shape, y_resampled)

    labelMap = {'N': 0, 'L': 1, 'R': 2, 'V': 3, 'A': 4, '/': 5}

    # Map the labels to integers based on labelMap
    y_resampled = np.vectorize(labelMap.get)(y_resampled)

    print(X_resampled.shape, y_resampled.shape, y_resampled)

    X = X_resampled  # (60000,259)
    y = y_resampled  # (60000,)
    num_classes = 6
    input_shape = (259, 1)

    # split the dataset into training (70% 42000), validation (15% 9000), and testing sets (15% 9000)
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    # Convert the labels to one-hot encoding
    y_train = to_categorical(y_train, num_classes=num_classes)
    y_val = to_categorical(y_val, num_classes=num_classes)
    y_test = to_categorical(y_test, num_classes=num_classes)

    # Reshape the input data
    X_train = X_train.reshape(-1, 259, 1)
    X_val = X_val.reshape(-1, 259, 1)
    X_test = X_test.reshape(-1, 259, 1)

    model = ECANet(input_shape, num_classes)
    # model = sampleone()

    # compile the model
    model.compile(optimizer=optimizers.Adam(learning_rate=0.0004),
                  loss=tf.keras.losses.CategoricalCrossentropy(),
                  metrics=['accuracy'])

    model.summary()

    best_weights_save_load_dir = "PrateekSingh_adam_best_weights=0.9874.h5"

    # Define the checkpoint callback
    checkpoint = ModelCheckpoint(best_weights_save_load_dir, monitor='val_accuracy', save_best_only=True,
                                 mode='max', verbose=1)

    # Define the Early Stopping callback
    early_stopping = EarlyStopping(monitor='val_loss', mode='min', patience=10, verbose=1)

    # Train the model with both callbacks
    history = model.fit(X_train, y_train, epochs=100, validation_data=(X_val, y_val), verbose=1, batch_size=64,
                        callbacks=[checkpoint, early_stopping])

    ShowHistory(history)

    model.load_weights(best_weights_save_load_dir)

    evaluateModel(model, X_test, y_test)
