import numpy
import pandas
import keras
from keras.models import Sequential
from keras.layers import *
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.preprocessing import LabelEncoder

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix
from sklearn.metrics import make_scorer
from keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler
from keras.callbacks import TensorBoard


# Create a TensorBoard logger
logger = TensorBoard(
    log_dir='logs',
    histogram_freq=0,
    write_graph=True
)


# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)


# it load caravan data and return the X and Y
def load_data(filename):
    data_frame = pandas.read_csv(filename)
    data_set = data_frame.values
    # set the input columns
    x = data_set[:, 9:].astype(float)
    # Set the label column
    y = data_set[:, 1]
    print(y)
    print(x)
    encoder = LabelEncoder()
    encoder.fit(y)
    encoded_y = encoder.transform(y)
    # print(X.shape)
    # print(Y.shape)
    return x, encoded_y


X, encoded_Y = load_data('C:\\Users\\voghoei\\OneDriveSchool\\OneDrive - University of Georgia\\Uni\\PHD Project\\SNP\\B_Dataset_9-5.csv')


def create_model():
    # create model
    model = Sequential()
    model.add(Dense(50, input_dim=740, kernel_initializer='normal', activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(500, kernel_initializer='normal', activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(1000, kernel_initializer='normal', activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(100, kernel_initializer='normal', activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(1, kernel_initializer='normal', activation='sigmoid'))

    # Compile model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


early_stopping = EarlyStopping(monitor='loss', min_delta=1e-3, patience=5, verbose=1, mode='auto')

# evaluate model with standardized dataset
estimators = []
estimators.append(('MinMaxScale', MinMaxScaler()))
estimators.append(('standardize', StandardScaler()))

estimators.append(('mlp', KerasClassifier(build_fn=create_model, epochs=1, batch_size=1, verbose=1)))
pipeline = Pipeline(estimators)
k_fold = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)


def tp(y_true, y_pred): return confusion_matrix(y_true, y_pred)[0, 0]
def tn(y_true, y_pred): return confusion_matrix(y_true, y_pred)[0, 0]
def fp(y_true, y_pred): return confusion_matrix(y_true, y_pred)[1, 0]
def fn(y_true, y_pred): return confusion_matrix(y_true, y_pred)[0, 1]


scoring = {'precision': 'precision', 'f1': 'f1_weighted', 'accuracy': 'accuracy', 'recall': 'recall_weighted',
           'tp': make_scorer(tp), 'tn': make_scorer(tn), 'fp': make_scorer(fp), 'fn': make_scorer(fn)}

results = cross_validate(pipeline, X, encoded_Y, scoring=scoring, cv=k_fold,  return_train_score=False ,fit_params={'mlp__callbacks':[early_stopping,logger]})

print("----------------------   The Folds Results --------------------")
print("accuracy = "+str(results['test_accuracy']))
print("recall = "+str(results['test_recall']))
print("precision = "+str(results['test_precision']))
print("f1 = "+str(results['test_f1']))
print("fp = "+str(results['test_fp']))
print("tp = "+str(results['test_tp']))
print("tn = "+str(results['test_tn']))
print("fn = "+str(results['test_fn']))
print("----------------------   The Overal Results --------------------")
print("accuracy: %.2f%% (%.2f%%)" % (results['test_accuracy'].mean()*100, results['test_accuracy'].std()*100))
print("recall = %.2f%% (%.2f%%)" % (results['test_recall'].mean()*100, results['test_recall'].std()*100))
print("precision %.2f%% (%.2f%%)" % (results['test_precision'].mean()*100, results['test_precision'].std()*100))
print("f1 %.2f%% (%.2f%%)" % (results['test_precision'].mean()*100, results['test_f1'].std()*100))
print("tp = %.0f " % (results['test_tp'].mean()))
print("tn = %.0f " % (results['test_tn'].mean()))
print("fp = %.0f " % (results['test_fp'].mean()))
print("fn = %.0f " % (results['test_fn'].mean()))
