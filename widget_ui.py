from ipywidgets import *
from IPython.display import display
import numpy as np
import time
import datetime
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score ,classification_report
import sklearn.decomposition as skdecomp
import data_utils


#### Data Preparation ####

def subsampling_ui(all_data, output_data):
    sampling_rate = BoundedFloatText(
        value=0.0001,
        min=0,
        max=1.0,
        description='Sampling rate:'
    )
    i = interact(
        data_utils.subsample_data, 
        data=fixed(all_data), 
        r=sampling_rate, 
        output_data=fixed(output_data)
    )


#### Preprocessing ####

def remove_correlations(preprocessors, corr, thres):
    remove = set()
    for col1 in range(1, len(corr)):
        for col2 in range(col1 + 1, len(corr)):
            if corr.values[col1][col2] > thres:
                remove.add(corr.columns[col1])
                break
    if remove:
        print('Features to be removed: {}.'.format(', '.join(remove)))
    else:
        print('For this threshold, no feature will be removed.')

    preprocessors['corr'] = remove

def correlations_ui(preprocessors, data):
    thres = FloatSlider(
        value=0.7,
        min=0,
        max=1,
        step=0.05,
        description='Threshold:',
        continuous_update=False,
        layout=Layout(width='80%')
    )
    i = interactive(
        remove_correlations, 
        preprocessors=fixed(preprocessors),
        corr=fixed(data.train_data.corr().compute()), 
        thres=thres
    )
    display(i)

def perform_pca(preprocessors, data, n):
    pca = skdecomp.PCA(n_components=n)
    features = data.drop('# label', axis=1).compute()
    pca.fit(features)
    print('PCA with {} principal components computed.'.format(n))

    preprocessors['pca'] = pca

def pca_ui(preprocessors, data):
    ncomponents = IntSlider(
        value=5,
        min=1,
        max=len(data.train_data.columns),
        step=1,
        description='Num components:',
        continuous_update=False,
        layout=Layout(width='80%')
    )
    i = interactive(
        perform_pca, 
        preprocessors=fixed(preprocessors),
        data=fixed(data.train_data), 
        n=ncomponents
    )
    display(i)


#### Classifiers ####

def set_max_iter(m, max_iter):
    max_iter[0] = m

def update_hidden_layer_size(hidden_layers, size, i):
    hidden_layers[i] = size

def configure_hidden_layers(n, hidden_layer_box, hidden_layers):
    if hidden_layer_box[0] is not None:
        hidden_layer_box[0].close()
    
    hidden_layers.clear()
    for i in range(n):
        hidden_layers.append(20)
    
    items = [interactive(update_hidden_layer_size,
                         hidden_layers=fixed(hidden_layers),
                         size=IntSlider(
                             value=20,
                             min=1,
                             max=100,
                             step=1,
                             description='Hidden layer {}:'.format(i),
                             continuous_update=False,
                             layout=Layout(width='80%')
                         ),
                         i=fixed(i))
             for i in range(n)]
    
    hidden_layer_box[0] = VBox(
        [widgets.Label('Hidden layer sizes:', layout=Layout(width='100%'))] + items,
        layout=Layout(border='solid', padding='10px')
    )
    display(hidden_layer_box[0])

def mlp_ui(classifiers):
    hidden_layer_box = [None]
    hidden_layers = [20, 20]
    max_iter = [30]

    max_iter_slider = IntSlider(
        value=30,
        min=10,
        max=500,
        step=10,
        description='Max iterations:',
        continuous_update=False,
        layout=Layout(width='80%')
    )
    i = interact(
        set_max_iter, 
        m=max_iter_slider,
        max_iter=fixed(max_iter)
    )

    nhidden = IntSlider(
        value=2,
        min=1,
        max=10,
        step=1,
        description='Num hidden layers:',
        continuous_update=False,
        layout=Layout(width='80%')
    )
    i = interact(
        configure_hidden_layers, 
        n=nhidden,
        hidden_layer_box=fixed(hidden_layer_box),
        hidden_layers=fixed(hidden_layers)
    )

    def init_mlp():
        mlp = MLPClassifier(hidden_layer_sizes=hidden_layers, 
                            max_iter=max_iter[0], 
                            warm_start=True)
        mlp.train = mlp.fit
        return mlp

    classifiers['mlp'] = {'init': init_mlp}

def nb_ui(classifiers):
    def init_nb():
        naiveBayes = GaussianNB(priors=[0.5, 0.5])
        naiveBayes.train = lambda x, y: naiveBayes.partial_fit(x, y, classes=[1,0])
        return naiveBayes

    classifiers['nb'] = {'init': init_nb}


#### Training ####

def train_on_window(preprocessors, apply_pca, classifier, window):
    labels = window[:, 0]
    features = window[:, 1:29]
    
    # Optionally apply PCA
    if apply_pca:
        features = preprocessors['pca'].transform(features)
    
    # Train classifier
    classifier.fit(features, labels)


def perform_training(
    data, preprocessors, apply_pca, window_size, classifiers, 
    classifier_name):
    
    # Initialize classifier
    if classifier_name == 'Multilayer Perceptron':
        classifier = classifiers['mlp']['init']()
        classifiers['mlp']['trained'] = classifier
    else:
        classifier = classifiers['nb']['init']()
        classifiers['nb']['trained'] = classifier
    
    print('Training {}'.format(classifier))
    start_time = time.time()
    
    iterator = data.train_data.iterrows()
    window = np.zeros((0,29))
    
    progress = IntProgress(
        min=0,
        max=data.ntrain,
        step=1,
        description='Training:',
        bar_style='danger'
    )
    display(progress)
    
    for idx, row in enumerate(iterator):
        window = np.append(window, [row[1]], axis=0)
        if window.shape[0] == window_size:
            train_on_window(preprocessors, apply_pca, classifier, window)
            window = np.zeros((0,29))
            progress.value = idx
    if len(window) > 0:
        train_on_window(preprocessors, apply_pca, classifier, window)
        progress.value = data.ntrain

    print('Time taken: {}'.format(time.time() - start_time))

def training_ui(data, sample_data, preprocessors, classifiers):
    use_sample_data = Checkbox(
        value=False,
        description='Use fast sample data'
    )
    display(use_sample_data)

    pca_checkbox = Checkbox(
        value=False,
        description='Apply PCA'
    )
    display(pca_checkbox)

    window_size_slider = IntSlider(
        value=500,
        min=100,
        max=1000,
        step=100,
        description='window size:',
        continuous_update=False,
        layout=Layout(width='80%')
    )
    display(window_size_slider)

    classifier_rb = RadioButtons(
        options=['Multilayer Perceptron', 'Naive Bayes'],
        description='Classifier:'
    )
    display(classifier_rb)

    start_training = Button(
        description='Start training',
        button_style='danger'
    )
    display(start_training)

    def training_func(*args):
        if use_sample_data.value:
            d = sample_data
        else:
            d = data

        perform_training(
            data=d, 
            preprocessors=preprocessors, 
            apply_pca=pca_checkbox.value, 
            window_size=window_size_slider.value,
            classifiers=classifiers,
            classifier_name=classifier_rb.value)

    start_training.on_click(training_func)


#### Prediction ####

def predict_on_window(preprocessors, apply_pca, classifier, window):
    features = window[:, 1:29]
    
    # Optionally apply PCA
    if apply_pca:
        features = preprocessors['pca'].transform(features)
    
    # Predict
    prediction = classifier.predict(features)
    return prediction

def perform_prediction(
    data, preprocessors, apply_pca, window_size, to_save, classifiers, 
    classifier_name):
    
    # Load classifier
    if classifier_name == 'Multilayer Perceptron':
        if 'trained' not in classifiers['mlp']:
            print("No Multilayer Perceptron available for prediction.")
            return
        classifier = classifiers['mlp']['trained']
    else:
        if 'trained' not in classifiers['nb']:
            print("No Naive Bayes classifier available for prediction.")
            return
        classifier = classifiers['nb']['trained']

    print('Predict with {}'.format(classifier))
    
    iterator = data.test_data.iterrows()
    window = np.zeros((0,29))
    
    progress = IntProgress(
        min=0,
        max=data.ntest,
        step=1,
        description='Predicting:',
        bar_style='info'
    )
    display(progress)
    
    full_prediction = np.array([])

    for idx, row in enumerate(iterator):
        window = np.append(window, [row[1]], axis=0)
        if window.shape[0] == window_size:
            prediction = predict_on_window(
                preprocessors, apply_pca, classifier, window)
            full_prediction = np.append(full_prediction, prediction)
            window = np.zeros((0,29))
            progress.value = idx
    if len(window) > 0:
        prediction = predict_on_window(
            preprocessors, apply_pca, classifier, window)
        full_prediction = np.append(full_prediction, prediction)
        progress.value = data.ntest

    # save result to file
    if to_save:
        classifier_str = classifier_name.replace(' ', '')
        pca_str = preprocessors['pca'].n_components_ if apply_pca else 'False'
        sample_str = str(data.sample_rate).replace('.', '')
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        np.save("results/{}_pca_{}_sample_{}_time_{}".format(
            classifier_str, pca_str, sample_str, timestamp), full_prediction)

    # give early evaluation
    print("Accuracy: {:0.2f}".format(accuracy_score(
            data.test_data['# label'], full_prediction)))
    print(classification_report(data.test_data['# label'], full_prediction))

def prediction_ui(data, sample_data, preprocessors, classifiers):
    use_sample_data = Checkbox(
        value=False,
        description='Use fast sample data'
    )
    display(use_sample_data)

    pca_checkbox = Checkbox(
        value=False,
        description='Apply PCA'
    )
    display(pca_checkbox)

    window_size_slider = IntSlider(
        value=500,
        min=100,
        max=1000,
        step=100,
        description='window size:',
        continuous_update=False,
        layout=Layout(width='80%')
    )
    display(window_size_slider)

    classifier_rb = RadioButtons(
        options=['Multilayer Perceptron', 'Naive Bayes'],
        description='Classifier:'
    )
    display(classifier_rb)

    save_checkbox = Checkbox(
        value=False,
        description='Save Prediction to File'
    )
    display(save_checkbox)

    start_prediction = Button(
        description='Start prediction',
        button_style='info'
    )
    display(start_prediction)

    def prediction_func(*args):
        if use_sample_data.value:
            d = sample_data
        else:
            d = data

        perform_prediction(
            data=d, 
            preprocessors=preprocessors, 
            apply_pca=pca_checkbox.value, 
            window_size=window_size_slider.value,
            to_save = save_checkbox.value,
            classifiers=classifiers,
            classifier_name=classifier_rb.value)

    start_prediction.on_click(prediction_func)
