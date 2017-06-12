from ipywidgets import *
from IPython.display import display
import numpy as np
import pandas as pd
import time
import datetime
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix
import sklearn.decomposition as skdecomp
import data_utils
from os.path import isfile


#### Data Preparation ####

def subsampling_ui(orig_data, output_data):
    sampling_rate = BoundedFloatText(
        value=0.0001,
        min=0,
        max=1.0,
        description='Sampling rate:'
    )
    i = interact(
        data_utils.subsample_data, 
        data=fixed(orig_data), 
        r=sampling_rate, 
        output_data=fixed(output_data)
    )


#### Preprocessing ####

def remove_correlations(preprocessors, data, thres, output):
    corr = data.corr().compute()
    remove = []
    for col1 in range(1, len(corr)):
        for col2 in range(col1 + 1, len(corr)):
            if corr.values[col1][col2] > thres:
                remove.append(corr.columns[col1])
                break
    if remove:
        if output:
            print('Features to be removed: {}.'.format(', '.join(remove)))
        preprocessors['corr'] = remove
    else:
        if output:
            print('For this threshold, no feature will be removed.')

def perform_pca(preprocessors, data, n, output):
    pca = skdecomp.PCA(n_components=n)
    features = data.drop('# label', axis=1).compute()
    pca.fit(features)
    if output:
        print('A PCA with {} pricipal components will be applied.'.format(n))
    preprocessors['pca'] = pca
    
def enable_preprocessing(selected, thres_interactive, ncomp_interactive, preprocessors, data):
    if selected == 'no preprocessing':
        # hide sliders
        thres_interactive.layout = Layout(display='none')
        ncomp_interactive.layout = Layout(display='none')

        # remove existing Pearson correlated features
        if 'corr' in preprocessors:
            del preprocessors['corr']
        # remove existing PCA
        if 'pca' in preprocessors:
            del preprocessors['pca']
    elif selected == 'Pearson correlation coefficient':
        # hide PCA slider
        ncomp_interactive.layout = Layout(display='none')
        # remove existing PCA
        if 'pca' in preprocessors:
            del preprocessors['pca']
            
        # show Pearson correlation coefficient slider
        thres_interactive.layout = Layout(display='block')
        # calculate correlations
        thres = thres_interactive.children[0].value
        remove_correlations(preprocessors, data, thres, output=False)
    elif selected == 'PCA':
        # hide Pearson correlation coefficient slider
        thres_interactive.layout = Layout(display='none')
        # remove existing Pearson correlated features
        if 'corr' in preprocessors:
            del preprocessors['corr']
            
        # show slider
        ncomp_interactive.layout = Layout(display='block')
        # perform pca
        n = ncomp_interactive.children[0].value
        perform_pca(preprocessors, data, n, output=False)
    
def preprocessors_ui(preprocessors, data):
    # Pearson correlation coefficient - threshold
    thres_slider = FloatSlider(
        value=0.7,
        min=0,
        max=1,
        step=0.05,
        description='Threshold:',
        continuous_update=False,
        layout=Layout(width='80%')
    )
    thres_interactive = interactive(
        remove_correlations, 
        preprocessors=fixed(preprocessors),
        data=fixed(data.train_data), 
        thres=thres_slider,
        output=fixed(True)
    )
    
    # PCA - number of principal components
    ncomp_slider = IntSlider(
        value=5,
        min=0,
        max=len(data.train_data.columns) - 1,
        step=1,
        description='Num components:',
        continuous_update=False,
        layout=Layout(width='80%')
    )
    ncomp_interactive = interactive(
        perform_pca, 
        preprocessors=fixed(preprocessors),
        data=fixed(data.train_data), 
        n=ncomp_slider,
        output=fixed(True)
    )
    
    # selection of preprocessing method
    select = widgets.Dropdown(
        options=['no preprocessing', 'Pearson correlation coefficient', 'PCA'],
        value='no preprocessing',
        description='Preprocessing:',
        disabled=False
    )
    
    i = interact(
        enable_preprocessing,
        selected=select,
        thres_interactive=fixed(thres_interactive),
        ncomp_interactive=fixed(ncomp_interactive),
        preprocessors=fixed(preprocessors),
        data=fixed(data.train_data)
    )
    
    display(thres_interactive)
    display(ncomp_interactive)

    # remove preprocessors from initialization
    if 'pca' in preprocessors:
        del preprocessors['pca']
    if 'corr' in preprocessors:
        del preprocessors['corr']


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

    classifiers['mlp'] = {'init': init_mlp, 'hl': hidden_layers}

def nb_ui(classifiers):
    def init_nb():
        naiveBayes = GaussianNB(priors=[0.5, 0.5])
        naiveBayes.train = lambda x, y: naiveBayes.partial_fit(x, y, classes=[1,0])
        return naiveBayes

    classifiers['nb'] = {'init': init_nb}


#### Training ####

def train_on_window(preprocessors, classifier, window):                             
    labels = window["# label"]
    features = window[list(window.columns[1:])]                        
    
    # Optionally remove Pearson correlated features
    if 'corr' in preprocessors:
        features = features.drop(preprocessors['corr'], axis=1)
    # Optionally apply PCA
    if 'pca' in preprocessors:
        features = preprocessors['pca'].transform(features)
    
    # Train classifier
    classifier.fit(features, labels)

def perform_training(
    data, preprocessors, window_size, classifiers, classifier_name):
    
    # Initialize classifier
    if classifier_name == 'Multilayer Perceptron':
        classifier = classifiers['mlp']['init']()
        classifiers['mlp']['trained'] = classifier
    else:
        classifier = classifiers['nb']['init']()
        classifiers['nb']['trained'] = classifier
    
    print('Training {}'.format(classifier))
    start_time = time.time()
    
    progress = IntProgress(
        min=0,
        max=data.ntrain,
        step=1,
        description='Training:',
        bar_style='danger'
    )
    display(progress)
    
    # process data in windows
    iterator = data.train_data.iterrows()
    window = np.zeros((0, len(data.train_data.columns)))
    
    for idx, row in enumerate(iterator):
        window = np.append(window, [row[1]], axis=0)
        if len(window) == window_size:
            train_on_window(preprocessors, classifier, pd.DataFrame(window, columns=data.train_data.columns))
            window = np.zeros((0, len(data.train_data.columns)))
            progress.value = idx
    if len(window) > 0:
        train_on_window(preprocessors, classifier, pd.DataFrame(window, columns=data.train_data.columns))
        progress.value = data.ntrain

    print('Time taken: {}'.format(time.time() - start_time))

def training_ui(data, sample_data, preprocessors, classifiers):
    use_sample_data = Checkbox(
        value=False,
        description='Use fast sample data'
    )
    display(use_sample_data)

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
        perform_training(
            data=sample_data if use_sample_data.value else data, 
            preprocessors=preprocessors,
            window_size=window_size_slider.value,
            classifiers=classifiers,
            classifier_name=classifier_rb.value)

    start_training.on_click(training_func)


#### Prediction ####

def predict_on_window(preprocessors, classifier, window):
    features = window[list(window.columns[1:])] 
    
    # Optionally remove Pearson correlated features
    if 'corr' in preprocessors:
        features = features.drop(preprocessors['corr'], axis=1)
    # Optionally apply PCA
    if 'pca' in preprocessors:
        features = preprocessors['pca'].transform(features)
    
    # Predict
    prediction = classifier.predict(features)
    return prediction

def perform_prediction(
    data, preprocessors, window_size, to_save, classifiers, classifier_name):
    
    # load classifier
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
        
    progress = IntProgress(
        min=0,
        max=data.ntest,
        step=1,
        description='Predicting:',
        bar_style='info'
    )
    display(progress)
    
    # process data in windows
    iterator = data.test_data.iterrows()
    window = np.zeros((0, len(data.test_data.columns)))
    conf_mat = np.zeros((2, 2))

    for idx, row in enumerate(iterator):
        window = np.append(window, [row[1]], axis=0)
        if window.shape[0] == window_size:
            prediction = predict_on_window(preprocessors, classifier, pd.DataFrame(window, columns=data.train_data.columns))
            conf_mat += confusion_matrix(window[:,0], prediction)
            window = np.zeros((0, len(data.test_data.columns)))
            progress.value = idx
    if len(window) > 0:
        prediction = predict_on_window(preprocessors, classifier, pd.DataFrame(window, columns=data.train_data.columns))
        conf_mat += confusion_matrix(window[:,0], prediction)
        progress.value = data.ntest

    # save result to file
    if to_save:
        classifier_str = classifier_name.replace(' ', '')
        pca_str = preprocessors['pca'].n_components_ \
                  if 'pca' in preprocessors else 'False'
        corr_str = '_'.join(preprocessors['corr']) \
                  if 'corr' in preprocessors else 'False'
        if classifier_name == 'Naive Bayes':
            classifier_str = "NB"
        else:
            classifier_str = "MLP_{}".format("_".join([str(i) for i in classifiers['mlp']['hl']]))

        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        np.save("results/{}_pca_{}_corr_{}_time_{}".format(
            classifier_str, pca_str, corr_str, timestamp), conf_mat)

    print(conf_mat)
    # give early evaluation
    accuracy = (conf_mat[0,0] + conf_mat[1,1]) / np.sum(conf_mat)
    print("Accuracy: {:0.4f}".format(accuracy))

def prediction_ui(data, sample_data, preprocessors, classifiers):
    use_sample_data = Checkbox(
        value=False,
        description='Use fast sample data'
    )
    display(use_sample_data)

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
        perform_prediction(
            data=sample_data if use_sample_data.value else data, 
            preprocessors=preprocessors,
            window_size=window_size_slider.value,
            to_save = save_checkbox.value,
            classifiers=classifiers,
            classifier_name=classifier_rb.value)

    start_prediction.on_click(prediction_func)
