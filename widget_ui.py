from ipywidgets import *
from IPython.display import display
from sklearn.neural_network import MLPClassifier
import data_utils
import preprocessing

def subsampling_ui(all_data, output_data):
    sampling_rate = BoundedFloatText(
        value=0.0001,
        min=0,
        max=1.0,
        description='Sampling rate:'
    )

    i = interact(data_utils.subsample_data, 
        data=fixed(all_data), 
        r=sampling_rate, 
        output_data=fixed(output_data))

def pca_ui(data):
    ncomponents = IntSlider(
        value=5,
        min=1,
        max=len(data.train_data.columns),
        step=1,
        description='Num components:',
        continuous_update=False,
        layout=Layout(width='80%')
    )

    i = interactive(preprocessing.perform_pca, 
        data=fixed(data.train_data), 
        n=ncomponents)
    display(i)

    # Wrap resulting transformation in a function so that it is only
    # evaluated in the end, because there may be intermediate results.
    def init_pca():
        return i.result

    return init_pca

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

def mlp_ui():
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
    i = interact(set_max_iter, 
        m=max_iter_slider,
        max_iter=fixed(max_iter))

    nhidden = IntSlider(
        value=2,
        min=1,
        max=10,
        step=1,
        description='Num hidden layers:',
        continuous_update=False,
        layout=Layout(width='80%')
    )

    i = interact(configure_hidden_layers, 
        n=nhidden,
        hidden_layer_box=fixed(hidden_layer_box),
        hidden_layers=fixed(hidden_layers))

    def init_mlp():
        mlp = MLPClassifier(hidden_layer_sizes=hidden_layers, 
                            max_iter=max_iter[0], 
                            warm_start=True)
        mlp.train = mlp.fit
        return mlp

    return init_mlp

def nb_ui():
    def init_nb():
        naiveBayes = GaussianNB(priors=[0.5, 0.5])
        naiveBayes.train = lambda x, y: naiveBayes.partial_fit(x, y, classes=[1,0])
        return naiveBayes

    return init_nb
