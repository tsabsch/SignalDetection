from ipywidgets import *
from IPython.display import display
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
        description='num components:',
        continuous_update=False,
        layout=Layout(width='80%')
    )

    i = interactive(preprocessing.perform_pca, 
        data=fixed(data.train_data), 
        n=ncomponents)
    display(i)

    def pca_func(features):
        return i.result.transform(features)

    return pca_func
