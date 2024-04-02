---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.16.1
  kernelspec:
    display_name: qudi
    language: python
    name: qudi
---

```python
import lmfit
import numpy as np
import time
from collections import OrderedDict
import matplotlib.pyplot as plt
```

```python
from qudi.util.datastorage import TextDataStorage
from qudi.util.paths import get_default_data_dir
```

```python
def polar_model(params, theta, data):
        model = params['floor'] + params['amplitude'] * np.cos(theta-params['phi'])**2
        return model - data 

polar_params = lmfit.Parameters()
polar_params.add('floor', value=1e3, vary=True)
polar_params.add('amplitude', value=20e3, min=0, vary=True)
polar_params.add('phi', value=np.pi/4, vary=True)
```

```python
def acquire_diagram(nb_points, background, background_time, time_per_point=1, axis='phi', reset=True, margin=1.4):
    
    x_axis = np.linspace(0, 180, num=nb_points)
    data = np.zeros(nb_points)
    
    # First go to 0
    pos = rotation.get_position(axis)
    velocity = rotation.get_velocity(axis)
    reset_time = np.abs(pos) / velocity
    
    rotation.set_position(axis, 0)
    
    time.sleep(reset_time)
    time.sleep(0.5)
    
    # Measure
    for j, x in enumerate(x_axis):
        if stop_scripts:
            break
        rotation.set_position(axis, x)
        
        time.sleep(180 / nb_points / velocity + margin)
        
        data[j] = get_pl(duration=time_per_point)
        
        print(x)
    if reset:
        rotation.set_position(axis, 0)
        
    return analyse_polar_data(x_axis, data, background, background_time, time_per_point)
```

```python
def analyse_polar_data(x_axis, data, background, background_time, time_per_point):

    data_to_save = OrderedDict()
    data_to_save['x'] = (2 * np.pi * x_axis /360)*2
    data_to_save['y_raw'] = data.T
    data_to_save['y_data'] = data.T - background
    
    parameters = OrderedDict()
    parameters['time_per_point'] = time_per_point
    parameters['background'] = background
    parameters['background_time'] = background_time
    parameters['background_err'] = np.sqrt(background*background_time)/background_time
    
    data_to_save['y_raw_err'] = np.sqrt(data_to_save['y_data']*time_per_point)/time_per_point
    data_to_save['y_data_err'] = np.sqrt(data_to_save['y_raw_err']**2 + parameters['background_err']**2)
    
    # Fit
    polar_params['amplitude'].value = data_to_save['y_data'].max()
    result = lmfit.minimize(polar_model, polar_params, args=(data_to_save['x'], data_to_save['y_data']))
    
    for name, param in result.params.items():
        name, param.value, param.stderr
        parameters[name] = param.value
        parameters['{}_err'.format(name)] = param.stderr
    
    
    parameters['polar_ratio'] = result.params['amplitude'].value/ \
                                (result.params['amplitude'].value+result.params['floor'].value)
        
    return data_to_save, parameters
```

```python
def plot_polarization_diagram(data, parameters):
    fig = plt.figure(figsize=(6,6))
    ax = fig.add_subplot(111, polar=True)
    
    p=ax.errorbar(data['x'], y=data['y_data'], yerr=data['y_data_err'], color='blue', 
            marker='o', linestyle="None", markersize=4, capsize=0, elinewidth=1, mec='k',
            mew=0.4, ecolor = 'k')

    ax.grid(True)
    fig = plt.gcf()
#     ax.set_title('{} polarization diagram'.format(poimanagerlogic.active_poi))
    
    x_fit = np.linspace(0, 2*np.pi, 1000)
    y_fit = polar_model(parameters, x_fit, 0)
    ax.plot(x_fit, y_fit, color='red', alpha=.1)

    text = ''
    text += 'Amplitude :  {:.3g}$\pm${:.2g} kcps \n'.format(parameters['amplitude']/1e3, parameters['amplitude_err']/1e3)
    text += 'Floor :  {:.3g}$\pm${:.2g} kcps \n'.format(parameters['floor']/1e3, parameters['floor_err']/1e3)
    text += 'Phi :  {:.3g}$\pm${:.2g} deg \n'.format(parameters['phi']/(2*np.pi)*360, parameters['phi_err']/(2*np.pi)*360)
#     text += 'Amplitude :  {:.3g}$\pm$ kcps \n'.format(parameters['amplitude']/1e3)
#     text += 'Floor :  {:.3g}$\pm$ kcps \n'.format(parameters['floor']/1e3)
#     text += 'Phi :  {:.3g}$\pm$ deg \n'.format(parameters['phi']/(2*np.pi)*360)
    text += 'Visibility :  {:.2g} %'.format(parameters['polar_ratio']*100)
    
    ax.text(x=1, y=.1, s=text, ha='left', va='bottom', transform=ax.transAxes)
    
    return fig, ax
```

```python
def save_polarization_diagram(data, parameters, fig=None):
    filepath = get_default_data_dir()
    
    storage = TextDataStorage(root_dir=rotation.module_default_data_dir)
    filepath, _, _ = storage.save_data(data, metadata=parameters)
    
    if fig is not None:
        storage.save_thumbnail(mpl_figure=fig, file_path=filepath)
```

```python
def get_pl(duration):
    counter_multiharp150._clock_frequency = 1/duration
    return counter_multiharp150._get_counter_hist()
```

```python
# filepath = get_default_data_dir()

# storage = TextDataStorage(root_dir=rotation.module_default_data_dir)
# filepath, _, _ = storage.save_data(data_polar, metadata=parameters_polar)
```

## Lambda/2 position

```python
rotation.home('phi')
```

```python
rotation.get_position('phi')
```

```python
rotation.get_velocity('phi')
```

```python
rotation.set_position('phi', 45)
```

# Background (current poi)

```python
background = 0
background_time = 5
```

```python
# background = get_pl(background_time)
# background
```

# Polarisation current poi

```python
stop_scripts = False
```

```python
axis = 'phi'
nb_points = 19
data_polar, parameters_polar = acquire_diagram(nb_points, background, background_time, time_per_point=1, axis=axis)
plt.show()
```

```python
fig, ax = plot_polarization_diagram(data_polar, parameters_polar)
```

```python
data_polar, parameters_polar 
```

```python
# rotation.module_default_data_dir
```

```python
#np.savetxt(rotation.module_default_data_dir+'\\fname', data_polar)
```

```python
save_polarization_diagram(data_polar, parameters_polar, fig=fig)
```

### Polarisation multiple poi

```python
# Warning : this does not work very well, do it by hand :)

# nb_points=180
# nb_repetition=1
# time_per_point=1
# ###
# regexp = 'Gustave_A_(1|2|3|4|5|6)$'
# # regexp = 'K(25)'

# generator = poi_automatic(regexp, no_z_focus=True)
# for poi in generator:
#     x, data = acquire_diagram(nb_points=nb_points, nb_repetition=nb_repetition, time_per_point=time_per_point)
#     drifter_scanner()
#     background = get_pl(5)
#     fig, ax = plot_polarization_diagram(x, data, background=background)
#     save_polarization_diagram(x, data, background=background, fig=fig)
```

```python

```

```python
# dummy 

x_axis = np.linspace(0, 180, num=50)
data = np.random.random(50)*3e3
background = 100
background_time = 5
time_per_point = 1
data_polar, parameters_polar = analyse_polar_data(x_axis, data, background, background_time, time_per_point)

fig, ax = plot_polarization_diagram(data_polar, parameters_polar)

plt.show()
```
