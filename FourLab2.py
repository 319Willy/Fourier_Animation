from manimlib.imports import*
import numpy as np
import scipy.fftpack
import scipy.integrate
from scipy import fftpack
import functions
from manimlib.mobject.FourierLib import*
#Fourier = Fourier()
USE_ALMOST_FOURIER_BY_DEFAULT = True
#The used Number of sample is located inside
#The FourierLib.py file
NUM_SAMPLES_FOR_FFT = 1000
DEFAULT_COMPLEX_TO_REAL_FUNC = lambda z : z.real

#Building Objected to be animated in the
#The Following Imagery Scene
class ImagedCreation(GraphScene):

    CONFIG = {
        "time_axes_config" : {
            "x_min" : 0,
            "x_max" : 4.4,
            "x_axis_config" : {
                "unit_size" : 3,
                "tick_frequency" : 0.25,
                "numbers_with_elongated_ticks" : [1, 2, 3],
            },
            "y_min" : 0,
            "y_max" : 2,
            "y_axis_config" : {"unit_size" : 0.8},
        },
        "time_label_t" : 3.4,
        "circle_plane_config" : {
            "x_radius" : 2.1,
            "y_radius" : 2.1,
            "x_unit_size" : 1,
            "y_unit_size" : 1,
        },
        "frequency_axes_config" : {
            "number_line_config" : {
                "color" : TEAL,
            },
            "x_min" : 0,
            "x_max" : 13.0,
            "x_axis_config" : {
                "unit_size" : 1,#1.4,
                "numbers_to_show" : list(range(0, 13,2)),
            },
            "y_min" : -7.0,
            "y_max" : 7.0,
            "y_axis_config" : {
                "unit_size" :0.5,#1.8,
                "tick_frequency" : 1,
                "label_direction" : LEFT,
            },
            "color" : TEAL,
        },
        "frequency_axes_box_color" : TEAL_E,
        "text_scale_val" : 0.75,
        "default_graph_config" : {
            "num_graph_points" : 100,
            "color" : YELLOW,
        },
        "equilibrium_height" : 1,
        "default_y_vector_animation_config" : {
            "run_time" : 5,
            "rate_func" : None,
            "remover" : True,
        },
        "default_time_sweep_config" : {
            "rate_func" : None,
            "run_time" : 5,
        },
        "default_num_v_lines_indicating_periods" : 20,
    }


    #Fourier = Fourier()
    def construct(self):

        frequency_axes = Fourier.get_frequency_axes(self)
        func = lambda t: -7*np.cos(2*PI*6*t)+6*np.sin(2*PI*2*t)+4*np.cos(2*PI*9*t)
        result = Fourier.get_fourier_graph(self,self.frequency_axes, func, 0, 15)
        full_graph = VGroup(frequency_axes,result).to_edge(LEFT,buff = 0)
        self.add(full_graph)


class VidCreation(GraphScene):
    CONFIG = {
        "time_axes_config" : {
            "x_min" : 0,
            "x_max" : 4.4,
            "x_axis_config" : {
                "unit_size" : 3,
                "tick_frequency" : 0.25,
                "numbers_with_elongated_ticks" : [1, 2, 3],
            },
            "y_min" : 0,
            "y_max" : 2,
            "y_axis_config" : {"unit_size" : 0.8},
        },
        "time_label_t" : 3.4,
        "circle_plane_config" : {
            "x_radius" : 2.1,
            "y_radius" : 2.1,
            "x_unit_size" : 1,
            "y_unit_size" : 1,
        },
        "frequency_axes_config" : {
            "number_line_config" : {
                "color" : TEAL,
            },
            "x_min" : 0,
            "x_max" : 13.0,
            "x_axis_config" : {
                "unit_size" : 1,#1.4,
                "numbers_to_show" : list(range(0, 13,2)),
            },
            "y_min" : -7.0,
            "y_max" : 7.0,
            "y_axis_config" : {
                "unit_size" :0.5,#1.8,
                "tick_frequency" : 1,
                "label_direction" : LEFT,
            },
            "color" : TEAL,
        },
        "frequency_axes_box_color" : TEAL_E,
        "text_scale_val" : 0.75,
        "default_graph_config" : {
            "num_graph_points" : 100,
            "color" : YELLOW,
        },
        "equilibrium_height" : 1,
        "default_y_vector_animation_config" : {
            "run_time" : 5,
            "rate_func" : None,
            "remover" : True,
        },
        "default_time_sweep_config" : {
            "rate_func" : None,
            "run_time" : 5,
        },
        "default_num_v_lines_indicating_periods" : 20,
    }

    def construct(self):
        frequency_axes = Fourier.get_frequency_axes(self)
        func = lambda t: -7*np.cos(2*PI*6*t)+6*np.sin(2*PI*2*t)+4*np.cos(2*PI*9*t)
        result = Fourier.get_fourier_graph(self,self.frequency_axes, func, 0, 15)
        #full_graph = VGroup(frequency_axes,result).to_edge(LEFT,buff = 0)

        self.play(ShowCreation(frequency_axes))
        self.wait()
        self.play(ShowCreation(result),run_time = 12)
