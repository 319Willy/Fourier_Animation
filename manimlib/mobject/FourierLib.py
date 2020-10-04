from manimlib.imports import*
from manimlib.constants import *
from manimlib.mobject.types.vectorized_mobject import VMobject
from manimlib.utils.config_ops import digest_config
import math

USE_ALMOST_FOURIER_BY_DEFAULT = True
NUM_SAMPLES_FOR_FFT = 1000
DEFAULT_COMPLEX_TO_REAL_FUNC = lambda z : z.real

class Fourier(GraphScene):

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
            "x_max" : 10.0,
            "x_axis_config" : {
                "unit_size" : 1,#1.4,
                "numbers_to_show" : list(range(1, 11)),
            },
            "y_min" : -5.0,
            "y_max" : 5.0,
            "y_axis_config" : {
                "unit_size" : 0.7,#1.8,
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
        "self.mean":0,
        "self.variance":0
    }


    def get_fourier_graph(self,
    axes, time_func, t_min, t_max,
    n_samples = NUM_SAMPLES_FOR_FFT,
    #complex_to_real_func = lambda z : z.real,
    #I changed this function to obtain the magnitude of
    #the complex number rather than just the real part
    complex_to_real_func = lambda z : abs(z),
    color = RED,
    ):
        # N = n_samples
        # T = time_range/n_samples
        #noise = np.random.normal(self.mean,self.variance,NUM_SAMPLES_FOR_FFT)
        time_range = float(t_max - t_min)
        t_range = t_max - t_min
        time_step_size = time_range/n_samples
        #norm_t_max = n_samples/(2*t_max)
        time_samples = np.vectorize(time_func)(np.linspace(t_min, t_max, n_samples))
        fft_output = np.fft.fft(time_samples)
        frequencies = np.linspace(0.0, n_samples/(2*time_range) , n_samples//2)
        #  #Cycles per second of fouier_samples[1]
        # (1/time_range)*n_samples
        # freq_step_size = 1./time_range
        graph = VMobject()
        graph.set_points_smoothly([
            axes.coords_to_point(
                x, 2*complex_to_real_func(y)/(n_samples),
            )
            #[ORIGINAL] for x, y in zip(frequencies, fft_output[:n_samples//57])
            for x, y in zip(frequencies, fft_output[:n_samples//(20//t_range)]) #Use this number to manipulte the length
            
        ])
        graph.set_color(color)
        f_min, f_max = [
            axes.x_axis.point_to_number(graph.points[i])
            for i in (0, -1)
        ]
        graph.underlying_function = lambda f : axes.y_axis.point_to_number(
            graph.point_from_proportion((f - f_min)/(f_max - f_min))
        )
        return graph



    def get_frequency_axes(self):
        frequency_axes = Axes(**self.frequency_axes_config)
        frequency_axes.x_axis.add_numbers()
        frequency_axes.y_axis.add_numbers(
            *frequency_axes.y_axis.get_tick_numbers()
        )
        box = SurroundingRectangle(
            frequency_axes,
            buff = MED_SMALL_BUFF,
            color = self.frequency_axes_box_color,
        )
        frequency_axes.box = box
        #frequency_axes.add(box)
        frequency_axes.to_corner(DOWN+RIGHT, buff = MED_SMALL_BUFF)

        frequency_label = TextMobject("Frequency")
        frequency_label.scale(self.text_scale_val)
        frequency_label.next_to(
            frequency_axes.x_axis.get_right(), DOWN,
            buff = MED_LARGE_BUFF,
            aligned_edge = RIGHT,
        )
        frequency_axes.label = frequency_label
        frequency_axes.add(frequency_label)

        self.frequency_axes = frequency_axes
        return frequency_axes



    def get_circle_plane(self):
        circle_plane = NumberPlane(**self.circle_plane_config)
        circle_plane.to_corner(DOWN+LEFT)
        circle = DashedLine(ORIGIN, TAU*UP).apply_complex_function(np.exp)
        circle.scale(circle_plane.x_unit_size)
        circle.move_to(circle_plane.coords_to_point(0, 0))
        circle_plane.circle = circle
        circle_plane.add(circle)
        circle_plane.fade()
        self.circle_plane = circle_plane
        return circle_plane
