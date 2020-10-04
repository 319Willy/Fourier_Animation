#Ready For Branching
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
NOISE = lambda t:0.001*np.cos(2*np.pi*9*t)
#DEFAULT_COMPLEX_TO_REAL_FUNC = lambda z : z.real
#
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

    def construct(self):

        time_axes = Axes(
            x_min = 0, x_max = 10,
            y_min = -5, y_max = 5,

        number_line_config = {
            "tick_size" : 0.05,
        },
        x_axis_config = {
            "unit_size" : 2,
            "tick_frequency":1,
            "include_numbers":True,
            "numbers_to_show": np.arange(0,12,1)

            },
        y_axis_config = {
            "unit_size":0.5,
            "tick_frequency":1,
            "include_numbers":True,
            "numbers_to_show":np.arange(-5,5,1),
            "label_direction":UP,
            "include_tip": True
        })
        #Ampt-Frequency-Mean-Variance
        time_func =self.time_signal(1,1,0,0)
        time_graph = time_axes.get_graph(time_func,color = YELLOW)
        t_sig = VGroup(time_axes,time_graph).to_edge(LEFT,buff = 0)

        frequency_axes = Fourier.get_frequency_axes(self)
        freq_graph = Fourier.get_fourier_graph(self,self.frequency_axes, time_func, 0, 20)
        f_sig = VGroup(frequency_axes,freq_graph).to_edge(3*LEFT,buff = 0)

        self.add(t_sig)


    def time_signal(self,amp,freq,mean,var):
        return lambda t: amp*np.cos(2*PI*freq*t)+np.random.normal(mean,var)

class MoveAlongSpectra(GraphScene):
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
            "x_max" : 50.0,
            "x_axis_config" : {
                "unit_size" : 0.225,#1.4,
                "tick_frequency" : 10,
                "numbers_to_show" : list(range(2, 20,2)),
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
        "mean":0,
        "variance":0.5
    }

    def construct(self):

        time_axes = Axes(
            x_min = 0, x_max = 20,
            y_min = -5, y_max = 5,

        number_line_config = {
            "tick_size" : 0.05,
        },
        x_axis_config = {
            "unit_size" : 1,
            "tick_frequency":1,
            "include_numbers":True,
            "numbers_to_show": np.arange(0,20,2)

            },
        y_axis_config = {
            "unit_size":0.5,
            "tick_frequency":1,
            "include_numbers":True,
            "numbers_to_show":np.arange(-5,5,1),
            "label_direction":UP,
            "include_tip": True
        })
        #Ampt-Frequency-Mean-Variance
        time_func =self.time_signal(2,10,0,6)
        time_graph = time_axes.get_graph(time_func,color = YELLOW)
        t_sig = VGroup(time_axes,time_graph).to_edge(LEFT,buff = 0)

        frequency_axes = Fourier.get_frequency_axes(self)
        freq_graph = Fourier.get_fourier_graph(self,self.frequency_axes, time_func, 0, 20)
        f_sig = VGroup(frequency_axes,freq_graph).to_edge(3*LEFT,buff = 0)

        circle =Dot(color = GREEN)
        #self.add(freq_graph)
        self.add(f_sig)
        #self.play(MoveAlongPath(circle,freq_graph,run_time = 20))
        #self.wait()


    def time_signal(self,amp,freq,mean,var):
        return lambda t: amp*np.cos(2*PI*freq*t)+np.random.normal(mean,var)


class GraphRedrawing(GraphScene):
    CONFIG = {
        "x_axis_label": "",
        "y_axis_label": "",
        "x_min": 0,
        "x_max": 15,

        "x_axis_width": 12,
        "y_min": -5,
        "y_max": 5,
        #"y_axis_height": 6,
        "y_tick_frequency": 1,

        "graph_origin": 5.5 * LEFT,
    }

    def construct(self):
        self.setup_axes()
        self.x_axis.add_numbers(*range(1, 15))

        graph = self.get_prior_graph(0)
        vt = ValueTracker(0)

        #1. use add_updater
        # def func(mob):
        #     graph2 = self.get_prior_graph(vt.get_value()/10)
        #     graph.become(graph2)
        #     return graph
        #
        # graph.add_updater(func)

        #2. use always_redraw
        def func():
            return self.get_prior_graph(vt.get_value())

        graph = always_redraw(func).set_color(RED)

        self.add(graph)
        self.play(vt.set_value, 10, run_time=6,rate_func = linear)
        self.wait()

    def get_prior_graph(self, t=6):
        def prior(x):
            return 4*np.cos(0.5*PI*x+2*t)

        return self.get_graph(prior)


class updating_sine(GraphScene):
    def construct(self):
        x_axis = NumberLine(x_min=-5,x_max=5,color = BLUE)
        tracker = ValueTracker(0)
        def pre_draw(update_param):
            func =lambda t:np.cos(2*PI*t +update_param)
            return func

        def prepare_for_redraw():
            dx = 0.0001
            x = tracker.get_value()
            update_param =x-dx
            func = pre_draw(update_param)
            sig_graph = FunctionGraph(func, x_min=-0, x_max=4,color = GREEN)
            return sig_graph.to_edge(LEFT, buff = 0)

        cos_graph = always_redraw(prepare_for_redraw)

        self.add(cos_graph)
        self.play(tracker.set_value, 50, rate_func=linear, run_time=8)
        self.wait()


class TimeSignal(GraphScene):
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
                "x_max" : 20.0,
                "x_axis_config" : {
                    "unit_size" : 0.6,#1.4,
                    "tick_frequency" : 10,
                    "numbers_to_show" : list(range(0, 20,5)),
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
            "mean":0,
            "variance":0.5
        }
        def construct(self):
            self.draw_sine()

        def draw_sine(self):
            time_axes =Axes(
            x_min = 0, x_max = 10,
            y_min = -5, y_max = 5,
            x_axis_config = {"unit_size":2}
            )
            tracker = ValueTracker(0)

            def pre_draw(update_param):
                func =lambda t:3*np.cos(2*PI*t +update_param)+np.random.normal(0,0.25)
                return func
            def prepare_for_redraw():
                dx = 0.0001
                x = tracker.get_value()
                update_param =x-dx
                func = pre_draw(update_param)
                sig_graph = time_axes.get_graph(func,color = RED)
                return sig_graph

            signal = prepare_for_redraw().move_to(ORIGIN)
            self.play(ShowCreation(signal),rate_func = linear,run_time = 3.5)
            time_graph = always_redraw(prepare_for_redraw)
            t_sig = VGroup(time_axes,time_graph).move_to(ORIGIN)
            self.remove(signal)
            self.add(t_sig)

            self.play(tracker.set_value, 20, rate_func=linear, run_time=5)
            self.wait()


class WrappingAnimation(GraphScene):
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
            "x_max" : 20.0,
            "x_axis_config" : {
                "unit_size" : 0.6,#1.4,
                "tick_frequency" : 10,
                "numbers_to_show" : list(range(0, 20,5)),
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
        "mean":0,
        "variance":0.5
    }
    def construct(self):
        time_axes = Axes(
            x_min = -10, x_max = 10,
            y_min = -5, y_max = 5,

        number_line_config = {
            "tick_size" : 0.05,
        },
        x_axis_config = {
            "unit_size" : 1,
            "tick_frequency":1,
            "include_numbers":True,
            "numbers_to_show": np.arange(0,20,2)

            },
        y_axis_config = {
            "unit_size":0.5,
            "tick_frequency":1,
            "include_numbers":True,
            "numbers_to_show":np.arange(-5,5,1),
            "label_direction":UP,
            "include_tip": True
        })
        time_func = lambda t:2*np.sin(2*PI*t)+2*np.cos(4*PI*t)

        circle_plane = Fourier.get_circle_plane(self)
        #freq_graph = Fourier.get_fourier_graph(self,self.circle_plane, time_func, 0, 20)
        Polygon
        #self.add(circle_plane,freq_graph)






        fft = self.get_dft(time_func,0,5)

        self.add(Polygon(fft))
        #ParametricFunction(
        #fft_g,
        #t_min=0,
        #t_max=5
        #)
        #graph = circle_plane.get_graph(fft)
        #self.add(circle_plane)
        #self.add(fft_g)


    def get_dft(self,func, t_min, t_max,
    complex_to_real_func = DEFAULT_COMPLEX_TO_REAL_FUNC,
    use_almost_fourier = USE_ALMOST_FOURIER_BY_DEFAULT,
    **kwargs ##Just eats these
    ):
        scalar = 1./(t_max - t_min) if use_almost_fourier else 1.0
        def fourier_transform(f):
            z = scalar*scipy.integrate.quad(
                lambda t : func(t)*np.exp(complex(0, -TAU*f*t)),
                t_min, t_max
            )[0]
            return z

        return fourier_transform


class Time2Frequency(GraphScene):
    CONFIG = {
        "time_axes_config" : {
            "x_min" : 0,
            "x_max" : 20,
            "x_axis_config" : {
                "unit_size" : 1,
                "tick_frequency" : 1,
                "numbers_with_elongated_ticks" : [5, 10, 15,20],
            },
            "y_min" : -7,
            "y_max" : 7,
            "y_axis_config" : {"unit_size" : 1},
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
            "x_max" : 50.0,
            "x_axis_config" : {
                "unit_size" : 0.45,
                "tick_frequency" : 2,
                "numbers_to_show" : list(range(5, 50,5)),
            },
            "y_min" : 0,
            "y_max" : 1.3,
            "y_axis_config" : {
                "unit_size" :3,
                "tick_frequency" : 0.5,
                "label_direction" : LEFT,
                "numbers_to_show" : list(range(0, 2,1)),
            },
            "color" : TEAL,
        },
        "frequency_axes_box_color" : RED,
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
        "mean":0,
        "variance":0.5
    }
    def construct(self):
        time_axis = self.set_time_axis()
        time_func = self.get_time_func(freq = 5)
        time_graph = self.get_time_graph(time_axis,time_func)
        hann_coords = self.set_window(time_axis,5) #5==> Contraction Factor
        cust_graph = self.get_cust_graph(time_axis,hann_coords[0],hann_coords[1])
        freq_axis = self.set_freq_axis()
        freq_graph =self.get_freq_graph(freq_axis,time_func,0,2) #To adjust the mainlope ONLY

        #self.animate_time_func(time_axis,time_graph)
        #self.animate_fourier(freq_axis,freq_graph)

    def set_window(self,axis,contraction_factor):
        X =np.linspace(axis.x_min/contraction_factor,axis.x_max/contraction_factor,20)
        hannwin = lambda t:np.hanning(len(t))
        Y = hannwin(X)
        return X ,Y

    def animate_time_func(self,time_axis,time_graph):
        t_gram = VGroup(time_graph,time_axis).scale(0.5).to_edge(UL)
        self.play(AnimationGroup(
            ShowCreation(time_axis,run_time = 3),ShowCreation(time_graph,run_time = 3),
            lag_ratio = 0
        ))
        self.wait()

    def animate_fourier(self,freq_axis,freq_graph):
        f_gram = VGroup(freq_axis,freq_graph).scale(0.6).to_edge(DL).shift(0.15*LEFT)
        self.play(AnimationGroup(
            ShowCreation(freq_axis,run_time = 3),ShowCreation(freq_graph,run_time = 3),
            lag_ratio = 0
        ))
        self.wait()

    def set_time_axis(self):
        time_axis = Axes(
            x_min = 0    ,x_max=10,
            y_min = -2.2     ,y_max=2.2,
            x_axis_config = {"unit_size": 2, "tick_frequency":5,"tick_size":0.25},
            y_axis_config = {"unit_size": 2,"tick_frequency":1},
            )
        return time_axis.set_color(BLUE)

    def get_time_func(self,freq):
        time_func = lambda t:np.sin(TAU*freq*t)#*np.hanning(t)
        return time_func
   
    def get_time_graph(self,time_axis,time_func):
        #time_func = self.get_time_func(freq)
        return time_axis.get_graph(time_func,color = YELLOW,stroke_width = 2)
    
    def set_freq_axis(self):
        freq_axis = Fourier.get_frequency_axes(self)
        return freq_axis

    def get_freq_graph(self,freq_axis,time_func,t_min,t_max):
        
        freq_graph = Fourier.get_fourier_graph(self,freq_axis,time_func,t_min,t_max)
        return freq_graph

    def get_cust_graph(self,axis,X,Y):
        coords = [[px,py] for px,py in zip(X,Y)]
        points = SmoothPtGraph.get_points_from_coords(self,axis,coords)
        graph = SmoothPtGraph(points,color=YELLOW)
        return graph



class SmoothPtGraph(VMobject):
    def __init__(self,set_of_points,**kwargs):
        super().__init__(**kwargs)
        self.set_points_smoothly(set_of_points)

    def get_points_from_coords(self,axes,coords):
        return [axes.coords_to_point(px,py)
            for px,py in coords
            ]