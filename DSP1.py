#Updated For GitHub
from manimlib.imports import *

# Fourier Imports:
from manimlib.mobject.FourierLib import*
USE_ALMOST_FOURIER_BY_DEFAULT = True
#The used Number of sample is located inside
#The FourierLib.py file
NUM_SAMPLES_FOR_FFT = 1000
#NOISE = lambda t:0.001*np.cos(2*np.pi*9*t)


class RandomTalk(GraphScene):

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
            "x_max" : 11.0,
            "x_axis_config" : {
                "unit_size" : 1,
                "tick_frequency" : 1,
                "numbers_to_show" : list(range(0, 11,2)),
            },
            "y_min" : 0,
            "y_max" : 4,
            "y_axis_config" : {
                "unit_size" :0.75,
                "tick_frequency" : 1,
                "label_direction" : LEFT,
                "numbers_to_show" : list(range(0, 4,3)),
            },
            "color" : BLUE,
        }
,
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


    def get_time_axis(self):
        time_axis = Axes(x_min = -6,x_max = 6,y_min = -1.2,y_max = 1.2,
                x_axis_config = {"unit_size":1,"tick_frequency":1,"include_numbers":True,"numbers_to_show":list(range(1,5,1)),"include_tip":False},
                y_axis_config = {"unit_size":0.5, "tick_frequency":1}).set_color(BLUE).move_to(ORIGIN)
        self.time_axis = time_axis
        return time_axis
  
    def get_time_func(self,Amp,Freq,Phi):
        #time_axis = self.get_time_axis()
        time_axis =self.time_axis
        time_func = lambda t:Amp*np.sin(Freq*TAU*t+Phi)
        time_signal = time_axis.get_graph(time_func,color = YELLOW)
        time_graph = VGroup(time_axis,time_signal).move_to(ORIGIN).to_edge(LEFT,buff=1)
        return time_graph

    def get_ft_func(self,Amp,Freq,Phi):
        #time_axis = self.get_time_axis()
        time_axis =self.time_axis
        time_func = lambda t:Amp*np.sin(Freq*TAU*t+Phi) + 2*np.sin(3*Freq*TAU*t+Phi)
        time_signal = time_axis.get_graph(time_func,color = YELLOW)
        time_graph = VGroup(time_axis,time_signal).move_to(ORIGIN).to_edge(LEFT,buff=1)
        return time_graph

    def get_noisy_func(self,Amp,Freq,Phi):
        time_axis = self.get_time_axis()
        time_func = lambda t:Amp*np.sin(Freq*TAU*t+Phi)+np.sin(5*TAU*t+Phi)+np.sin(10*TAU*t+Phi)
        
        time_signal = time_axis.get_graph(time_func,color = YELLOW,stroke_width=4)
        time_graph = VGroup(time_axis,time_signal)
        return time_signal

    def get_harmonics(self,har_Phi):
        time_axis = self.get_time_axis()
        funcs = [lambda t:3*np.sin(TAU*t+self.har_Phi), lambda t: np.sin(5*TAU*t+self.har_Phi), lambda t: np.sin(10*TAU*t+self.har_Phi)]        
        colors = [RED,GREEN_SCREEN,BLUE_E]
        harmonics = VGroup(*[time_axis.get_graph(f,stroke_width=4,color = color) for f,color in zip(funcs,colors)])
        #harmonics = VGroup(*har_elements)
        
        return harmonics


    def get_points_from_coords(self,axes,coords):
            return [axes.coords_to_point(px,py)
                for px,py in coords
                ]

    def get_dots_from_coords(self,axes,coords,radius=0.1):
            points = self.get_points_from_coords(axes,coords)
            dots = VGroup(*[
                Dot(radius=radius).move_to([px,py,pz])
                for px,py,pz in points
                ]
            )
            return dots

    def get_cust_graph(self,axis,X,Y):
            coords = [[px,py] for px,py in zip(X,Y)]
            points = SmoothPtGraph.get_points_from_coords(self,axis,coords)
            graph = DiscreteGraphFromSetPoints(points,color=TEAL,stroke_width = 4)
            return graph

    def intro(self,):
        
        Amp_tracker = ValueTracker(1)
        Freq_tracker = ValueTracker(1)
        Phi_tracker = ValueTracker(0)
        Amp_num = DecimalNumber(1,num_decimal_places = 1).save_state()
        Freq_num = DecimalNumber(1,num_decimal_places = 1).save_state()
        Phi_num = DecimalNumber(1,num_decimal_places = 1).save_state()
        self.offset = 1

        title = TextMobject("Interesting  Ideas  in  Digital  Signal  Processing",color = YELLOW)
        subtitle = TextMobject("Choose a convinient sub title",color = BLUE).next_to(title,DOWN)

        famous = TextMobject("Let's consider the very famous sinusoidal waveform",color = TEAL).to_edge(UP)
        #                 0      1     2      3         4       5     6
        sin = TexMobject("A", "\\sin","(","\\omega", "t - ", "\\phi",")",color = YELLOW).scale(1.3)   
        self.sin = sin 
        Amp = sin[0]
        Freq = sin[3]
        Phi = sin[5]
        Freq.save_state()
        self.play(FadeIn(famous,run_time = 1.5),Write(sin))
        self.wait()
        self.play(FadeOut(famous),sin.to_edge,UP)
        self.wait()

        def update_time_func(time_graph,dt):
            time_graph.become(self.get_time_func(Amp_tracker.get_value(),Freq_tracker.get_value(),Phi_tracker.get_value() ))
     


        time_graph = self.get_time_func(1,1,0)
        time_signal = time_graph[1]
        time_axis = time_graph[0]
        time_signal.save_state()

        time_graph.add_updater(update_time_func)
        
        self.play(FadeInFromDown(time_axis))
        self.wait(0.5)
        self.play(ShowCreation(time_signal),rate_func = linear)
        
        def weggle_amp():
            self.play(ReplacementTransform(Amp,Amp_num.next_to(sin[1][0],LEFT,buff = 0.1).set_color(TEAL_C)))
            Amp_num.add_updater(lambda m: m.set_value(Amp_tracker.get_value()))
            Amp_num.add_updater(lambda m: m.next_to(sin[1][0],LEFT,buff = 0.1).set_color(TEAL_C))
            self.add(time_graph)
            self.play(Amp_tracker.set_value,2.5,run_time = 0.5)
            self.play(Amp_tracker.set_value,1,run_time = 0.5)
            
        def weggle_freq():
            Amp_num.clear_updaters()
            self.play(Transform(Amp,sin[0]), ReplacementTransform(Freq,Freq_num.move_to(sin[3]).set_color(TEAL_C)))
            Freq_num.add_updater(lambda m: m.set_value(Freq_tracker.get_value()))
            Freq_num.add_updater(lambda m: m.move_to(sin[3]).set_color(TEAL_C))           
            self.add(time_graph)
            self.play(Freq_tracker.set_value,3,run_time = 1)
            self.play(Freq_tracker.set_value,1,run_time = 0.5)
            self.wait()
            
        def weggle_phi():
            self.play(ReplacementTransform(Phi,Phi_num.move_to(sin[5]).set_color(TEAL_C)))
            Phi_num.add_updater(lambda m: m.set_value(Phi_tracker.get_value()))
            Phi_num.add_updater(lambda m: m.move_to(sin[5]).set_color(TEAL_C))           
            self.add(time_graph)
            self.play(Phi_tracker.set_value,50,run_time = 10)
            self.play(Phi_tracker.set_value,1,run_time = 1)
            self.wait()
        
        weggle_freq()
        

    def msg_containers(self):
        self.clear()
        
        sin = TexMobject("A", "\\sin","(","\\omega", "t - ", "\\phi",")",color = WHITE).scale(1.5)
        
        self.play(ShowCreation(sin))
        self.play(sin[0].set_color,BLUE,sin[3].set_color,BLUE,sin[5].set_color,BLUE)
        sin.save_state()
        self.play(sin.scale,3)
        self.play(Indicate(sin[0]),Indicate(sin[3]),Indicate(sin[5]))
        self.wait()
        self.play(sin.restore,sin.shift,3*UP)
        arrow1 = Arrow(sin[0].get_bottom(),sin[0].get_bottom() + 2*DOWN,color = BLUE)
        arrow2 = Arrow(sin[3].get_bottom(),sin[3].get_bottom() +2*DOWN,color = BLUE)
        arrow3 = Arrow(sin[5].get_bottom(),sin[5].get_bottom()+2*DOWN,color = BLUE)
        arrows = VGroup(arrow1,arrow2,arrow3)
        arrow_cover = Brace(arrows,DOWN,color = YELLOW)
        info_store = TextMobject("Information Containers"," \"Storage\"",color = WHITE).arrange(DOWN).scale(0.7).next_to(arrow_cover,DOWN)
        self.play(Write(arrow1),Write(arrow2),Write(arrow3))
        self.play(GrowFromCenter(arrow_cover))
        self.play(FadeInFromDown(info_store))
        presenter1 = VGroup(sin,arrows,arrow_cover,info_store)
        why_sinusoids = TextMobject("Why","Sinusoids?",color = WHITE).arrange(3*DOWN,aligned_edge = LEFT).scale(3).to_edge(RIGHT,buff = 1)
        self.play(presenter1.move_to,5*LEFT) 
        
        fourier_box = Rectangle(width = 2,height = 4, color = TEAL,stroke_width = 6).move_to(2.5*RIGHT)
        fourier_text = TextMobject("Fourier","Analyzer",color = YELLOW).arrange(DOWN).move_to(fourier_box)
        self.play(FadeIn(fourier_box),FadeIn(fourier_text))
        self.offset = 0
        #Create random noisy signal
        noisy = self.get_noisy_func(2,1,0).move_to(ORIGIN)
        def update_noisy(noisy,dt):
            noisy.become(self.get_noisy_func(3,1,self.offset)).move_to(ORIGIN)
            self.offset +=0.1
        self.add(noisy)
        noisy.add_updater(update_noisy)
        self.har_Phi=0
        harmonics = self.get_harmonics(self.har_Phi)
        def update_harmonics(harmonica,dt):
            harmonica.become(self.get_harmonics(self.har_Phi)).arrange(2*DOWN,aligned_edge = LEFT).next_to(fourier_box,RIGHT)
            self.har_Phi += 0.15

        self.wait(2)
        self.play(ShowCreation(harmonics))
        harmonics.add_updater(update_harmonics)
        self.wait(5)
        combined_analyzer = VGroup(fourier_box,fourier_text,noisy,harmonics)

    def harmonic_deconstructor(self):
        time_axis = self.get_time_axis()
        self.har_Phi=0
        Amp = 3
        Freq = 1
        self.Phase =0 
        noisy = self.get_noisy_func(Amp,Freq,self.Phase).to_edge(RIGHT)
        harmonics = self.get_harmonics(self.har_Phi).to_edge(UP).arrange(2*DOWN)
        signals = VGroup(noisy,harmonics).arrange(3*RIGHT)
        def update_phases(signals,dt):
            signals.become(VGroup(
                self.get_noisy_func(3,1,self.Phase),self.get_harmonics(self.Phase).arrange(2*DOWN)
            ).arrange(4*RIGHT))
            self.Phase += 0.1
        def update_noisy(noisy,dt):
            noisy.become(self.get_noisy_func(3,1,self.Phase)).move_to(ORIGIN)
            self.Phase +=0.07
        def update_harmonics(harmonics,dt):
            harmonics.become(self.get_harmonics(self.har_Phi)).arrange(4*DOWN)
            self.har_Phi +=0.1
        self.add(harmonics.shift(UP))
        #signals.add_updater(update_noisy)
        harmonics.add_updater(update_harmonics)
        funcs = [lambda t:3*np.sin(TAU*t+self.har_Phi), lambda t: np.sin(5*TAU*t+self.har_Phi), lambda t: np.sin(10*TAU*t+self.har_Phi)]        
        seq1 = lambda t:3*np.sin(TAU*t+self.har_Phi) + np.sin(5*TAU*t+self.har_Phi)
        seq1_g = time_axis.get_graph(seq1,color = YELLOW,stroke_width=4).move_to(harmonics[0])
        self.wait()
        harmonics.remove_updater(update_harmonics)
        self.play(harmonics[1].move_to,harmonics[0])
        
        self.play(AnimationGroup(
            ReplacementTransform(harmonics[1],seq1_g.set_color(RED)),
            FadeOut(harmonics[0]),
            lag_ratio = 0
        ))
        self.play(seq1_g.move_to ,ORIGIN,run_time = 1.5)
        self.play(harmonics[2].move_to,seq1_g)
        self.play(AnimationGroup(
            ReplacementTransform(seq1_g,noisy.move_to(ORIGIN)),
            FadeOut(harmonics[2]),
            lag_ratio = 0
        ))
        #self.add(noisy)
        #self.remove()
        noisy.add_updater(update_noisy)
        self.wait(8)
        #self.add(seq1_g)
       # self.add(harmonics[0])
        
    def frequency_as_handle(self):
        boarder = Rectangle(width = FRAME_WIDTH,height = FRAME_HEIGHT,stroke_width = 6, color = ORANGE)
        disector = Line(TOP,BOTTOM,stroke_width = 4, color = ORANGE)
        boarders = VGroup(boarder,disector)
        time_axis = Axes(x_min = 0,x_max = 8,y_min = -7,y_max = 7,
                x_axis_config = {"unit_size":0.75,"tick_frequency":1,"include_numbers":True,"numbers_to_show":list(range(1,6,1)),"include_tip":True},
                y_axis_config = {"unit_size":0.43, "tick_frequency":3,"include_numbers":True,"label_direction": UP,"numbers_to_show":list(range(-7,7,3)),"include_tip":True}).set_color(BLUE)
        self.time_axis = time_axis
        time_func = lambda t:4*np.sin(TAU*t-TAU/4)+1.5*np.sin(7*TAU*t)+1*np.sin(8*TAU*t)+0.5*np.sin(9*TAU*t)
        time_signal = time_axis.get_graph(time_func,color = YELLOW)
        time_graph = VGroup(time_axis,time_signal).to_edge(LEFT,buff=0)
        disector.next_to(time_signal)
        #funcs = [lambda t:np.sin(TAU*t), lambda t: 0.9*np.sin(2*TAU*t), lambda t: 0.6*np.sin(4*TAU*t), lambda t: 0.4*np.sin(8*TAU*t) ]        
        funcs = [lambda t:4*np.sin(TAU*t-TAU/4), lambda t: 1.5*np.sin(7*TAU*t), lambda t: 1*np.sin(8*TAU*t), lambda t: 0.5*np.sin(9*TAU*t) ]        

        colors = [WHITE,RED,GREEN_SCREEN,BLUE_E]
        harmonics = VGroup(*[time_axis.get_graph(f,stroke_width=4,color = color) for f,color in zip(funcs,colors)]).arrange(2*DOWN,aligned_edge = LEFT).to_edge(RIGHT)
        
        #Some Animation
        self.play(Write(time_graph),run_time = 2)
        self.wait()
        self.play(AnimationGroup(
            TransformFromCopy(time_signal,harmonics[0]),
            ShowCreation(boarders),
            TransformFromCopy(time_signal,harmonics[1]),
            TransformFromCopy(time_signal,harmonics[2]),
            TransformFromCopy(time_signal,harmonics[3]),
            lag_ratio = 0.7
        ))
        self.wait()

        #Extra Animations
        self.play(FadeOut(boarders),FadeOut(time_graph))
        self.wait()
        output_axes = Axes(x_min = 0,x_max = 12,y_min = 0,y_max = 6,
                x_axis_config = {"unit_size":0.5,"tick_frequency":1,"include_numbers":True,"numbers_to_show":list(range(1,12,3)),"include_tip":True},
                y_axis_config = {"unit_size":0.7, "tick_frequency":0.5,"include_numbers":True,"numbers_to_show":list(range(1,5,1)),"include_tip":True,"label_direction": UP}).set_color(BLUE).to_edge(LEFT).shift(3*DOWN)
        out_axes = Axes(x_min = 0,x_max = 12,y_min = 0,y_max = 6,
                x_axis_config = {"unit_size":0.5,"tick_frequency":1,"include_numbers":True,"numbers_to_show":list(range(1,12,3)),"include_tip":True},
                y_axis_config = {"unit_size":0.7, "tick_frequency":0.5,"include_numbers":True,"numbers_to_show":list(range(1,5,1)),"include_tip":True,"label_direction": UP}).set_color(BLUE).to_edge(RIGHT).shift(3*DOWN)
        
        x =       [0,1,2,3,4,5,6,7,8,9,10,11]
        y =       [0,4,0,0,0,0,0,1.5,1,0.5,0,0,0]
        y_z =     [0,0,0,0,0,0,0,0,0,0,0,0,0]
        y_phase = [0,4,0,0,0,0,0,0,0,0,0,0,0]

        #AMP VS FREQ
        coords = [[px,py] for px,py in zip(x,y)]
        points = self.get_points_from_coords(output_axes,coords)
        dots = VGroup(*[Dot(color = YELLOW).move_to(output_axes).move_to([px,py,pz]) for px,py,pz in points])
        # ZERO VS FREQ
        coords_z = [[px,py] for px,py in zip(x,y_z)]
        points_z = self.get_points_from_coords(output_axes,coords_z)
        dots_z = VGroup(*[Dot(color = YELLOW).move_to(output_axes).move_to([px,py,pz]) for px,py,pz in points_z])

        #PHASE VS Freq
        #out_axes = output_axes.copy().to_edge(RIGHT)
        coords_p = [[px,py] for px,py in zip(x,y_phase)]
        points_p = self.get_points_from_coords(out_axes,coords_p)
        points_zp = self.get_points_from_coords(out_axes,coords_z)
        dots_p = VGroup(*[Dot(color = RED).move_to(out_axes).move_to([px,py,pz]) for px,py,pz in points_p])

        dots_zp = VGroup(*[Dot(color = YELLOW).move_to(out_axes).move_to([px,py,pz]) for px,py,pz in points_zp])

        lines = VGroup(*[Line(dots_z[i],dots[i]) for i in range(11)])
        [lines.set_color(color) for color in colors]
        self.add(output_axes,dots_z) 

        lines_p = VGroup(*[Line(dots_zp[i],dots_p[i]) for i in range(11)])
        [lines_p.set_color(color) for color in colors]
        
        
        self.play(Transform(dots_z[1],dots[1]),TransformFromCopy(harmonics[0],lines[1].set_color(WHITE)))
        self.play(Transform(dots_z[7],dots[7]),Transform(harmonics[1],lines[7].set_color(RED)))
        self.play(Transform(dots_z[8],dots[8]),Transform(harmonics[2],lines[8].set_color(GREEN_SCREEN)))
        self.play(Transform(dots_z[9],dots[9]),Transform(harmonics[3],lines[9].set_color(BLUE)))
        
        self.play(FadeIn((time_axis.move_to(harmonics[0]).set_color(GREEN))))

        self.wait()  
        self.play(Transform(time_axis,out_axes),ShowCreation(dots_zp))
        self.play(Transform(dots_zp[1],dots_p[1]),Transform(harmonics[0],lines_p[1].set_color(RED)))
        self.wait()
        #New Day Animations
        self.play(self.camera_frame.scale, 1.6,self.camera_frame.shift,2*UP)
        fourier_box = Rectangle(width = FRAME_WIDTH/1.25,height = 2,stroke_width=6,color = YELLOW).shift(4.5*UP)
        fourier_transform_text = TexMobject("Fourier\\ Transform").scale(2).move_to(fourier_box)
        ar_st = fourier_box.get_bottom()
        left_arrow = Arrow(ar_st+3*LEFT,ar_st+3*LEFT + 2*DOWN)
        right_arrow = Arrow(ar_st+3*RIGHT, ar_st+3*RIGHT + 2*DOWN )
        midUP_arrow = Arrow(fourier_box.get_top()+2*UP,fourier_box.get_top())
        self.add(fourier_box,fourier_transform_text,left_arrow,right_arrow,midUP_arrow)
        
        y = TexMobject(r"y = f(t)",color = YELLOW).next_to(midUP_arrow,UP).scale(1.5)
        Y = TexMobject(r"Y = F(\omega)",color = BLUE).next_to(fourier_box,DOWN).scale(1.5)
        AMP = TexMobject(r"Amp = \left| Y \right|",color = YELLOW).next_to(left_arrow,DOWN).scale(1.5)
        PHASE = TexMobject(r"Phase = \angle Y",color = YELLOW).next_to(right_arrow,DOWN).scale(1.5)
        self.add(y,Y,AMP,PHASE)

        self.wait()
        self.clear()
        
        AMP = TexMobject(r"\\ = \left| Y \right|",color = YELLOW).next_to(Y,RIGHT).scale(1.5)
        PHASE = TexMobject(r"\angle Y",color = YELLOW).next_to(AMP,RIGHT).scale(1.5)
        eqn = VGroup(Y,AMP,PHASE).move_to(ORIGIN)
        self.add(y,eqn)
        self.wait()
        
    def introducing_fourier_plot(self):
        box = Rectangle(width = FRAME_WIDTH,height = FRAME_HEIGHT, color = TEAL,stroke_width = 4)
        divider = Line(box.get_left(),box.get_right(),color  =TEAL,stroke_width = 4)
        #self.add(box,divider)
        
        #Get Time/Frequency Axes
        time_axis = Axes(x_min = 0,x_max = 5,y_min = -4.0,y_max = 4.0,
                x_axis_config = {"unit_size":2,"tick_frequency":2,"include_numbers":True,"numbers_to_show":list(range(0,6,2)),"include_tip":True},
                y_axis_config = {"unit_size":0.5, "tick_frequency":1,"include_numbers":True,"label_direction": UP,"numbers_to_show":list(range(-3,5,2)),"include_tip":True}).set_color(BLUE)
        self.time_axis = time_axis
        freq_axes = Fourier.get_frequency_axes(self)
        
        #Get time/Frequency Function
        time_func = lambda t:3*np.sin(TAU*2*t) + 2*np.sin(TAU*6*t)
        time_graph = self.get_ft_func(3,1,0).to_edge(UP,buff = 0).scale(0.9)
        time_graph.save_state()
        freq_graph = Fourier.get_fourier_graph(self,freq_axes,time_func,0,5)
        freq_graph = VGroup(freq_axes,freq_graph).move_to(ORIGIN).to_edge(DOWN,buff = 0).to_edge(LEFT,buff = 1).scale(0.9)
        freq_graph.save_state()
        #self.add(freq_axes,freq_graph,time_graph)

        #Some Animations
        self.play(ShowCreation(time_graph.move_to(ORIGIN).scale(1.2)))
        self.wait()
        self.play(time_graph.restore)
        self.wait()

        self.play(ShowCreation(VGroup(box,divider)),run_time = 0.5)
        self.wait()

        self.play(FadeIn(freq_graph[0]),ShowCreation(freq_graph[1],run_time = 5 ,rate_func = linear))
        self.wait()
    
    def construct(self):
        #self.intro()
        #self.msg_containers()
        #self.harmonic_deconstructor()
        #self.frequency_as_handle()
        self.introducing_fourier_plot()

class DiscreteGraphFromSetPoints(VMobject):
    def __init__(self,set_of_points,**kwargs):
        super().__init__(**kwargs)
        self.set_points_as_corners(set_of_points)



class SmoothPtGraph(VMobject):
    def __init__(self,set_of_points,**kwargs):
        super().__init__(**kwargs)
        self.set_points_smoothly(set_of_points)

    def get_points_from_coords(self,axes,coords):
        return [axes.coords_to_point(px,py)
            for px,py in coords
            ]
