#Updated For GitHub
from manimlib.imports import *
from scipy.integrate import quad
# Fourier Imports:
from manimlib.mobject.FourierLib import*
USE_ALMOST_FOURIER_BY_DEFAULT = True
#The used Number of sample is located inside
#The FourierLib.py file
NUM_SAMPLES_FOR_FFT = 1000
#NOISE = lambda t:0.001*np.cos(2*np.pi*9*t)
from DroneCreature.DroneCreature import *

class RandomTalk(GraphScene,MovingCameraScene):

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
        time_axis = Axes(x_min = -5,x_max = 5,y_min = -1.2,y_max = 1.2,
                x_axis_config = {"unit_size":1,"tick_frequency":1,"include_numbers":True,"numbers_to_show":list(range(1,5,1)),"include_tip":False},
                y_axis_config = {"unit_size":1, "tick_frequency":1}).set_color(BLUE).move_to(ORIGIN)
        self.time_axis = time_axis
        return time_axis
  
    def get_time_func(self,Amp,Freq,Phi):
        #Uncomment for Intro Scene
        #time_axis = self.get_time_axis()
        time_axis =self.time_axis
        time_func = lambda t:Amp*np.sin(Freq*TAU*t+Phi)
        time_signal = time_axis.get_graph(time_func,color = YELLOW)
        time_graph = VGroup(time_axis,time_signal).move_to(ORIGIN).to_edge(LEFT,buff=1)
        return time_graph

    def acquire_time_func(self,x_mn,x_mx,f_min = None,f_max=None):
        time_axis = Axes(x_min = x_mn,x_max = x_mx,y_min = -4.0,y_max = 4.0,
                x_axis_config = {"unit_size":1,"tick_frequency":2,"include_numbers":True,"numbers_to_show":list(range(0,6,2)),"include_tip":True},
                y_axis_config = {"unit_size":0.5, "tick_frequency":1,"include_numbers":True,"label_direction": UP,"numbers_to_show":list(range(-3,5,2)),"include_tip":True}).set_color(BLUE)
        
        #time_axis =self.time_axis
        time_func = lambda t:3*np.sin(1*TAU*t+0)
        if f_min==None and f_max==None:        
            time_signal = time_axis.get_graph(time_func,color = YELLOW)
        else:
            time_signal = time_axis.get_graph(time_func,x_min = f_min,x_max =f_max,color = YELLOW)
        time_graph = VGroup(time_axis,time_signal).move_to(ORIGIN)
        return time_graph

    def get_ft_func(self,x_mn,x_mx,f_min = None,f_max=None):
        time_axis = Axes(x_min = x_mn,x_max = x_mx,y_min = -4.0,y_max = 4.0,
                x_axis_config = {"unit_size":2,"tick_frequency":2,"include_numbers":True,"numbers_to_show":list(range(0,6,2)),"include_tip":True},
                y_axis_config = {"unit_size":0.5, "tick_frequency":1,"include_numbers":True,"label_direction": UP,"numbers_to_show":list(range(-3,5,2)),"include_tip":True}).set_color(BLUE)
        
        #time_axis =self.time_axis
        time_func = lambda t:3*np.sin(2*TAU*t+0) + 2*np.sin(6*TAU*t+0)
        if f_min==None and f_max==None:        
            time_signal = time_axis.get_graph(time_func,color = YELLOW)
        else:
            time_signal = time_axis.get_graph(time_func,x_min = f_min,x_max =f_max,color = YELLOW)
        time_graph = VGroup(time_axis,time_signal).move_to(ORIGIN)
        return time_graph

    def get_noisy_func(self,Amp,Freq,Phi):
        #time_axis = self.get_time_axis()
        time_axis =self.time_axis
        time_func = lambda t:Amp*np.sin(Freq*TAU*t+Phi)+np.sin(5*TAU*t+Phi)+np.sin(10*TAU*t+Phi)
        
        time_signal = time_axis.get_graph(time_func,color = YELLOW,stroke_width=3)
        time_graph = VGroup(time_axis,time_signal)
        return time_signal

    def get_harmonics(self,har_Phi):
        #time_axis = self.get_time_axis()
        time_axis =self.time_axis
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
            graph = DiscreteGraphFromSetPoints(points,color=TEAL,stroke_width = 6)
            return graph

###########################################################

        
    def intro(self,):
        time_axis = Axes(x_min = -5,x_max = 5,y_min = -2,y_max = 2,
                x_axis_config = {"unit_size":1,"tick_frequency":1,"include_numbers":True,"numbers_to_show":list(range(-5,5,1)),"include_tip":False},
                y_axis_config = {"unit_size":1, "tick_frequency":1,"include_numbers":False,"numbers_to_show":list(range(-3,3,1)),"include_tip":True}).set_color(BLUE).move_to(ORIGIN)
        self.time_axis = time_axis
        
        Amp_tracker = ValueTracker(1)
        Freq_tracker = ValueTracker(1)
        Phi_tracker = ValueTracker(0)
        Amp_num = DecimalNumber(1,num_decimal_places = 1).save_state()
        Freq_num = DecimalNumber(1,num_decimal_places = 1).save_state()
        Phi_num = DecimalNumber(1,num_decimal_places = 1).save_state()
        para_num = VGroup(Amp_num,Freq_num,Phi_num)
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
        para_var = VGroup(Amp,Freq,Phi)
        Freq.save_state()
        self.play(FadeIn(famous,run_time = 1.5),Write(sin))
        self.wait()
        self.play(FadeOut(famous),sin.to_edge,UP)
        

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
            self.play(Amp_tracker.set_value,2.5,run_time = 1)
            self.wait()
            self.play(Amp_tracker.set_value,1,run_time = 1)
            
        def weggle_freq():
            Amp_num.clear_updaters()
            self.play(Transform(Amp,sin[0]), ReplacementTransform(Freq,Freq_num.move_to(sin[3]).set_color(TEAL_C)))
            Freq_num.add_updater(lambda m: m.set_value(Freq_tracker.get_value()))
            Freq_num.add_updater(lambda m: m.move_to(sin[3]).set_color(TEAL_C))           
            self.add(time_graph)
            self.play(Freq_tracker.set_value,3,run_time = 1)
            self.wait()
            self.play(Freq_tracker.set_value,1,run_time = 1)
            self.wait()
            
        def weggle_phi():
            self.play(ReplacementTransform(Phi,Phi_num.move_to(sin[5]).set_color(TEAL_C)))
            Phi_num.add_updater(lambda m: m.set_value(Phi_tracker.get_value()))
            Phi_num.add_updater(lambda m: m.move_to(sin[5]).set_color(TEAL_C))           
            self.add(time_graph)
            self.play(Phi_tracker.set_value,10,run_time = 5)
            self.wait()
            self.play(Phi_tracker.set_value,0,run_time = 5)
            self.wait()
        
        weggle_amp()
        weggle_freq()
        weggle_phi()
        
        self.play(FadeOut(time_graph))
        self.wait()

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
        
        fourier_box = Rectangle(width = 2,height = 4, color = TEAL,stroke_width = 6).move_to(2*RIGHT)
        fourier_text = TextMobject("Fourier","Analyzer",color = YELLOW).arrange(DOWN).move_to(fourier_box)
        self.play(FadeIn(fourier_box),FadeIn(fourier_text))
        self.offset = 0
        #Create random noisy signal
        time_axis = Axes(x_min = -3,x_max = 3,y_min = -1.5,y_max = 1.5,
                x_axis_config = {"unit_size":0.55,"tick_frequency":1,"include_numbers":True,"numbers_to_show":list(range(-5,5,1)),"include_tip":False},
                y_axis_config = {"unit_size":0.5, "tick_frequency":1,"include_numbers":False,"numbers_to_show":list(range(-3,3,1)),"include_tip":True}).set_color(BLUE).move_to(ORIGIN)
        self.time_axis = time_axis

        noisy = self.get_noisy_func(2,1,0).next_to(fourier_box,LEFT)
        def update_noisy(noisy,dt):
            noisy.become(self.get_noisy_func(3,1,self.offset)).next_to(fourier_box,LEFT)
            self.offset +=0.1
        self.add(noisy)
        noisy.add_updater(update_noisy)
        self.har_Phi=0
        harmonics = self.get_harmonics(self.har_Phi).arrange(2*DOWN,aligned_edge = LEFT).next_to(fourier_box,RIGHT)
        def update_harmonics(harmonica,dt):
            harmonica.become(self.get_harmonics(self.har_Phi)).arrange(2*DOWN,aligned_edge = LEFT).next_to(fourier_box,RIGHT)
            self.har_Phi += 0.15

        self.wait()
        self.play(ShowCreation(harmonics[0]))
        self.wait()
        self.play(ShowCreation(harmonics[1]))
        self.wait()
        self.play(ShowCreation(harmonics[2]))
        self.wait()
        harmonics.add_updater(update_harmonics)
        self.wait(2)
        combined_analyzer = VGroup(fourier_box,fourier_text,noisy,harmonics)

    def definitions(self):
        wonder = VGroup(*list(map(TextMobject, [
            "Can We Do The Oppsite?","Extract the (ingredient) signals that make up a full wavefrom",
            "It turns out, there is a generilzed form of how the ingredient would look like,",
            "The Ansewer: Sinusoids"
        ]))).arrange(DOWN,aligned_edge = LEFT).set_width(FRAME_WIDTH - 2).to_edge(UL)
        self.play(FadeIn(wonder[0].scale(1.2)))
        self.wait()
        self.play(FadeIn(wonder[1]))
        self.wait()
        self.play(FadeIn(wonder[2]))
        self.wait()
        fourierDef = VGroup(*list(map(TextMobject, [
            "\"Any periodic signal x(t) may be decomposed into infinte series of sine",
            "and cosine functions at different frequencies, phases and amplitudes.\"",
        ])))
        fourierDef.arrange(
            DOWN, buff = MED_LARGE_BUFF,
            aligned_edge = LEFT
        ).set_color(TEAL)
        fourierDef.set_width(FRAME_WIDTH - 2)

        theory = VGroup(fourierDef,TextMobject("-Fourier Theorem",color = YELLOW).scale(0.7)).arrange(DOWN,aligned_edge = RIGHT).to_edge(LEFT)
        
        self.add(theory)

    def harmonic_deconstructor(self):
        #Define Harmonics TexEquations
        harEqns = VGroup(*list(map(TexMobject, [
                r"3\sin(2\pi\ 1\ t)",
                r"\sin(2\pi\ 5\ t)" ,
                r"\sin(2\pi\ 10\ t)",
                
        ]))).set_color(WHITE)
        
        time_axis = Axes(x_min = -5,x_max = 5,y_min = -1.5,y_max = 1.5,
                x_axis_config = {"unit_size":1,"tick_frequency":1,"include_numbers":True,"numbers_to_show":list(range(-5,5,1)),"include_tip":False},
                y_axis_config = {"unit_size":0.5, "tick_frequency":1,"include_numbers":False,"numbers_to_show":list(range(-3,3,1)),"include_tip":True}).set_color(BLUE).move_to(ORIGIN)
        self.time_axis = time_axis

        self.har_Phi=0
        Amp = 3
        Freq = 1
        self.Phase =0 
        noisy = self.get_noisy_func(Amp,Freq,self.Phase).to_edge(RIGHT)
        harmonics = self.get_harmonics(self.har_Phi).to_edge(UP).arrange(3*DOWN)
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
            harmonics.become(self.get_harmonics(self.har_Phi)).arrange(3*DOWN)
            self.har_Phi +=0.1
        #self.add(harmonics.shift(UP))
        #Create Animations Here
        harmonics.move_to(ORIGIN)

        self.play(ShowCreation(harmonics[0]),FadeIn(harEqns[0].next_to(harmonics[0],0.6*UP)))
        
        self.play(ShowCreation(harmonics[1]),FadeIn(harEqns[1].next_to(harmonics[1],0.6*UP)))
        
        self.play(ShowCreation(harmonics[2]),FadeIn(harEqns[2].next_to(harmonics[2],0.6*UP)))
        
        self.wait()
        
        #signals.add_updater(update_noisy)
        harmonics.add_updater(update_harmonics)
        funcs = [lambda t:3*np.sin(TAU*t+self.har_Phi), lambda t: np.sin(5*TAU*t+self.har_Phi), lambda t: np.sin(10*TAU*t+self.har_Phi)]        
        seq1 = lambda t:3*np.sin(TAU*t+self.har_Phi) + np.sin(5*TAU*t+self.har_Phi)
        seq1_g = time_axis.get_graph(seq1,color = YELLOW,stroke_width=4).move_to(harmonics[0])
        self.remove(harEqns[0],harEqns[1],harEqns[2])
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
        harEqn = TexMobject(r"3\sin(2\pi\ 1\ t) + \sin(2\pi\ 5\ t) + \sin(2\pi\ 10\ t)").set_color(BLUE).next_to(noisy,UP)
        self.play(Write(harEqn))
        self.play(ShowPassingFlashAround(harEqn))
        self.wait(6)

        #self.add(seq1_g)
       # self.add(harmonics[0])
        
    def frequency_as_handle(self):
        MovingCameraScene.setup(self)
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

        self.play(AnimationGroup(
            ShowCreation(VGroup(left_arrow,right_arrow,AMP,PHASE)),
            ShowCreation(VGroup(fourier_box,fourier_transform_text)),
            ShowCreation(VGroup(midUP_arrow,y,Y)),
            lag_ratio = 0.4
        ))
        #self.add(y,Y,AMP,PHASE)
        
        self.wait()
        '''
        self.clear()
        
        AMP = TexMobject(r"\\ = \left| Y \right|",color = YELLOW).next_to(Y,RIGHT).scale(1.5)
        PHASE = TexMobject(r"\angle Y",color = YELLOW).next_to(AMP,RIGHT).scale(1.5)
        eqn = VGroup(Y,AMP,PHASE).move_to(ORIGIN)
        self.add(y,eqn)
        self.wait()
        '''
    
    def introducing_fourier_plot(self):

        MovingCameraScene.setup(self)
        box = Rectangle(width = FRAME_WIDTH,height = FRAME_HEIGHT, color = TEAL,stroke_width = 4)
        divider = Line(box.get_left(),box.get_right(),color  =TEAL,stroke_width = 4)
        #self.add(box,divider)
        
        #Get Time/Frequency Axes
        time_axis = Axes(x_min = 0,x_max = 5,y_min = -4.0,y_max = 4.0,
                x_axis_config = {"unit_size":1,"tick_frequency":2,"include_numbers":True,"numbers_to_show":list(range(0,6,2)),"include_tip":True},
                y_axis_config = {"unit_size":0.5, "tick_frequency":1,"include_numbers":True,"label_direction": UP,"numbers_to_show":list(range(-3,5,2)),"include_tip":True}).set_color(BLUE)
        self.time_axis = time_axis
        freq_axes = Fourier.get_frequency_axes(self)
        
        time_func_tx = TexMobject(r"3 sin(2\pi.2t) + 2sin(2\pi.6t)",tex_to_color_map = {"2t":RED,"6t":GREEN})

        #Get time/Frequency Function
        time_func = lambda t:3*np.sin(TAU*2*t) + 2*np.sin(TAU*6*t)
        
        time_graph = self.get_ft_func(0,6).to_edge(UP,buff = 0).scale(0.9)
        time_graph.save_state()
        freq_graph = Fourier.get_fourier_graph(self,freq_axes,time_func,0,3)
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
        time_func_tx.next_to(divider,1.5*DOWN)
        self.play(FadeIn(time_func_tx),run_time = 2)
        self.wait()
        
        time_func = lambda t:3*np.sin(TAU*t)
        time_signal = time_axis.get_graph(time_func,color = YELLOW)
        time_axis_old = time_axis
        time_graph2 = VGroup(time_axis,time_signal).move_to(ORIGIN).to_edge(DOWN,buff=0.2)
        self.play(FadeOut(freq_graph))
        self.remove(time_func_tx)
        self.play(ReplacementTransform(time_graph,time_graph2))
        self.wait()
        
        #Write Fourier Equation:
        f_eqn = TexMobject(r"\int_{-\infty }^{\infty }",r"f(t)", r"e^{-2 \pi i f t}").scale(1.3).to_edge(UP,buff = 1.5)
        defin = TexMobject(r"\begin{matrix}0>t<5 & sin(2\pi.t) \\ t>5 & Undefined  \\ t<0 & Undefined  \end{matrix}")
        B1 = Brace(defin,LEFT,color = YELLOW)
        ft = TexMobject(r"f(t) = ").next_to(B1,LEFT)
        defin = VGroup(ft,defin,B1).next_to(time_graph2,RIGHT)

        
       
        time_graph2.save_state()

        f_eqn[0].set_color(RED)
        self.play(FadeIn(f_eqn))
        self.wait()
        self.play(time_graph2.to_edge,LEFT)
        self.play(FadeIn(defin.to_edge(RIGHT)))
        
        self.add(time_axis)
        #self.play(f_eqn.to_edge,RIGHT)
        #defin.next_to(f_eqn,LEFT)
        self.play(defin.to_edge,UL,defin.align_to,f_eqn,UP,f_eqn.to_edge,RIGHT,time_graph2.restore)
        
        self.wait()
        

        #Update Time Graph
        
        time_axis = Axes(x_min = -20,x_max = 25,y_min = -4,y_max = 3,
                x_axis_config = {"unit_size":0.5,"tick_frequency":2,"include_numbers":True,"numbers_to_show":list(range(-20,25,5)),"include_tip":True},
                y_axis_config = {"unit_size":0.5, "tick_frequency":10,"include_numbers":False,"label_direction": UP,"numbers_to_show":list(range(-3,5,2)),"include_tip":False}
                ).set_color(BLUE)
        #self.time_axis = time_axis
        #Expand Axes
        exp_func = self.acquire_time_func(-50,50).move_to(time_graph2)
        #self.add(exp_func)
        self.play(AnimationGroup(
            FadeOut(time_graph2),FadeIn(exp_func),lag_ratio = 0
        ))
        self.wait()
        self.play(self.camera_frame.scale,4)
        self.wait()
        self.play(self.camera_frame.scale,1/4,FadeOut(exp_func))
        #self.play(self.camera_frame.scale,1/4,FadeOut(exp_func),FadeOut(f_eqn))
        
        ##Some Math
        #define window function w(t):
        w_def = TexMobject(r"\begin{matrix}0>t<5 & 1 \\ t>5 & 0  \\ t<0 & 0  \end{matrix}")
        B2 = Brace(w_def,LEFT,color = YELLOW)
        wt = TexMobject(r"w(t) = ").next_to(B2,LEFT)
        window = VGroup(w_def,wt,B2).to_edge(UR).align_to(defin,UP)
        #self.play(FadeIn(window))
        #self.wait()
        #self.play(FadeOut(divider),VGroup(window,defin).to_edge,UP)
        self.play(FadeOut(divider),VGroup(f_eqn,defin).to_edge,UP)

        #######
        new_f_eqn = TexMobject(r"\int_{-\infty }^{\infty }f(t). e^{-2 \pi i f t}=", r"\int_{-0}^{5 }f(t).e^{-2 \pi i f t}")#.next_to(divider,UP).align_to(joined,LEFT)
        dyingL_int = TexMobject(r" \int_{-\infty }^{0}",r"f(t)",r".e^{-2\pi if t}+")
        dyingR_int = TexMobject(r"+\int_{5 }^{\infty}",r"f(t)",r".e^{-2\pi if t}")
        new_f_eqn[0].set_color(WHITE)
        new_f_eqn[1].set_color(WHITE)
        f_eqnCopy = f_eqn.copy().move_to(ORIGIN).to_edge(LEFT)
        new_f_eqn0 = VGroup(dyingL_int,new_f_eqn[1],dyingR_int).arrange_submobjects(RIGHT).move_to(ORIGIN)
        qL = TexMobject("?",color = YELLOW_C).scale(1.5).move_to(dyingL_int[1])
        qR = TexMobject("?",color = YELLOW_C).scale(1.5).move_to(dyingR_int[1])
        self.play(AnimationGroup(
            FadeIn(new_f_eqn0[0]),
            FadeIn(new_f_eqn0[1]),
            FadeIn(new_f_eqn0[2]),
            lag_ratio = 0.7
        ))
        self.wait()
        self.play(Transform(dyingL_int[1],qL))
        self.wait()
        self.play(Transform(dyingR_int[1],qR))
        self.wait()
    
        ######
        self.play(new_f_eqn0[1].set_color,GREEN_SCREEN)
        self.play(FadeOut(f_eqn),FadeIn(window.move_to(f_eqn)))

        #######
        new_f_eqn = TexMobject(r"\int_{-\infty }^{\infty }[f(t).w(t)]. e^{-2 \pi i f t}=", r"\int_{0}^{5 }[f(t).w(t)].e^{-2 \pi i f t}").set_color(GREEN_SCREEN)#.next_to(divider,UP).align_to(joined,LEFT)
        dyingL_int = TexMobject(r" \int_{-\infty }^{0}",r"[?.w(t)]",r"e^{-2\pi if t}+")
        dyingR_int = TexMobject(r"+\int_{5 }^{\infty}",r"[?.w(t)]",r"e^{-2\pi if t}")
        new_f_eqn[0].set_color(WHITE)
        new_f_eqn[1].set_color(GREEN_SCREEN)
        f_eqnCopy = f_eqnCopy.copy().shift(2*DOWN).to_edge(LEFT)
        new_f_eqn = VGroup(dyingL_int,new_f_eqn[1],dyingR_int).arrange_submobjects(RIGHT).scale(0.9).move_to(ORIGIN).shift(2*DOWN)
        
        self.play(AnimationGroup(
            TransformFromCopy(new_f_eqn0[0],new_f_eqn[0]),
            TransformFromCopy(new_f_eqn0[1],new_f_eqn[1]),
            TransformFromCopy(new_f_eqn0[2],new_f_eqn[2]),
            lag_ratio = 0.7
        ))
        self.wait()
        ######



        jnt_def = TexMobject(r"\begin{matrix}0>t<5 & sin(2\pi.t) \\ t>5 & 0  \\ t<0 & 0  \end{matrix}")
        B3 = Brace(jnt_def,LEFT,color = YELLOW)
        fw = TexMobject(r"f(t).w(t) = ").next_to(B3,LEFT)
        joined = VGroup(jnt_def,fw,B3).next_to(window,5*DOWN).align_to(defin,LEFT)
        self.play(FadeOut(new_f_eqn0),Write(joined))
        vr = Vector(RIGHT).next_to(joined,RIGHT)
        fully = TexMobject("Fully\\ Defined\\ over\\ ",r"\left\langle -\infty ,+\infty  \right\rangle",color = BLUE).arrange(DOWN).next_to(vr,RIGHT)
        self.play(Write(vr),ShowCreation(fully))
        self.wait()

        self.play(AnimationGroup(
            FadeOut(VGroup(defin,window)),
            ApplyMethod(VGroup(joined,vr,fully,new_f_eqn.set_color(YELLOW_C)).to_edge,UP),
            FadeOutAndShiftDown(dyingL_int),
            FadeOutAndShiftDown(dyingR_int),
            FadeIn(divider),
            lag_ratio = 0.3
        ))


        #expAxes_func = self.acquire_time_func(-10,10,0,5).move_to(exp_func).shift(LEFT)
        ##########
        #self.play(FadeOut(exp_func))
        exp_func = VGroup(time_axis,time_axis.get_graph(time_func,color = YELLOW)).move_to(exp_func)
        expAxes_func = VGroup(time_axis,time_axis.get_graph(time_func,x_min = 0,x_max = 5,color = YELLOW)).move_to(exp_func)
        self.play(FadeIn(exp_func))
        
        Ntime_graph = self.get_time_func(3,1,0).move_to(ORIGIN)
        
        def rec(t):
            if t>=0 and t<=5:
                y = 3
            else:
                y = 0
            return y
        t = [*np.arange(-20,20.01,0.01)]
        w = [rec(i) for i in t]
        

        #Animations
        
        rec_window = self.get_cust_graph(time_axis,t,w).set_color(RED)
        self.play(ShowCreation(rec_window),run_time = 4)
        self.wait()
        self.play(AnimationGroup(
            FadeOut(exp_func[1]),FadeIn(expAxes_func[1]),lag_ratio = 0
        ))
        self.play(FadeOut(divider),VGroup(expAxes_func,rec_window).shift,DOWN)
        self.wait()
        fourier_expantion = TexMobject(r"\mathcal{F}[f(t)] \longrightarrow \mathcal{F}[f(t).w(t)]\longrightarrow \mathcal{F}[f(t)]*\mathcal{F}(w(t))").next_to(rec_window,UP)
        self.play(ShowCreation(fourier_expantion))
        self.wait()
        '''
        #######
        new_f_eqn = TexMobject(r"\int_{-\infty }^{\infty }f(t). e^{-2 \pi i f t}=", r"\int_{-0}^{5 }[f(t).w(t)].e^{-2 \pi i f t}+").next_to(divider,UP).align_to(joined,LEFT)
        dyingL_int = TexMobject(r" \int_{-\infty }^{0}",r"f(t)",r"e^{-2\pi if t}+")
        dyingR_int = TexMobject(r"\int_{5 }^{\infty}",r"f(t)",r"e^{-2\pi if t}")
        new_f_eqn[0].set_color(WHITE)
        new_f_eqn[1].set_color(WHITE)
        new_f_eqn = VGroup(new_f_eqn[0],dyingL_int,new_f_eqn[1],dyingR_int).arrange_submobjects(RIGHT).scale(0.9).next_to(divider,UP).align_to(joined,LEFT)
        fourier_expantion = TexMobject(r"\mathcal{F}[f(t)] \longrightarrow \mathcal{F}[f(t).w(t)]\longrightarrow \mathcal{F}[f(t)]*\mathcal{F}(w(t))").next_to(new_f_eqn,2*DOWN).align_to(new_f_eqn,LEFT)
        self.play(FadeIn(new_f_eqn))
        self.wait()
        #########
        self.play(FadeOut(divider),VGroup(expAxes_func,rec_window).shift,DOWN)
        self.play(ShowCreation(fourier_expantion))
        self.wait()
        '''
    
    def default_window(self):

        MovingCameraScene.setup(self)
        self.camera_frame.scale(1.2).save_state()
        box = Rectangle(width = FRAME_WIDTH,height = FRAME_HEIGHT, color = BLACK,stroke_width = 6)
        time_axis = Axes(x_min = -6,x_max =11,y_min = -3,y_max = 3,
                x_axis_config = {"unit_size":0.25,"tick_frequency":2,"include_numbers":False,"numbers_to_show":list(range(-20,25,5)),"include_tip":True},
                y_axis_config = {"unit_size":0.5, "tick_frequency":10,"include_numbers":False,"label_direction": UP,"numbers_to_show":list(range(-3,5,2)),"include_tip":False}
                ).set_color(BLUE)
        freq_axes = Axes(x_min = 0,x_max = 6,y_min = 0,y_max = 3.1,
                x_axis_config = {"unit_size":0.75,"tick_frequency":1,"include_numbers":True,"numbers_to_show":list(range(0,6,1)),"include_tip":True},
                y_axis_config = {"unit_size":0.75, "tick_frequency":1,"include_numbers":False,"label_direction": UP,"numbers_to_show":list(range(1,4,2)),"include_tip":True}
                ).set_color(BLUE)
        time_func = lambda t:3*np.sin(TAU*t)
        exp_func = VGroup(time_axis,time_axis.get_graph(time_func,color = YELLOW)).to_edge(LEFT)
        expAxes_func = VGroup(time_axis,time_axis.get_graph(time_func,x_min = 0,x_max = 5,color = YELLOW)).move_to(exp_func)

        #Equations:
        ft = TexMobject(r"f(t) = 3sin(2.\pi.1t)",color = BLUE_C)
        wt = TexMobject(r"w(t) = \begin{matrix}0>t<5 & 1 \\ otherwise & 0 \end{matrix}").scale(0.7)
        fw = TexMobject(r"f(t).w(t) = " , r"\begin{matrix}0>t<5 & 3sin(2.\pi.1t) \\ \\ Otherwise & 0 \end{matrix}",color = TEAL).scale(0.7)
        fw = VGroup(fw[0],Brace(fw[1],LEFT,color = RED),fw[1]).arrange_submobjects(RIGHT)
        snc = TexMobject(r"sinc",r"\rightleftharpoons {sin(t)\over t}",color = BLUE_C)
        ft.add_updater(lambda m:m.next_to(exp_func[1],DOWN))
        #fe1 = TexMobject(r"\underrightarrow{\mathcal{F}[f(t)]}")
        fe =TexMobject(r"\mathcal{F}[f(t).w(t)]\longrightarrow \mathcal{F}[f(t)]*\mathcal{F}(w(t))",tex_to_color_map = {"f(t)" : YELLOW,"w(t)" : RED}).scale(0.9)
        #fe = VGroup(fe1,fe2).arrange_submobjects(DOWN)
        conv_txt = TextMobject("-Fundamental Theory Of Convolution",color = YELLOW).scale(0.7)
        conv_txt.add_updater(lambda m: m.next_to(fe,DOWN))
        

        #Define Rectangular window
        def rec(t):
            if t>=0 and t<=5:
                y = 3
            else:
                y = 0
            return y
        t = [*np.arange(-5,10.01,0.01)]
        w = [rec(i) for i in t]
        rec_window = self.get_cust_graph(time_axis,t,w).set_color(RED)
        wt.add_updater(lambda m: m.next_to(rec_window,UP))

        #Animations: Creation of time domain
        self.play(self.camera_frame.scale,0.7,self.camera_frame.move_to,exp_func)
        self.play(ShowCreation(exp_func[1]),FadeIn(ft),ShowCreation(box))
        
        self.wait()
        
        self.play(AnimationGroup(
            ShowCreation(rec_window,run_time = 2.5),FadeIn(wt),FadeOut(exp_func[1],run_time = 2.5),FadeIn(expAxes_func[1],run_time = 2.5),
            lag_ratio = 0
        ))
        ft.add_updater(lambda m:m.next_to(expAxes_func[1],DOWN))
        self.wait(2)
        windowed_func = VGroup(rec_window,expAxes_func[1]).save_state()

        #Time Domain Seperation
        f1 = expAxes_func[1]
        f1.generate_target()
        f1.target.to_edge(DOWN)

        f2 = exp_func[1]
        f2.generate_target()
        f2.target.to_edge(DOWN)

        w1 = rec_window
        w1.generate_target()
        w1.target.to_edge(UP)

        #Animate Seperation
        self.play(AnimationGroup(
            MoveToTarget(f1),MoveToTarget(w1),ApplyMethod(self.camera_frame.scale,1/0.8),lag_ratio = 0
        ))
        self.play(AnimationGroup(
            FadeOut(f1),FadeIn(f2.target),lag_ratio = 0
        ))
        self.wait()

        #Fourier: rec_window & Sine
        
        sinc = lambda t: 3*np.sinc(t)
        sinc_graph = VGroup(time_axis,time_axis.get_graph(sinc,x_min = -7,x_max = 7,stroke_width = 6, color = ORANGE)).to_edge(UR)
        
        x = [0,1,2,3,4,5]
        y = [0,3,0,0,0,0]
        y_z = [0,0,0,0,0,0]
        

        freq_axes.to_edge(DR)
        coords = [[px,py] for px,py in zip(x,y)]
        points = self.get_points_from_coords(freq_axes,coords)
        dots = VGroup(*[Dot(color = YELLOW).move_to(freq_axes).move_to([px,py,pz]) for px,py,pz in points])

        coords_z = [[px,py] for px,py in zip(x,y_z)]
        points_z = self.get_points_from_coords(freq_axes,coords_z)
        dots_z = VGroup(*[Dot(color = YELLOW).move_to(freq_axes).move_to([px,py,pz]) for px,py,pz in points_z])

        lines = VGroup(*[Line(dots_z[i],dots[i]) for i in range(6)]).set_color(GREEN_SCREEN)
        freq_graph = VGroup(freq_axes,dots,lines,dots_z).to_edge(DR)

        #Animating fourier creation
        vr1 = Vector(3*RIGHT).next_to(sinc_graph,2*LEFT).set_color(TEAL)
        vr2 = vr1.copy().next_to(freq_graph,2*LEFT).set_color(TEAL)
        vr = VGroup(vr1,vr2)
        self.play(Restore(self.camera_frame),ShowCreation(vr1),ShowCreation(sinc_graph[1]),run_time = 2)
        self.wait()
        snc.add_updater(lambda m:m.next_to(sinc_graph[1],DOWN))
        self.play(FadeIn(snc[0]))
        self.wait()
        self.play(FadeIn(snc[1]))
        self.wait()
        self.play(AnimationGroup(
            ShowCreation(vr2),ShowCreation(freq_graph[0]),ShowCreation(dots_z),lag_ratio = 1
        ))
        self.play(AnimationGroup(
            Transform(dots_z,dots,run_time = 2),ShowCreation(freq_graph[2],run_time = 3),lag_ratio = 0.1
        ))
        self.wait()
        f_graph = Fourier.get_fourier_graph(self,freq_axes,time_func,0,3)
        
        #Combined finite windowed Animation
        midVec  = Vector(RIGHT).next_to(VGroup(windowed_func,f2.target),buff = 0)
        fw.to_edge(UP,buff = 0.1).align_to(VGroup(windowed_func,f2.target),LEFT)
        fe.to_edge(UR)
        self.play(AnimationGroup(
            FadeOut(vr),ApplyMethod(windowed_func.restore,run_time = 2),FadeOut(f2.target,run_time = 2),
            ShowCreation(fw),
            ShowCreation(SurroundingRectangle(fw)),
            FadeOut(snc),
            lag_ratio = 0
        ))
        self.wait()
        self.play(AnimationGroup(
            ReplacementTransform(sinc_graph[1],f_graph,run_time = 2.5),
            ShowCreation(fe),
            ShowCreation(SurroundingRectangle(fe,color = GREEN_SCREEN)),
            ShowCreation(conv_txt),
            lag_ratio = 0.35
        ))
        self.play(AnimationGroup(
            ApplyMethod(f_graph.shift,2*UP),ApplyMethod(freq_graph.shift,2*UP),
            lag_ratio = 0
        ))
        self.wait()

        #Create Removable Junck
        #self.clear()
            
    def func_2f(self):
        time_axis = Axes(x_min = -9,x_max = 9.5,y_min = -3,y_max = 3,
                x_axis_config = {"unit_size":0.7,"tick_frequency":2,"include_numbers":False,"numbers_to_show":list(range(-20,25,5)),"include_tip":True},
                y_axis_config = {"unit_size":0.65, "tick_frequency":10,"include_numbers":False,"label_direction": UP,"numbers_to_show":list(range(-3,5,2)),"include_tip":False}
                ).set_color(BLUE)
        freq_axes = Axes(x_min = 0,x_max = 6,y_min = 0,y_max = 3.1,
                x_axis_config = {"unit_size":2,"tick_frequency":1,"include_numbers":True,"numbers_to_show":list(range(0,6,1)),"include_tip":True},
                y_axis_config = {"unit_size":1, "tick_frequency":1,"include_numbers":False,"label_direction": UP,"numbers_to_show":list(range(1,4,2)),"include_tip":True}
                ).set_color(BLUE)
        time_func = lambda t: 2*np.sin(TAU*t) + np.sin(4*TAU*t)
        func_2f = VGroup(time_axis,time_axis.get_graph(time_func,x_min = -4.5,x_max = 4.5,color = YELLOW,stroke_width = 3))
        func1 =  VGroup(time_axis,time_axis.get_graph(lambda t: 2*np.sin(TAU*t),x_min = -4.5,x_max = 4.5,color = YELLOW,stroke_width = 3))
        func2 = VGroup(time_axis,time_axis.get_graph(lambda t: np.sin(4*TAU*t),x_min = -2.5,x_max = 2.5,color = YELLOW,stroke_width = 3))
        #Define Rectangular window
        def rec(t):
            if t>=-4.5 and t<=4.5:
                y = 3
            else:
                y = 0
            return y
        t = [*np.arange(-9,9.01,0.01)]
        w = [rec(i) for i in t]
        rec_window = self.get_cust_graph(time_axis,t,w).set_color(RED)
        windowed_func = VGroup(rec_window,func_2f[1]).move_to(time_axis).to_edge(UL)
    

        x = [0,1,2,3,4,5]
        y = [0,2,0,0,1,0]
        y_z = [0,0,0,0,0,0]
        

        freq_axes.to_edge(DL)
        coords = [[px,py] for px,py in zip(x,y)]
        points = self.get_points_from_coords(freq_axes,coords)
        dots = VGroup(*[Dot(color = YELLOW).move_to(freq_axes).move_to([px,py,pz]) for px,py,pz in points])

        coords_z = [[px,py] for px,py in zip(x,y_z)]
        points_z = self.get_points_from_coords(freq_axes,coords_z)
        dots_z = VGroup(*[Dot(color = YELLOW).move_to(freq_axes).move_to([px,py,pz]) for px,py,pz in points_z])

        lines = VGroup(*[Line(dots_z[i],dots[i]) for i in range(6)]).set_color(GREEN_SCREEN)
        freq_graph = VGroup(freq_axes,dots,lines,dots_z).to_edge(DR)
        
        self.play(FadeIn(windowed_func))
        
        self.play(AnimationGroup(
            ShowCreation(freq_graph[0]),ShowCreation(dots_z),lag_ratio = 1
        ))
        
        self.wait()
        f_graph = Fourier.get_fourier_graph(self,freq_axes,time_func,0,3).set_stroke(color = RED,width = 6)
        sinc1 = time_axis.get_graph(lambda t:2*np.sinc(t),x_min = -4.5,x_max = 4.5, color = RED,stroke_width = 6).to_edge(LEFT)
        sinc2 = time_axis.get_graph(lambda t:1*np.sinc(t),x_min = -4.5,x_max = 4.5, color = RED,stroke_width = 6).to_edge(RIGHT,buff = 1.2)
        sinc = VGroup(sinc1,sinc2)
        funcs = VGroup(func1[1].to_edge(LEFT),func2[1].to_edge(RIGHT))
        self.play(ReplacementTransform(func_2f[1],funcs))
    
        self.wait()
        self.play(AnimationGroup(
            Transform(dots_z[1],dots[1],run_time = 2),
            ReplacementTransform(funcs[0],lines[1]),
            lag_ratio = 0.3
        ))
        
        self.play(AnimationGroup(
            Transform(dots_z[4],dots[4],run_time = 2),
            ReplacementTransform(funcs[1],lines[4]),
            lag_ratio = 0.3
        ))
        self.wait()
        self.play(ReplacementTransform(rec_window,sinc))
        self.wait()
        self.play(ApplyMethod(sinc.shift,2.25*DOWN),run_time = 1.5,rate_func = linear)
        self.play(Transform(sinc,f_graph),run_time = 0.5) 
        self.wait()
        ft = TexMobject(r"Y(f) = \int_{0}^{T}[3sin(2\pi.1t) + 1sin(2\pi.4t)]. e^{-2\pi ft}.dt",color = WHITE).scale(1).to_edge(UP,buff=1)
        self.play(Write(ft),run_time = 3)
        self.wait()

    def fourier_parts(self):
        
        MovingCameraScene.setup(self)
        given = TextMobject("Contineous-Time," "Periodic f(t)") #Animate A box around Periodic and refer to transform
        fTrans = TextMobject("Fourier Transform")
        fEqn = TexMobject(
         r"\int_{-\infty }^{+\infty }",
         r"f(t)", 
         r"e^{-j\omega t}",
         r".dt",color = TEAL,
         tex_to_color_map = {
             "\omega": YELLOW
         })
        int = fEqn[0]
        integral  = VGroup(int,fEqn[5])
        given.next_to(fEqn,2*UP)
        fTrans.next_to(fEqn,2*DOWN)

        #Adding
        self.add(given,fEqn,fTrans)
        
        #Lets Ditch integeral -> animate fading out the integral
        self.play(FadeOut(given),FadeOut(fTrans))
        int_frame = SurroundingRectangle(integral)
        exp =  fEqn[2:5]
        exp_frame  = SurroundingRectangle(exp)
        self.play(Indicate(integral))
        self.play(ShowCreationThenDestructionAround(exp_frame))
        #self.remove(exp_frame,int_frame)
        self.play(fEqn.to_edge,UL,buff = 1)
        self.play(integral.set_color,GREY)
        
        
        exp1 = TexMobject(
        r"e^{-j\omega t}",
        r"=\cos(\omega t) - j\sin(\omega t)"
        ).scale(0.8).next_to(fEqn,2*DOWN).shift(0.3*RIGHT)
        ft_exp = TexMobject(
        r"f(t).e^{-j\omega t} = ",
        r"f(t).\cos(\omega t)", "-" 
        r"jf(t)\sin(\omega t)",
        tex_to_color_map = {
            "f(t)":TEAL
        }

        ).to_edge(LEFT).shift(0.25*UP)
        
        self.play(TransformFromCopy(exp,exp1))
        self.play(TransformFromCopy(exp1,ft_exp))
        self.wait()
        b1 = Brace(ft_exp[2:4],DOWN).set_color(RED)
        b2 = Brace(ft_exp[5:7],DOWN).set_color(GREEN)
        parts = TextMobject("Real Part","Imaginary Part",color = BLUE).scale(0.6)
        self.play(AnimationGroup(
            ShowCreation(b1),FadeIn(parts[0].next_to(b1,DOWN,buff=0)),
            ShowCreation(b2),FadeIn(parts[1].next_to(b2,DOWN,buff=0)),
            lag_ratio = 0.7
        ))

        real_part = VGroup(b1,ft_exp[2:4],parts[0])
        imag_part = VGroup(b2,ft_exp[5:7],parts[1])

        self.wait()
        self.play(FadeOut(fEqn),FadeOut(exp1))
        #self.remove(real_part,imag_part)
        #self.play(FadeOut(ft_exp))
        self.clear()
        ################
        
        
        # #############        

        
        omega_s = 5
        omega_w = 0
        omega_tracker = ValueTracker(1)
        omega_num = DecimalNumber(1,num_decimal_places = 1)
        time_axis = Axes(x_min = 0,x_max = 8,y_min = -1.1,y_max = 1.1,
                x_axis_config = {"unit_size":0.5,"tick_frequency":1,"include_numbers":False,"numbers_to_show":list(range(-5,5,1)),"include_tip":False},
                y_axis_config = {"unit_size":1, "tick_frequency":1,"include_numbers":False,"numbers_to_show":list(range(-3,3,1)),"include_tip":False}).set_color(BLUE).move_to(ORIGIN)
        self.time_axis = time_axis
        ft = lambda t: np.cos(omega_s*t)
        Rp = lambda t: np.cos(omega_s*t)* np.cos(omega_w*t)
        Ip = lambda t: np.cos(omega_s*t)* np.sin(omega_w*t)
        funcs = [ft ,Rp , Ip]
        colors = [WHITE,RED,GREEN_SCREEN,BLUE_E]
        
        def get_plts(omega_w):
            time_axis = self.time_axis
            ft = lambda t: np.cos(omega_s*t)
            Rp = lambda t: np.cos(omega_s*t)* np.cos(omega_w*t)
            Ip = lambda t: np.cos(omega_s*t)* np.sin(omega_w*t)
            ftPlot = time_axis.get_graph(ft,stroke_width = 4,color = WHITE)
            RpPlot = time_axis.get_graph(Rp,stroke_width = 4, color = RED)
            IpPlot = time_axis.get_graph(Ip,stroke_width = 4,color = GREEN_SCREEN)
            ftPlot = VGroup(time_axis,ftPlot)
            RpPlot = VGroup(time_axis.copy(),RpPlot)
            IpPlot = VGroup(time_axis.copy(),IpPlot)
            return VGroup(ftPlot,RpPlot,IpPlot)

        plts = get_plts(1)
        
        omega_num.scale(0.7)
        omega_num2 = omega_num.copy()
        omega_num.move_to(ft_exp[3][5]).set_color(RED)
        omega_num2.move_to(ft_exp[6][10]).set_color(GREEN_SCREEN)
        def update_plts(plts,dt):
            
            omega_num.set_value(omega_tracker.get_value())
            omega_num.move_to(ft_exp[3][5]).set_color(RED)
            omega_num2.set_value(omega_tracker.get_value())
            omega_num2.move_to(ft_exp[6][10]).set_color(GREEN_SCREEN)
            
            plts[0].become(get_plts(omega_tracker.get_value())[0])
            plts[1].become(get_plts(omega_tracker.get_value())[1])
            plts[2].become(get_plts(omega_tracker.get_value())[2])
            ftPlot = VGroup(time_axis,plts[0])
            RpPlot = VGroup(time_axis.copy(),plts[1])
            IpPlot = VGroup(time_axis.copy(),plts[2])
            base = VGroup(RpPlot,IpPlot).arrange(RIGHT,buff = 1)
            plots = VGroup(ftPlot,base).arrange(DOWN,buff = 1).to_edge(UP,buff = 0.5)
        
        ftPlot = VGroup(time_axis,plts[0])
        RpPlot = VGroup(time_axis.copy(),plts[1])
        IpPlot = VGroup(time_axis.copy(),plts[2])
        base = VGroup(RpPlot,IpPlot).arrange(RIGHT,buff = 1)
        plots = VGroup(ftPlot,base).arrange(DOWN,buff = 1).to_edge(UP,buff = 0.5)
        
        cos_t = TexMobject(r"f(t) = ","cos(5t)",color = TEAL).scale(0.7).next_to(plots[0],UP,buff = 0)
        almost = TexMobject(
            r"cos(5t) .\left\{ cos(\omega t) - j sin(\omega t) \right\}",
            tex_to_color_map = {
                "cos(5t)":TEAL
            }
            ).scale(0.7).next_to(plots[0],DOWN)
        
        
        #ShowCreration one by one
        self.camera_frame.scale(1.2).save_state()
        self.play(self.camera_frame.scale,0.5,self.camera_frame.move_to,plots[0])
        self.play(ShowCreation(plots[0][1]),FadeIn(cos_t))
        self.wait()
        self.play(FadeIn(almost))
        self.wait()
        #self.play(self.camera_frame.restore)
        self.play(self.camera_frame.scale,1.3,self.camera_frame.shift,3*DOWN)
        self.play(ShowCreation(plots[1][0]),ShowCreation(plots[1][1]))
        self.add(plts)
        
        
        plts.add_updater(update_plts)
        real_part.next_to(plots[1][0],DOWN) 
        imag_part.next_to(plots[1][1],DOWN)
        self.play(FadeIn(real_part),FadeIn(imag_part))
    
        
        self.remove(ft_exp[3][5],ft_exp[6][10])
        self.add(omega_num,omega_num2)
        #Animate without Mean plots
        self.play(omega_tracker.set_value,10,run_time = 10,rate_func = linear)
        self.wait()
        self.play(omega_tracker.set_value,0,run_time = 10,rate_func = linear)
        self.play(self.camera_frame.scale,1.3,self.camera_frame.shift,3*DOWN)
        
        ####################
        # Create Mean graphs
        ####################
        
        #Create two axis for mean plots
        axes = Axes(x_min = 0,x_max = 10,y_min = -4,y_max = 4,
                x_axis_config = {"unit_size":0.5,"tick_frequency":1,"include_numbers":True,"numbers_to_show":list(range(1,10,2)),"include_tip":False},
                y_axis_config = {"unit_size":0.5, "tick_frequency":1,"include_numbers":True,"label_direction": UP,"numbers_to_show":list(range(-4,4,1)),"include_tip":False}).set_color(BLUE).to_edge(LEFT)
       
        axes.to_edge(LEFT).shift(5.5*DOWN)

        s_axes = Axes(x_min = 0,x_max = 10,y_min = -4,y_max = 4,
                x_axis_config = {"unit_size":0.5,"tick_frequency":1,"include_numbers":True,"numbers_to_show":list(range(1,10,2)),"include_tip":False},
                y_axis_config = {"unit_size":0.5, "tick_frequency":1,"include_numbers":True,"label_direction": UP,"numbers_to_show":list(range(-4,4,1)),"include_tip":False}).set_color(BLUE).to_edge(RIGHT).shift(5.5*DOWN)
      
        t = [*np.arange(0,1)]
        y = [lambda t: np.cos(4*t)*np.cos(1*t)]#,lambda t: np.cos(2*t)*np.cos(3*t),lambda t: np.cos(3*t)*np.cos(3*t),lambda t: np.cos(4*t)*np.cos(3*t),lambda t: np.cos(5*t)*np.cos(3*t)]
        ft = [quad(i,0,5)[0] for i in y] 
        
        
        coords = [[px,py] for px,py in zip(t,ft)]
        points = self.get_points_from_coords(axes,coords)
        pointts = self.get_points_from_coords(s_axes,coords)
        dots = VGroup(*[Dot(color = RED).move_to(axes).move_to([px,py,pz]) for px,py,pz in points])
        dotts = VGroup(*[Dot(color = GREEN).move_to(s_axes).move_to([px,py,pz]) for px,py,pz in pointts])
        ##############
        
        self.add(axes,dots,s_axes,dotts)
        
        pts=[0]
        tt = [0]
        tt_s = [0]
        cos_pts = [0]
        sin_pts = [0]
        def update_mean(dots,dt):
            t = omega_tracker.get_value()
            tt.append(t)
            y = lambda t: np.cos(omega_s*t)*np.cos(omega_tracker.get_value()*t)
            fft = quad(y,0,8)[0]
            cos_pts.append(quad(y,0,8)[0])
            coords = [tt,ft]
            point = axes.coords_to_point(t,fft)
            pts.append(coords)
            dot = Dot(color = RED).move_to(axes).move_to(point)
            self.add(dots)
            dots.become(dot)
            #For Sin
        def update_mean_sin(dotts,dt):
            t = omega_tracker.get_value()
            tt_s.append(t)
            y = lambda t: np.cos(omega_s*t)*np.sin(omega_tracker.get_value()*t)
            fft = quad(y,0,8)[0]
            sin_pts.append(quad(y,0,8)[0])
            coords = [tt,ft]
            point = s_axes.coords_to_point(t,fft)
            pts.append(coords)
            dot_s = Dot(color = GREEN).move_to(s_axes).move_to(point)
            self.add(dotts)
            dotts.become(dot_s)

        self.play(self.camera_frame.move_to, 3.5*DOWN)
        dots.add_updater(update_mean)
        dotts.add_updater(update_mean_sin)
        self.play(omega_tracker.set_value,10,run_time = 10,rate_func = linear)
        
        cos_mean = self.get_cust_graph(axes,tt,cos_pts).set_color(RED).to_edge(LEFT)
        sin_mean = self.get_cust_graph(s_axes,tt_s,sin_pts).set_color(GREEN).to_edge(RIGHT)
        self.play(AnimationGroup(
            FadeIn(cos_mean,run_time = 2.5),FadeIn(sin_mean,run_time = 2.5),lag_ratio = 0
        ))
        
        self.wait()
        self.play(omega_tracker.set_value,0,run_time = 10,rate_func = linear)
        self.wait
    
    def construct(self):
        #self.intro()
        self.default_window()
        '''
        self.clear()
        self.msg_containers()
        self.clear()
        self.harmonic_deconstructor()
        self.clear()
        self.definitions()
        self.clear()
        self.frequency_as_handle()
        self.clear()
        self.introducing_fourier_plot()
        self.clear()
        self.default_window()
        self.func_2f()
        self.clear()
        self.fourier_parts()
        self.wait()
        self.clear()
        '''

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



class FirstSection(RandomTalk):

    def Beam(self,length,width = 0.05 ,color = WHITE):
        
        top = Rectangle(width = length, height = width/3,stroke_opacity = 0,fill_opacity = 0.8).set_color(color)
        middle = Rectangle(width = length, height = width,stroke_opacity = 0,fill_opacity = 1.0).set_color(color)
        bottom = Rectangle(width = length, height = width/3,stroke_opacity = 0,fill_opacity = 0.8).set_color(color)
        body = VGroup(top,middle,bottom).arrange(DOWN,buff = 0)
        return body

    def Rainbow(self,leng = 7):
        
        rainbow_colors = [RED_D, ORANGE,YELLOW_C,GREEN_SCREEN,BLUE_D,PURPLE_C]
        beams = VGroup(*[self.Beam(length = leng,color = i) for i in rainbow_colors]).arrange(DOWN,buff = 0)
        return beams

    def construct(self):
        MovingCameraScene.setup(self)
        self.camera_frame.save_state()
        sunLight = self.Beam(length = 4.5,width = 0.1).to_edge(LEFT).shift(2*RIGHT)
        prism = RegularPolygon(3,fill_color = BLUE_D,stroke_color = BLUE_C,fill_opacity = 1,stroke_opacity = 0.75,stroke_width = 6).scale(1.5).rotate(PI)
        beams = self.Rainbow().to_edge(RIGHT)
        prism.move_to(beams.get_left())
        beams.rotate(-PI/8).shift(1.3*DOWN)
        compound1 = VGroup(sunLight,beams,prism).move_to(ORIGIN).save_state()
        
        s_box = SurroundingRectangle(compound1,color = WHITE)
        self.play(ShowCreation(sunLight))
        self.wait()
        self.play(Write(prism))
        self.wait()
        self.play(Write(beams),run_time = 2)
        self.add(prism)
        self.wait()
        Jim = Alex().scale(0.35).next_to(prism,RIGHT,buff = 2.5).shift(DOWN).set_color(TEAL)
        Jim.look_at(beams)
        self.play(DroneCreatureSays(
            Jim, TextMobject("Wow, a Rainbow!!",color = TEAL), 
            bubble_kwargs = {"height" : 1.5, "width" : 3},
            target_mode="speaking",
            bubble_class=ThoughtBubble,
        ))
        self.play(Blink(Jim))
        self.wait()
        self.play(Blink(Jim))
        self.wait()
        self.play(self.camera_frame.scale,1.5,ShowCreation(s_box),VGroup(s_box,compound1,Jim,Jim.bubble,Jim.bubble.content).shift,3*UL)
        
        Ale = Alex().scale(0.7).to_edge(DR).shift(2*DOWN + 3*RIGHT)
        
        Len = Alex().scale(0.7).to_edge(DL).set_color(BLUE).shift(2*DOWN+ 3*LEFT)
        Ale.look_at(UL)
        self.play(DroneCreatureSays(
            Len, TextMobject("Color Decomposition",color = TEAL), 
            bubble_kwargs = {"height" : 4, "width" : 4},
            target_mode="speaking",
            bubble_class=ThoughtBubble,
        ))
        self.play(Blink(Len))
        self.wait()
        self.play(Blink(Len))
        
        self.play(ReplacementTransform(Len,Ale),FadeOut(Len.bubble),FadeOut(Len.bubble.content))

        self.play(DroneCreatureSays(
            Ale, TextMobject("Ingredient",color = TEAL), 
            bubble_kwargs = {"height" : 4, "width" : 4},
            target_mode="speaking",
            bubble_class=ThoughtBubble))
        self.play(Blink(Ale))
        self.wait()
        self.play(Blink(Ale))
        self.wait()

        self.remove(Jim,Jim.bubble,Jim.bubble.content,Ale,Ale.bubble,Ale.bubble.content,s_box)
        self.play(compound1.restore,self.camera_frame.restore)
        self.wait()
        prism2 = RegularPolygon(3,fill_color = BLUE_D,stroke_color = BLUE_C,fill_opacity = 1,stroke_opacity = 0.75,stroke_width = 6).scale(1.5)
        
        beams.become(self.Rainbow(leng = 4)).rotate(-PI/8).move_to(beams).shift(0.5*UP+RIGHT)
        prism2.move_to(beams.get_right())
        sunLight2 = sunLight.copy().to_edge(RIGHT,buff = 0).shift(0.7*RIGHT)
        self.play(AnimationGroup(
            TransformFromCopy(prism,prism2,run_time = 2),
            FadeIn(sunLight2),
            #FadeIn(prism2),
            lag_ratio = 0.7
        ))
        self.add(prism2)
        self.wait()
        Len.scale(0.8).to_edge(DL)
        self.play(DroneCreatureSays(
            Len, TextMobject("Sunlight Broken",color = BLUE_D), 
            bubble_kwargs = {"height" : 1.5, "width" : 4},
            target_mode="speaking",
            bubble_class=ThoughtBubble,
        ))
        Jim.scale(1.15).to_edge(DR).look_at(UP)
        self.play(DroneCreatureSays(
                    Jim, TextMobject("Sunlight Restored!!",color = TEAL), 
                    bubble_kwargs = {"height" : 1.5, "width" : 4},
                    target_mode="speaking",
                    bubble_class=ThoughtBubble,
                ))
        self.play(Blink(Len))
        self.play(Blink(Jim))
        self.wait()
        self.play(Blink(Len))
        self.play(Blink(Jim))
        self.wait()

        self.remove(Jim,Jim.bubble,Jim.bubble.content,Len,Len.bubble,Len.bubble.content)
        c = [RED_D, ORANGE,YELLOW_C,GREEN_SCREEN,BLUE_D,PURPLE_C]
        w_bar = self.Beam(length = 10,width = 0.4)
        r_bars = self.Beam(length = 10,width = 0.4).next_to(w_bar,2*DOWN).set_color_by_gradient(c)
        self.play(AnimationGroup(
            FadeOut(VGroup(prism,prism2)), FadeOut(sunLight),FadeOut(beams),ReplacementTransform(sunLight2,w_bar),
            lag_ratio = 0
        ))

        self.play(TransformFromCopy(w_bar,r_bars))
        self.play(VGroup(w_bar,r_bars).shift,2*UP)
        Len.to_edge(LEFT).shift(UP)
        self.play(DroneCreatureSays(
            Len, TextMobject("A Spectrum!",color = BLUE_D), 
            bubble_kwargs = {"height" : 1.5, "width" : 4},
            target_mode="speaking",
            bubble_class=ThoughtBubble,
        ))
        self.wait(2)

class fourier_prism(FirstSection,RandomTalk):


    def Rainbow(self,leng = 7,arr = 1,bff = 0):
        rainbow_colors = [RED_D, ORANGE,YELLOW_C,GREEN_SCREEN,BLUE_D,PURPLE_C]
        beams = VGroup(*[self.Beam(length = leng,color = i) for i in rainbow_colors]).arrange_submobjects(arr*DOWN,buff = bff)
        return beams

    def func_rainbow(self,arr = 3):
        time_axis = Axes(x_min = -5,x_max = 5,y_min = -2,y_max = 2,
                x_axis_config = {"unit_size":0.5,"tick_frequency":1,"include_numbers":False,"numbers_to_show":list(range(1,10,2)),"include_tip":False},
                y_axis_config = {"unit_size":0.25, "tick_frequency":1,"include_numbers":False,"label_direction": UP,"numbers_to_show":list(range(-4,4,1)),"include_tip":False}).set_color(BLUE).to_edge(LEFT)
        
        rainbow_colors = [RED_D, ORANGE,YELLOW_C,GREEN_SCREEN,BLUE_D,PURPLE_C]
        freqs = [1,2,3,4,5,6]
        t_funcs = [lambda t:3*np.sin(TAU*1*t),lambda t:np.sin(TAU*2*t),lambda t:np.sin(TAU*3*t),lambda t:np.sin(TAU*4*t), lambda t:np.sin(TAU*5*t), lambda t:np.sin(TAU*6*t)]
        funcs = VGroup(*[time_axis.get_graph(f,color = i) for f,i in zip(t_funcs,rainbow_colors)]).arrange(arr*DOWN)
        tf = lambda t:3*np.sin(TAU*1*t) + np.sin(TAU*2*t)+ np.sin(TAU*3*t)+ np.sin(TAU*4*t) + np.sin(TAU*5*t) + np.sin(TAU*6*t)
        time_func = time_axis.get_graph(tf,color = WHITE)
        return funcs,time_func

    def Scene1(self):
        prism = RegularPolygon(3,fill_color = RED,stroke_color = BLUE_C,fill_opacity = 1,stroke_opacity = 0.0,stroke_width = 6).scale(1.5).rotate(PI).shift(0.5*DOWN)
        original_beams = self.Rainbow().to_edge(RIGHT)
        beams = self.Rainbow(leng = 5,arr = 4,bff = 0.25).to_edge(RIGHT)
        original_sunLight = self.Beam(length = 4.5,width = 0.1).to_edge(LEFT).shift(2*RIGHT)
        sunLight = self.Beam(length = 5.0,width = 0.1).to_edge(LEFT)
        t_funcs = [lambda t:3*np.sin(TAU*1*t),lambda t:np.sin(TAU*2*t),lambda t:np.sin(TAU*3*t),lambda t:np.sin(TAU*4*t), lambda t:np.sin(TAU*5*t), lambda t:np.sin(TAU*6*t)]
        
        graphs = self.func_rainbow()
        graphs[1].to_edge(LEFT)
        graphs[0].move_to(beams)
        
        self.add(original_beams,original_sunLight,prism)
        self.wait()
        self.play(AnimationGroup(
            ReplacementTransform(original_sunLight,sunLight),
            ReplacementTransform(original_beams[0],beams[0]),
            ReplacementTransform(original_beams[1],beams[1]),
            ReplacementTransform(original_beams[2],beams[2]),
            ReplacementTransform(original_beams[3],beams[3]),
            ReplacementTransform(original_beams[4],beams[4]),
            ReplacementTransform(original_beams[5],beams[5]),
            lag_ratio = 0.15
        ))
        self.wait()
        
        #self.play(Transform(beams,graphs[0]))
        self.play(Transform(sunLight,graphs[1]))
        self.play(AnimationGroup(
            Transform(beams[0],graphs[0][0]),
            Transform(beams[1],graphs[0][1]),
            Transform(beams[2],graphs[0][2]),
            Transform(beams[3],graphs[0][3]),
            Transform(beams[4],graphs[0][4]),
            Transform(beams[5],graphs[0][5]),
            lag_ratio = 0.25
        ))
        self.wait(2)
    
        
    def Scene2(self):
        graphs = self.func_rainbow()
        graphs[1].to_edge(LEFT)
        graphs[0].to_edge(RIGHT)
        
        
        prism = RegularPolygon(3,fill_color = RED,stroke_color = BLUE_C,fill_opacity = 1,stroke_opacity = 0.0,stroke_width = 6).scale(1.5).rotate(PI).shift(0.5*DOWN)
        prism.set_color_by_gradient([DARK_BLUE,BLUE_D,PURPLE_C])
        
        compound1 = VGroup(prism,graphs[0],graphs[1])
        sr = SurroundingRectangle(compound1,color = GREEN_SCREEN)
        #compound1 = VGroup(prism,graphs[0],graphs[1],sr)
        title = TextMobject("Fourier Theorem",color = TEAL).scale(2).to_edge(UP).shift(UP)
        line = Line(LEFT*5,RIGHT*5,stroke_width = 2).next_to(title,DOWN)
        self.play(Write(graphs[0]),Write(graphs[1]))
        self.wait()

        Jim = Alex().scale(0.8).next_to(graphs[1].get_right(),6*DOWN).shift(1.3*LEFT).set_color(BLUE_D)
        Jim.look_at(UR)
        self.play(DroneCreatureSays(
            Jim, TextMobject("Sinusoids!!",color = BLUE), 
            bubble_kwargs = {"height" : 3, "width" : 3},
            target_mode="speaking",
            bubble_class=ThoughtBubble,
        ))
        self.wait()
        self.remove(Jim,Jim.bubble,Jim.bubble.content)
        self.play(Write(prism),run_time = 2)
        self.play(AnimationGroup(
            
            ApplyMethod(self.camera_frame.scale,1.25),
            ApplyMethod(compound1.shift,DOWN),
            ShowCreation(sr.shift(DOWN)),
            ShowCreation(title),Write(line),
            Rotating(prism,radians = PI,run_time = 0.5),
            ShowCreation(SurroundingRectangle(prism,color = YELLOW_C)),
            lag_ratio = 0.4
        ))
       
        self.wait()
        
    def Scene3(self):
        #Discuss informattion distribution in fourier transforms
        t_funcs = [lambda t:3*np.sin(TAU*1*t),lambda t:np.sin(TAU*2*t),lambda t:np.sin(TAU*3*t),lambda t:np.sin(TAU*4*t), lambda t:np.sin(TAU*5*t), lambda t:np.sin(TAU*6*t)]
        

        graphs = self.func_rainbow()
        harmonics = graphs[0]
        harmonics.to_edge(RIGHT)
        self.add(harmonics)
        
        harmonics.generate_target()
        harmonics.target.rotate(PI/2).move_to(ORIGIN).scale(0.8).to_edge(UP,buff = 0)
        self.play(MoveToTarget(harmonics),run_time = 2)
        harmonics.save_state()
        s1t = TexMobject("3","sin(","1",".t)",color = RED).next_to(harmonics[0],LEFT).scale(1.5)
        s2t = TexMobject("1","sin(","2",".t)",color = ORANGE).next_to(harmonics[1],LEFT).scale(1.5)
        s3t = TexMobject("1","sin(","3",".t)",color = YELLOW_C).next_to(harmonics[2],LEFT).scale(1.5)
        s4t = TexMobject("1","sin(","4",".t)",color = GREEN_SCREEN).next_to(harmonics[3],LEFT).scale(1.5)
        s5t = TexMobject("1","sin(","5",".t)",color = BLUE_D).next_to(harmonics[4],LEFT).scale(1.5)
        s6t = TexMobject("1","sin(","6",".t)",color = PURPLE_C).next_to(harmonics[5],LEFT).scale(1.5)
        
        output_axes = Axes(x_min = 0,x_max =7,y_min = 0,y_max = 6,
                x_axis_config = {"unit_size":1.5,"tick_frequency":1,"include_numbers":True,"numbers_to_show":list(range(1,8,1)),"include_tip":True},
                y_axis_config = {"unit_size":0.7, "tick_frequency":1,"include_numbers":True,"numbers_to_show":list(range(1,5,1)),"include_tip":True,"label_direction": UP}).set_color(BLUE).to_edge(DOWN,buff = 0).shift(4*LEFT)
        out_axes = Axes(x_min = 0,x_max = 12,y_min = 0,y_max = 6,
                x_axis_config = {"unit_size":0.5,"tick_frequency":1,"include_numbers":True,"numbers_to_show":list(range(1,12,3)),"include_tip":True},
                y_axis_config = {"unit_size":0.7, "tick_frequency":0.5,"include_numbers":True,"numbers_to_show":list(range(1,5,1)),"include_tip":True,"label_direction": UP}).set_color(BLUE).to_edge(RIGHT).shift(3*DOWN)
        
        x =       [0,1,2,3,4,5,6,7]
        y =       [0,4,1,1,1,1,1,0]
        y_z =     [0,0,0,0,0,0,0,0]
        y_phase = [0,4,0,0,0,0,0,0]

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

        lines = VGroup(*[Line(dots_z[i],dots[i],stroke_width = 6) for i in range(8)])
        [lines.set_color(color) for color in [RED_D, ORANGE,YELLOW_C,GREEN_SCREEN,BLUE_D,PURPLE_C]]
        self.add(output_axes,dots_z) 

        lines_p = VGroup(*[Line(dots_zp[i],dots_p[i]) for i in range(8)])
        [lines_p.set_color(color) for color in [RED_D, ORANGE,YELLOW_C,GREEN_SCREEN,BLUE_D,PURPLE_C]]
        
        
        self.wait()
        self.play(ShowCreation(s1t))
        self.play(Transform(dots_z[1],dots[1]),Transform(harmonics[0],lines[1].set_color(RED_D)  ))
        self.play(Transform(s1t,s2t))
        self.play(Transform(dots_z[2],dots[2]),Transform(harmonics[1],lines[2].set_color(ORANGE)  ))
        self.play(Transform(s1t,s3t))
        self.play(Transform(dots_z[3],dots[3]),Transform(harmonics[2],lines[3].set_color(YELLOW_C)  ))
        self.play(Transform(s1t,s4t))
        self.play(Transform(dots_z[4],dots[4]),Transform(harmonics[3],lines[4].set_color(GREEN_SCREEN)  ))
        self.play(Transform(s1t,s5t))
        self.play(Transform(dots_z[5],dots[5]),Transform(harmonics[4],lines[5].set_color(BLUE_D)  ))
        self.play(Transform(s1t,s6t))
        self.play(Transform(dots_z[6],dots[6]),Transform(harmonics[5],lines[6].set_color(PURPLE_C)  ))
        self.play(FadeOut(s1t))
        self.wait()
        harmonics2 = harmonics.copy()
        
        self.play(harmonics.restore)
        #self.play(Transform(dots_z[7],dots[7]),TransformFromCopy(harmonics[6],lines[6]))
        harmonics2.save_state()
        self.wait()
        self.play(harmonics2.restore)
        harmonics.generate_target()
        harmonics.target.rotate(PI/2)
        self.play(MoveToTarget(harmonics))
        for i in range(6):
            self.play(ApplyMethod(harmonics[i].to_edge,UP,buff = 1.5))
        graphs[1].stretch_to_fit_width(10).move_to(ORIGIN).to_edge(UP)
        self.play(Transform(harmonics,graphs[1]))
    
    def construct(self,):

        MovingCameraScene.setup(self)
        self.clear()
        self.Scene1()
        self.clear()
        self.Scene2()
        self.clear()
        self.Scene3()

class Exp3D(ThreeDScene):

    
    
    def construct(self):
        
        grid = NumberPlane(
        dimension=2,
        x_min=-2*FRAME_X_RADIUS,
        x_max= 2*FRAME_X_RADIUS,
        y_min= -4*FRAME_Y_RADIUS,
        y_max= 4*FRAME_Y_RADIUS,
        z_min= -4*FRAME_X_RADIUS,
        z_max= 4*FRAME_X_RADIUS,
    )
        axes = ThreeDAxes()
        self.add(grid,axes)
        l1 = Line(5*LEFT,5*RIGHT,color = BLUE)
        l2 = Line(5*IN,5*OUT,color = BLUE)
        self.offset = 0
        def cosine(phi):
            cosine= ParametricFunction(
                lambda t: np.array([t, 2*np.cos(3*t + phi), 0]),
            t_min=-2*PI,t_max=0,color = YELLOW,stroke_width = 7)
            return cosine
        
        def sine(phi):
            sine = ParametricFunction(
                lambda t: np.array([t, 0, -2*np.sin(3*t + phi)]),
            t_min=-2*PI,t_max=0,color = RED,stroke_width = 7)
            return sine

        def circle(phi):
            circle = ParametricFunction(
                lambda t: np.array([0, 2*np.cos(3*t+phi), -2*np.sin(3*t+phi)]),
            t_min=-2*PI,t_max=0,color = GREEN_SCREEN,stroke_width = 7)
            return circle

        def update_cos(cosine,dt):
            cosine.become(
                ParametricFunction(
                lambda t: np.array([t, 2*np.cos(3*t + self.offset), 0]),
            t_min=-2*PI,t_max=0,color = YELLOW,stroke_width = 7)
            )
            self.offset += 0.15

        def update_sin(sine,dt):
            sine.become(ParametricFunction(
                lambda t: np.array([t, 0, -2*np.sin(3*t + self.offset)]),
            t_min=-2*PI,t_max=0,color = RED,stroke_width = 7))

        def update_circle(circle,dt):
            circle.become(
                ParametricFunction(
                lambda t: np.array([0, 2*np.cos(3*t+self.offset), -2*np.sin(3*t+self.offset)]),
            t_min=-2*PI,t_max=0,color = GREEN_SCREEN,stroke_width = 7)
            )

        cosine = cosine(0)
        sine = sine(0)
        circle = circle(0)

        d = Dot(radius = 0.15,color = BLUE)
        self.add_fixed_orientation_mobjects(d)
        #d.add_updater(lambda m: m.move_to(np.array([0, np.cos(2*PI+self.offset+0.2), 0])))
        
        cosine.add_updater(update_cos)
        sine.add_updater(update_sin)
        d.add_updater(lambda m: m.move_to(np.array([0, 2*np.cos(self.offset),-2*np.sin(self.offset)])))
        #l1.add_updater(lambda m: m.move_to(np.array([0, 2*np.cos(self.offset),-2*np.sin(self.offset)])))
        #l2.add_updater(lambda m: m.move_to(np.array([0, 2*np.cos(self.offset),-2*np.sin(self.offset)])))
        #self.add(cosine,sine,circle,l1,l2)
        self.add(cosine,sine,circle)
        
        
        #self.wait(3)
        
        #d.add_updater(lambda m: m.move_to(np.array([0, 0,-2*np.sin(2*PI+self.offset+0.2),])))
        self.move_camera(phi = 75*DEGREES)
        
        #self.begin_ambient_camera_rotation(rate = 0.3)
        #self.move_camera(theta=-45*DEGREES)
        self.wait(2)
        self.move_camera(phi = 90*DEGREES)
        
        #self.move_camera(theta=-180*DEGREES)
        
        self.wait(2)
        self.move_camera(theta=-180*DEGREES)
        self.wait(2)


class Cover(GraphScene):
    def construct(self):
        grid = NumberPlane()
        name  = TextMobject("Theory of Control").scale(2.5).set_color_by_gradient(RED,YELLOW,BLUE,PURPLE,GREEN)
        cat = TextMobject("DSP","Fourier","PID","LQR","MPC","BODE","RootLocus","FILTERS","STM32","MATLAB","ROBOTICS","DRONES").set_opacity(0.5)
        cat[0].next_to(name,2*UP)
        cat[1].next_to(name,2*DOWN)
        cat[2].next_to(name,2*LEFT)
        cat[3].next_to(name,2*RIGHT)
        cat[4].next_to(name,UL)
        cat[5].next_to(name,DR)
        cat[6].move_to(ORIGIN).to_edge(UP)
        cat[7].move_to(ORIGIN).to_edge(DOWN)
        cat[8].move_to(ORIGIN).to_edge(UR)
        cat[9].move_to(ORIGIN).to_edge(UL)
        cat[10].move_to(ORIGIN).to_edge(DL)
        cat[11].move_to(ORIGIN).to_edge(DR)
        boxes = VGroup([Rectangle(fill=True,color = BLACK).move_to(i) for i in cat])

        
        self.add(grid,boxes,name,cat)