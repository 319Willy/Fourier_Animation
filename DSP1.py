#Updated For GitHub
from manimlib.imports import *
from manimlib.mobject.FourierLib import*
#FUNCS =[lambda t:3*np.sin(TAU*t+self.har_Phi), lambda t: np.sin(5*TAU*t+self.har_Phi), lambda t: np.sin(10*TAU*t+self.har_Phi)]        
 
class RandomTalk(ThreeDScene):

    def get_time_axis(self):
        time_axis = Axes(x_min = -6,x_max = 6,y_min = -1.2,y_max = 1.2,
                x_axis_config = {"unit_size":1,"tick_frequency":1,"include_numbers":True,"numbers_to_show":list(range(1,5,1)),"include_tip":False},
                y_axis_config = {"unit_size":0.5, "tick_frequency":1}).set_color(BLUE).move_to(ORIGIN)
        self.time_axis = time_axis
        return time_axis
  
    def get_time_func(self,Amp,Freq,Phi):
        time_axis = self.get_time_axis()
        time_func = lambda t:Amp*np.sin(Freq*TAU*t+Phi)
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
        time_axis = Axes(x_min = 0,x_max = 8,y_min = -2.5,y_max = 2.5,
                x_axis_config = {"unit_size":0.75,"tick_frequency":1,"include_numbers":True,"numbers_to_show":list(range(1,6,1)),"include_tip":False},
                y_axis_config = {"unit_size":1.1, "tick_frequency":1}).set_color(BLUE)
        self.time_axis = time_axis
        time_func = lambda t:np.sin(TAU*t)+0.9*np.sin(2*TAU*t)+0.6*np.sin(4*TAU*t)+0.4*np.sin(8*TAU*t)
        time_signal = time_axis.get_graph(time_func,color = YELLOW)
        time_graph = VGroup(time_axis,time_signal).to_edge(LEFT,buff=0)
        disector.next_to(time_signal)
        funcs = [lambda t:np.sin(TAU*t), lambda t: 0.9*np.sin(2*TAU*t), lambda t: 0.6*np.sin(4*TAU*t), lambda t: 0.4*np.sin(8*TAU*t) ]        
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


    def construct(self):
        #self.intro()
        #self.msg_containers()
        #self.harmonic_deconstructor()
        self.frequency_as_handle()