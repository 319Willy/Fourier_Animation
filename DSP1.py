#Updated For GitHub
from manimlib.imports import *
from manimlib.mobject.FourierLib import*

class RandomTalk(ThreeDScene,ThreeDCamera):
    def construct(self):
        self.intro()

    def intro(self,):
        self.dir = 1
        Amp_tracker = ValueTracker(1)
        Freq_tracker = ValueTracker(1)
        time_offset = DecimalNumber(1).save_state()
        self.offset = 1
        offset = self.offset
        time_axis = Axes(x_min = 0,x_max = 4,y_min = -3,y_max = 3,
                        x_axis_config = {"unit_size":3,"tick_frequency":1,"include_numbers":True,"numbers_to_show":list(range(1,5,1)),"include_tip":False},
                        y_axis_config = {"unit_size":1, "tick_frequency":1}).set_color(TEAL)

        
        
        title = TextMobject("Interesting  Ideas  in  Digital  Signal  Processing",color = YELLOW)
        subtitle = TextMobject("Choose a convinient sub title",color = BLUE).next_to(title,DOWN)

        famous = TextMobject("Let's consider the very famous sinusoidal waveform",color = TEAL).to_edge(UP)
        #                 0      1     2      3         4       5     6
        sin = TexMobject("A", "\\sin","(","\\omega", "t - ", "\\phi",")",color = YELLOW).scale(1.3)
        Amp = sin[0]
        Freq = sin[3]
        phase = sin[5]
        Freq.save_state()
        self.play(FadeIn(famous,run_time = 1.5),Write(sin))
        self.wait()
        self.play(FadeOut(famous),sin.to_edge,UP)
        self.wait()
        def update_offset(offset,dt):
            offset.become(DecimalNumber(self.offset))
            self.offset +=self.dir/100
            

        def update_time_amp(time_graph,dt):
            time_graph.become(get_time_func(self,self.offset,2))
            self.offset = Amp_tracker.get_value()
        def update_time_freq(time_graph,dt):
            time_graph.become(get_time_func(self,2,self.offset)) 
            self.offset = Freq_tracker.get_value()       
        def get_time_func(self,Amp,Freq):

            time_func = lambda t:Amp*np.sin(Freq*TAU*t)
            time_signal = time_axis.get_graph(time_func,color = YELLOW)
            time_graph = VGroup(time_axis,time_signal).move_to(ORIGIN).to_edge(LEFT,buff=1)
            return time_graph

        time_graph = get_time_func(self,2,3)
        time_signal = time_graph[1]
        time_signal.save_state()
        #time_graph.add_updater(update_time_freq)
        #time_graph.add_updater(update_time_freq)
        time_graph.add_updater(update_time_amp)
        
        self.play(FadeInFromDown(time_axis))
        self.wait(0.5)
        self.play(ShowCreation(time_signal),rate_func = linear)
        time_offset.add_updater(lambda m: m.set_value(Amp_tracker.get_value()))
        time_offset.add_updater(lambda m: m.next_to(sin[1][0],LEFT,buff = 0.1).set_color(TEAL_C))
        self.play(FadeOut(Amp))
        self.add(time_offset)
        self.add(time_graph)
        self.play(Amp_tracker.set_value,3,run_time = 3,rate_func = linear)
        self.wait()
        self.play(Amp_tracker.set_value,1,run_time = 3,rate_func = linear)
        self.wait()
        self.wait()




            