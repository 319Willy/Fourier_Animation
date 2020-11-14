#Updated For GitHub
from manimlib.imports import *
from manimlib.mobject.FourierLib import*

class RandomTalk(ThreeDScene,ThreeDCamera):
    def construct(self):
        self.intro()

    def intro(self,):
        Amp_tracker = ValueTracker(1)
        Freq_tracker = ValueTracker(1)
        Phi_tracker = ValueTracker(0)
        Amp_num = DecimalNumber(1,num_decimal_places = 1).save_state()
        Freq_num = DecimalNumber(1,num_decimal_places = 1).save_state()
        Phi_num = DecimalNumber(1,num_decimal_places = 1).save_state()
        self.offset = 1
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

        def update_time_func(time_graph,dt):
            time_graph.become(get_time_func( self,Amp_tracker.get_value(),Freq_tracker.get_value(),Phi_tracker.get_value() ))
     
        def get_time_func(self,Amp,Freq,Phi):

            time_func = lambda t:Amp*np.sin(Freq*TAU*t+Phi)
            time_signal = time_axis.get_graph(time_func,color = YELLOW)
            time_graph = VGroup(time_axis,time_signal).move_to(ORIGIN).to_edge(LEFT,buff=1)
            return time_graph

        time_graph = get_time_func(self,1,1,0)
        time_signal = time_graph[1]
        time_signal.save_state()

        time_graph.add_updater(update_time_func)
        
        self.play(FadeInFromDown(time_axis))
        self.wait(0.5)
        self.play(ShowCreation(time_signal),rate_func = linear)
        
        Amp_num.add_updater(lambda m: m.set_value(Amp_tracker.get_value()))
        Amp_num.add_updater(lambda m: m.next_to(sin[1][0],LEFT,buff = 0.1).set_color(TEAL_C))

        Freq_num.add_updater(lambda m: m.set_value(Freq_tracker.get_value()))
        Freq_num.add_updater(lambda m: m.next_to(sin[3],LEFT,buff = 0.1).set_color(TEAL_C))

        Phi_num.add_updater(lambda m: m.set_value(Phi_tracker.get_value()))
        Phi_num.add_updater(lambda m: m.move_to(sin[5]).set_color(TEAL_C))
        self.play(FadeOut(phase))
        self.add(Phi_num)
        self.add(time_graph)
        self.play(Phi_tracker.set_value,10,run_time = 3,rate_func = linear)
        self.wait()
        self.play(Phi_tracker.set_value,10,run_time = 3,rate_func = linear)
        self.wait()




            