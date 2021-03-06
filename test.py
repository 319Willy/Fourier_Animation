from manimlib.imports import*

class TestingGraphs(MovingCameraScene):

    def construct(self):
        curve = FunctionGraph(
            lambda t: np.sin(t),
            x_min = -TAU,
            x_max = TAU,
            color = YELLOW,
            stroke_width = 4
        )
        
        self.add(NumberLine(
            x_min = -70,
            x_max = 70,
            y_min = -20,
            y_max = 20,
            unit_size=0.1,
            include_tick = True,
            tick_frequency = 5,
            tick_size = 0.1,
            include_numbers = True,
            number_scale_val = 0.75,
            numbers_to_show = range(-70,71,10),
            include_tip = True,
            color = BLUE
           )
            )
        self.add(curve)

class Number2PointTest(Scene):
    def construct(self):
        n_line = NumberLine(
            x_min = -50,
            x_max = 50,
            unit_size = 0.1,
            tick_frequency = 10,
            tick_size = 0.05,
            include_numbers = True,
            numbers_to_show = range(-50,51,10),
        )
        self.add(n_line)
        Tracker = ValueTracker(-50)
        def get_line_obj():
            start_pt = n_line.number_to_point(Tracker.get_value()) #Scale the number into the numberline scale
            end_pt = start_pt+1.5*UP
            arrow = Arrow(
                start_pt,end_pt,
                buff=0, color = RED
            )
            #Create a number object displayable on the screen
            num = DecimalNumber(Tracker.get_value(), color = RED) 
            num.next_to(arrow,UP)
            return VGroup(arrow,num)

        obj = always_redraw(get_line_obj)
        self.add(obj)
        self.play(
            Tracker.set_value,50,
            run_time = 5
        )
        self.wait()

class BuildAxes(Scene):
    def construct(self):
        self.c2p_test()

    def c2p_test(self):
        axes = Axes(
            x_min = -10, x_max = 10,
            y_min = -100, y_max = 100,
        
        number_line_config = {
            "tick_size" : 0.05,
        },
        x_axis_config = {
            "unit_size" : 0.5,
            "tick_frequency":2,
            "include_numbers":True,
            "numbers_to_show": np.append(
                np.arange(-10,0,2),np.arange(2,10,2)
            )            
            },
        y_axis_config = {
            "unit_size":6/200,
            "tick_frequency":50,
            "include_numbers":True,
            "numbers_to_show":[-100,-50,50,100],
            "label_direction":UP,
            "include_tip": True
        })

        self.add(axes)
        self.wait()
        line = Line(
            axes.c2p(-8,-58),axes.c2p(4,50)
        )
        self.play(ShowCreation(line))


class GetGraphTest(Scene):
    def construct(self):

        axes = Axes(
            x_min = 0, x_max = 5,
            y_min = -3, ymax = 3,
        
        number_line_config = {
            "color":RED,
            "stroke_width":3,
        },
        x_axis_config = {
            "unit_size":1,
            "include_numbers":True
        },
        y_axis_config={
            "unit_size":1,
            "tick_frequency":1,
            "include_numbers":True,
            "label_direction":UP
        },
        )
        axes.center().shift(LEFT)
        graph = axes.get_graph(
            lambda t: 2*np.sin(2*PI*2*t),
            color = YELLOW
        )
        self.add(axes,graph)

#We can alternatively use graphsscene class
#which takes care of axes setup intuatively

class GraphSceneTest(GraphScene):
    CONFIG = {
        "x_min":0, "x_max":15,
        "y_min":0,  "y_max":0.5,
        "x_axis_width":15,
        "y_axis_height":5,
        "y_tick_frequency":0.125,
        "graph_origin":2.5*DOWN+5.5*LEFT,
        "axes_color":GREEN,
        "axes_width":1
    }
    def construct(self):
        self.setup_axes(animate=True)
        self.y_axis.add_numbers(
            0.25,0.50,
            number_config={
                "num_decimal_places":2
            },
            direction = LEFT,
        )
        self.x_axis.add_numbers(
            *range(1,15),
        )


        graph1 = self.get_graph(
            lambda x: (x**3/6)*np.exp(-x),
            color = RED,
            stroke_width = 6
        )
        self.wait()
        self.play(ShowCreation(graph1),run_time=2)
        self.wait()
        #Draw Riemann Rectangles
        rect = self.get_riemann_rectangles(graph1,dx=0.5)
        rect.set_color(BLUE)
        rect.set_stroke(RED,1)
        self.play(ShowCreation(rect),run_time = 5)
        self.wait()