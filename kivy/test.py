from kivy.app import App
from kivy.uix.widget import Widget
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.gridlayout import GridLayout
from kivy.uix.image import Image
from kivy.clock import Clock
from kivy.graphics.texture import Texture
from kivy.uix.label import Label

from kivy.config import Config

import cv2

from kivy.core.window import Window
from kivy.graphics import Color, Rectangle

Window.clearcolor = (115/225, 29/225, 38/225, 1)
Window.minimum_height = 720
Window.minimum_width = 1280
Window.toggle_fullscreen()

class CamApp(App):

    def build(self):
        self.img1=Image(size_hint=(0.7, 1.0))
        layout = BoxLayout(orientation='vertical')
        topRow = BoxLayout(orientation='horizontal', size_hint=(1.0, 0.8))
        bottomRow = BoxLayout(orientation='horizontal', size_hint=(1.0, 0.2))


        layout.add_widget(topRow)
        layout.add_widget(bottomRow)

        #topRow.add_widget(Label(text='Left', font_size='40sp', size_hint=(0.2, 1.0)))
        topRow.add_widget(self.img1)
        box_info = BoxLayout(size_hint=(0.28, 0.91))
        info = Label(text='Right', font_size='40sp', size_hint=(0.3, 1.0))
        box_info.add_widget(info)
        topRow.add_widget(box_info)

        title = Label(text='Emotional Recognition Project', font_size='40sp')
        bottomRow.add_widget(Label(text='Emotional Recognition Project', font_size='40sp'))
        #opencv2 stuffs
        self.capture = cv2.VideoCapture(0)
        #cv2.namedWindow("CV2 Image")
        Clock.schedule_interval(self.update, 1.0/33.0)

        with bottomRow.canvas.before:
            Color(0, 1, 0, 1)  # green; colors range from 0-1 instead of 0-255
            bottomRow.rect = Rectangle(size=bottomRow.size,
                                  pos=bottomRow.pos)
        with info.canvas.before:
            Color(0,1,0,1)
            info.rect = Rectangle(size=info.size,pos=info.pos)

        def update_rect(instance, value):
            instance.rect.pos = instance.pos
            instance.rect.size = instance.size

        # listen to size and position changes
        bottomRow.bind(pos=update_rect, size=update_rect)
        info.bind(pos=update_rect, size=update_rect)
        return layout

    def update(self, dt):
        # display image from cam in opencv window
        ret, frame = self.capture.read()
        #cv2.imshow("CV2 Image", frame)
        frame = cv2.resize(frame, (1280, 720), fx=0, fy=0, interpolation=cv2.INTER_CUBIC)
        # convert it to texture
        buf1 = cv2.flip(frame, 0)
        buf = buf1.tostring()

        texture1 = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='bgr') 
        #if working on RASPBERRY PI, use colorfmt='rgba' here instead, but stick with "bgr" in blit_buffer. 
        texture1.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
        # display image from the texture
        self.img1.texture = texture1

if __name__ == '__main__':
    Config.set('graphics', 'width', '1280')
    Config.set('graphics', 'height', '720')
    Config.set('graphics', 'fullscreen', 'auto')
    Config.set('graphics', 'window_state', 'maximized')
    CamApp().run()
    cv2.destroyAllWindows()