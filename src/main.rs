use glutin::dpi::LogicalSize;
use glutin::event::{Event, KeyboardInput, VirtualKeyCode, WindowEvent};
use glutin::event_loop::{ControlFlow, EventLoop};
use glutin::window::WindowBuilder;
use glutin::{Api, ContextBuilder, GlProfile, GlRequest};
use pathfinder_canvas::{
    Canvas, CanvasFontContext, CanvasRenderingContext2D, LineJoin, Path2D, Transform2F,
};
use pathfinder_canvas::{TextAlign, TextBaseline};
use pathfinder_color::{rgbau, rgbf, rgbu, ColorF, ColorU};
use pathfinder_geometry::rect::RectF;
use pathfinder_geometry::vector::{vec2f, vec2i, Vector2F};
use pathfinder_gl::{GLDevice, GLVersion};
use pathfinder_renderer::concurrent::rayon::RayonExecutor;
use pathfinder_renderer::concurrent::scene_proxy::SceneProxy;
use pathfinder_renderer::gpu::options::{DestFramebuffer, RendererMode, RendererOptions};
use pathfinder_renderer::gpu::renderer::Renderer;
use pathfinder_renderer::options::{BuildOptions, RenderTransform};
use pathfinder_renderer::scene::Scene;
use pathfinder_resources::embedded::EmbeddedResourceLoader;

use rustfft::num_complex::Complex32;
use rustfft::{FFTplanner, FFT};

use rodio::Source as _;

use std::collections::VecDeque;
use std::env;
use std::fs;
use std::sync::mpsc;
use std::time;

struct Eq {
    fft_planner: FFTplanner<f32>,
    scratch: Vec<Complex32>,
    levels: Vec<Complex32>,
    last_levels: Vec<f32>,
    next: VecDeque<Complex32>,
    factor: f32,
    len: usize,
}

impl Eq {
    fn new(size: usize) -> Self {
        Self {
            fft_planner: FFTplanner::new(false),
            scratch: Vec::with_capacity(size),
            levels: vec![Default::default(); size],
            last_levels: Vec::with_capacity(size / 2),
            next: VecDeque::with_capacity(size),
            factor: (size as f32).sqrt().recip(),
            len: size,
        }
    }

    fn len(&self) -> usize {
        self.len
    }

    fn push(&mut self, sample: f32) {
        let sample = Complex32 {
            re: sample,// * self.factor,
            im: 0.,
        };

        while self.next.len() >= self.len() {
            self.next.pop_front();
        }

        self.next.push_back(sample);
    }

    fn update_level(&mut self, i: usize) -> Option<f32> {
        const FRACTION: f32 = 0.7;
        let level = self.levels.get(i)?.norm();
        match self.last_levels.get_mut(i) {
            Some(last) => {
                let diff = level - *last;

                let out = *last + diff * FRACTION;

                *last = out;

                Some(out)
            }
            None => {
                debug_assert_eq!(i, self.last_levels.len());

                self.last_levels.push(level);

                Some(level)
            }
        }
    }

    fn draw(&mut self, rect: RectF) -> Option<Path2D> {
        const CURVE_CONTROL_DIST: f32 = 1000.0;

        let dist = CURVE_CONTROL_DIST / self.len() as f32;

        self.scratch.clear();
        self.scratch.extend(self.next.iter().copied());

        self.fft_planner
            .plan_fft(self.len())
            .process(&mut self.scratch, &mut self.levels);

        let mut path = Path2D::new();

        let count = self.last_levels.capacity();

        let factor = vec2f(rect.width() / (count - 1) as f32, -rect.height());

        let mut last = vec2f(0., self.update_level(0)?) * factor;

        path.move_to(rect.lower_left() + last);

        let mut cur = vec2f(1., self.update_level(1)?) * factor;
        let mut last_control = last + (cur - last).normalize() * dist;

        for i in 2..count {
            let next = vec2f(i as f32, self.update_level(i)?) * factor;

            let offset = (next - last).normalize();

            path.bezier_curve_to(
                rect.lower_left() + last_control,
                rect.lower_left() + cur - offset * dist,
                rect.lower_left() + cur,
            );

            last = cur;
            last_control = cur + offset * dist;
            cur = next;
        }

        path.bezier_curve_to(
            rect.lower_left() + last_control,
            rect.lower_left() + cur - (cur - last).normalize() * dist,
            rect.lower_left() + cur,
        );

        Some(path)
    }
}

fn main() {
    use rand::Rng;

    const WIDTH: usize = 800;
    const HEIGHT: usize = 600;

    let filename = env::args()
        .skip(1)
        .next()
        .expect("Please pass a sound file to play");

    let decoder =
        rodio::decoder::Decoder::new_looped(fs::File::open(filename).expect("Could not open file"))
            .expect("Could not read file");

    struct SendSource<Src, Smp> {
        sender: mpsc::Sender<Smp>,
        inner: Src,
    }

    impl<Src> Iterator for SendSource<Src, Src::Item>
    where
        Src: rodio::Source,
        Src::Item: rodio::Sample + Copy,
    {
        type Item = Src::Item;

        fn next(&mut self) -> Option<Src::Item> {
            let inner_next = self.inner.next()?;

            self.sender.send(inner_next).expect("Failure");

            Some(inner_next)
        }
    }

    impl<Src> rodio::Source for SendSource<Src, Src::Item>
    where
        Src: rodio::Source,
        Src::Item: rodio::Sample + Copy,
    {
        fn current_frame_len(&self) -> Option<usize> {
            self.inner.current_frame_len()
        }

        fn channels(&self) -> u16 {
            self.inner.channels()
        }

        fn sample_rate(&self) -> u32 {
            self.inner.sample_rate()
        }

        fn total_duration(&self) -> Option<time::Duration> {
            self.inner.total_duration()
        }
    }

    impl<Src> SendSource<Src, Src::Item>
    where
        Src: rodio::Source,
        Src::Item: rodio::Sample,
    {
        fn new(inner: Src) -> (Self, mpsc::Receiver<Src::Item>) {
            let (sender, recv) = mpsc::channel();

            (Self { inner, sender }, recv)
        }
    }

    let (source, recv) = SendSource::<_, f32>::new(decoder.convert_samples());

    let (_stream, handle) = rodio::OutputStream::try_default().expect("Couldn't get audio device");

    let player = rodio::Sink::try_new(&handle).expect("Couldn't get audio device");

    // Calculate the right logical size of the window.
    let event_loop = EventLoop::new();
    let logical_window_size = LogicalSize::new(WIDTH as f64, HEIGHT as f64);

    // Open a window.
    let window_builder = WindowBuilder::new()
        .with_title("Minimal example")
        .with_inner_size(logical_window_size);

    // Create an OpenGL 3.x context for Pathfinder to use.
    let gl_context = ContextBuilder::new()
        .with_gl(GlRequest::Specific(Api::OpenGlEs, (3, 1)))
        .with_gl_profile(GlProfile::Core)
        .build_windowed(window_builder, &event_loop)
        .unwrap();

    let size = gl_context.window().inner_size();

    // Load OpenGL, and make the context current.
    let gl_context = unsafe { gl_context.make_current().unwrap() };
    gl::load_with(|name| gl_context.get_proc_address(name) as *const _);

    let physical_size = gl_context.window().inner_size();

    // Create a Pathfinder renderer.
    let device = GLDevice::new(GLVersion::GLES3, 0);
    let options = RendererOptions {
        background_color: Some(ColorF::new(0., 0., 0., 1.)),
        dest: DestFramebuffer::full_window(vec2i(
            physical_size.width as i32,
            physical_size.height as i32,
        )),
        show_debug_ui: false,
    };
    let resources = EmbeddedResourceLoader;
    let mode = RendererMode::default_for_device(&device);
    let mut renderer = Renderer::new(device, &resources, mode, options);

    // Clear to swf stage background color.
    let mut scene = Scene::new();
    scene.set_view_box(RectF::new(
        Vector2F::zero(),
        vec2f(size.width as f32, size.height as f32),
    ));

    let mut eq = Eq::new(20);

    // Render the canvas to screen.
    let mut scene = SceneProxy::from_scene(scene, renderer.mode().level, RayonExecutor);
    let build_options = BuildOptions::default();
    let mut last = time::Instant::now();
    scene.build_and_render(&mut renderer, build_options.clone());
    gl_context.swap_buffers().unwrap();
    let now = time::Instant::now();

    let num_frames = 100;
    let mut times = Vec::with_capacity(num_frames);
    times.push(now - last);

    last = now;

    player.append(source);

    event_loop.run(move |event, _, control_flow| {
        *control_flow = ControlFlow::Poll;

        while let Some(val) = dbg!(recv.recv()).ok() {
            eq.push(val);
        }

        let size = gl_context.window().inner_size();
        let size = vec2f(size.width as f32, size.height as f32);

        if times.len() >= num_frames {
            let avg = times.drain(..).sum::<time::Duration>() / num_frames as u32;

            println!(
                "{:?} fps",
                time::Duration::new(1, 0).as_nanos() / avg.as_nanos()
            );
        }

        match event {
            Event::WindowEvent {
                event: WindowEvent::CloseRequested,
                ..
            }
            | Event::WindowEvent {
                event:
                    WindowEvent::KeyboardInput {
                        input:
                            KeyboardInput {
                                virtual_keycode: Some(VirtualKeyCode::Escape),
                                ..
                            },
                        ..
                    },
                ..
            } => {
                *control_flow = ControlFlow::Exit;
            }
            Event::WindowEvent {
                event: WindowEvent::Resized(physical_size),
                ..
            } => {
                gl_context.resize(physical_size);
                renderer.options_mut().dest = DestFramebuffer::full_window(vec2i(
                    physical_size.width as i32,
                    physical_size.height as i32,
                ));
                renderer.dest_framebuffer_size_changed();
            }
            _ => {
                scene.set_view_box(RectF::new(Vector2F::zero(), size));

                let mut context = Canvas::new(size)
                    .get_context_2d(pathfinder_canvas::CanvasFontContext::from_system_source());

                context.set_stroke_style(rgbau(255, 0, 0, 255));
                context.set_line_width(3.0);
                context.stroke_path(eq.draw(RectF::new(vec2f(0., 0.), size)).unwrap());

                let canvas = context.into_canvas();
                scene.replace_scene(canvas.into_scene());
                scene.set_view_box(RectF::new(Vector2F::zero(), size));
                scene.build_and_render(&mut renderer, build_options.clone());
                gl_context.swap_buffers().unwrap();
                let now = time::Instant::now();
                times.push(now - last);
                last = now;

                *control_flow =
                    ControlFlow::WaitUntil(now + time::Duration::from_secs_f64(1. / 60.));
            }
        }
    });
}
