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
    highest_levels: Vec<f32>,
    next: VecDeque<Complex32>,
    buf_len: usize,
    len: usize,
}

impl Eq {
    fn new(num_levels: usize, buffer_size: usize) -> Self {
        Self {
            fft_planner: FFTplanner::new(false),
            scratch: Vec::with_capacity(buffer_size),
            levels: vec![Default::default(); buffer_size],
            last_levels: Vec::with_capacity(num_levels),
            highest_levels: Vec::with_capacity(num_levels),
            next: VecDeque::with_capacity(buffer_size),
            buf_len: buffer_size,
            len: num_levels,
        }
    }

    fn len(&self) -> usize {
        self.len
    }

    fn push(&mut self, sample: f32) {
        let sample = Complex32 { re: sample, im: 0. };

        while self.next.len() >= self.buf_len {
            self.next.pop_front();
        }

        self.next.push_back(sample);
    }

    fn update_level(&mut self, i: usize, dt: f32) -> Option<(f32, f32)> {
        const FRACTION: f32 = 120.;
        const HIGHEST_FRACTION: f32 = 1.0;

        let frac = (FRACTION * dt).min(1.);
        let highest_frac = (HIGHEST_FRACTION * dt).min(1.);

        let actual_index =
            ((i as f32) / self.len() as f32).powi(2) * (self.levels.len() as f32 / 2.);
        let (lower, higher) = (actual_index.floor() as usize, actual_index.ceil() as usize);
        let prop = actual_index - lower as f32;
        let level =
            self.levels.get(lower)?.norm() * prop + self.levels.get(higher)?.norm() * (1. - prop);
        let level = level.abs().log10().powi(4).max(0.).min(100.) / 100.;
        let last = match self.last_levels.get_mut(i) {
            Some(last) => {
                let out = *last * (1. - frac) + level * frac;

                *last = out;

                out
            }
            None => {
                debug_assert_eq!(i, self.last_levels.len());

                self.last_levels.push(level);

                level
            }
        };

        let highest = match self.highest_levels.get_mut(i) {
            Some(highest) => {
                let out = (*highest * (1. - highest_frac)).max(last);

                *highest = out;

                out
            }
            None => {
                debug_assert_eq!(i, self.highest_levels.len());

                self.highest_levels.push(level);

                level
            }
        };

        Some((last, highest))
    }

    fn draw(&mut self, rect: RectF, dt: f32) -> Option<(Path2D, Path2D)> {
        const CURVE_CONTROL_DIST: f32 = 1000.0;

        let dist = CURVE_CONTROL_DIST / self.len() as f32;

        self.scratch.clear();
        self.scratch.extend(self.next.iter().copied());

        let cur_count = self.scratch.len();

        self.fft_planner
            .plan_fft(cur_count)
            .process(&mut self.scratch, &mut self.levels[..cur_count]);

        let mut last_path = Path2D::new();
        let mut highest_path = Path2D::new();

        let count = self.last_levels.capacity();

        let factor = vec2f(rect.width() / (count - 1) as f32, -rect.height());

        let (mut last, mut highest_last) = self
            .update_level(0, dt)
            .map(|(last, highest)| (vec2f(0., last) * factor, vec2f(0., highest) * factor))?;

        last_path.move_to(rect.lower_left() + last);
        highest_path.move_to(rect.lower_left() + highest_last);

        let (mut cur, mut highest_cur) = self
            .update_level(1, dt)
            .map(|(last, highest)| (vec2f(1., last) * factor, vec2f(1., highest) * factor))?;

        let mut last_control = last + (cur - last).normalize() * dist;
        let mut highest_last_control =
            highest_last + (highest_cur - highest_last).normalize() * dist;

        for i in 2..count {
            let (next, highest_next) = self.update_level(i, dt).map(|(last, highest)| {
                (
                    vec2f(i as f32, last) * factor,
                    vec2f(i as f32, highest) * factor,
                )
            })?;

            let offset = (next - last).normalize();
            let highest_offset = (highest_next - highest_last).normalize();
            last_path.bezier_curve_to(
                rect.lower_left() + last_control,
                rect.lower_left() + cur - offset * dist,
                rect.lower_left() + cur,
            );
            highest_path.bezier_curve_to(
                rect.lower_left() + highest_last_control,
                rect.lower_left() + highest_cur - highest_offset * dist,
                rect.lower_left() + highest_cur,
            );

            last = cur;
            last_control = cur + offset * dist;
            cur = next;
            highest_last = highest_cur;
            highest_last_control = highest_cur + highest_offset * dist;
            highest_cur = highest_next;
        }

        last_path.bezier_curve_to(
            rect.lower_left() + last_control,
            rect.lower_left() + cur - (cur - last).normalize() * dist,
            rect.lower_left() + cur,
        );
        highest_path.bezier_curve_to(
            rect.lower_left() + highest_last_control,
            rect.lower_left() + highest_cur - (highest_cur - highest_last).normalize() * dist,
            rect.lower_left() + highest_cur,
        );

        Some((last_path, highest_path))
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

    struct SendSource<Src> {
        sender: mpsc::Sender<f32>,
        inner: Src,
        accumulator: Vec<f32>,
    }

    impl<Src> Iterator for SendSource<Src>
    where
        Src: rodio::Source<Item = f32>,
    {
        type Item = Src::Item;

        fn next(&mut self) -> Option<Src::Item> {
            let inner_next = self.inner.next()?;

            self.accumulator.push(inner_next);

            let channels = self.inner.channels() as usize;

            if self.accumulator.len() >= channels {
                // self.sender
                //     .send(self.accumulator.drain(..).take(channels).sum::<f32>() / channels as f32)
                //     .expect("Failure");
                self.sender
                    .send(self.accumulator.drain(..).next().unwrap())
                    .expect("Failure");
            }

            Some(inner_next)
        }
    }

    impl<Src> rodio::Source for SendSource<Src>
    where
        Src: rodio::Source<Item = f32>,
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

    impl<Src> SendSource<Src>
    where
        Src: rodio::Source<Item = f32>,
    {
        fn new(inner: Src) -> (Self, mpsc::Receiver<Src::Item>) {
            let (sender, recv) = mpsc::channel();

            let channels = inner.channels() as usize;
            (
                Self {
                    inner,
                    sender,
                    accumulator: Vec::with_capacity(channels),
                },
                recv,
            )
        }
    }

    let (source, recv) = SendSource::new(decoder.convert_samples());

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

    let mut eq = Eq::new(50, 44100 / 8);

    // Render the canvas to screen.
    let mut scene = SceneProxy::from_scene(scene, renderer.mode().level, RayonExecutor);
    let build_options = BuildOptions::default();
    let mut last = time::Instant::now();
    scene.build_and_render(&mut renderer, build_options.clone());
    gl_context.swap_buffers().unwrap();
    let now = time::Instant::now();

    let num_frames = 1000;
    let mut times = Vec::with_capacity(num_frames);
    let mut dt: f32 = 0.;
    times.push(now - last);

    last = now;

    player.append(source);

    event_loop.run(move |event, _, control_flow| {
        *control_flow = ControlFlow::Poll;

        while let Some(val) = recv.try_recv().ok() {
            eq.push(val);
        }

        let size = gl_context.window().inner_size();
        let size = vec2f(size.width as f32, size.height as f32);

        if times.len() >= num_frames {
            let avg = times.drain(..).take(num_frames).sum::<time::Duration>() / num_frames as u32;

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

                if let Some((last_path, highest_path)) =
                    eq.draw(RectF::new(vec2f(0., 0.), size), dt)
                {
                    context.set_line_width(3.0);

                    context.set_stroke_style(rgbau(255, 255, 255, 255));
                    context.stroke_path(last_path);

                    context.set_stroke_style(rgbau(255, 0, 0, 255));
                    context.stroke_path(highest_path);

                    let canvas = context.into_canvas();
                    scene.replace_scene(canvas.into_scene());
                    scene.set_view_box(RectF::new(Vector2F::zero(), size));
                    scene.build_and_render(&mut renderer, build_options.clone());
                    gl_context.swap_buffers().unwrap();
                }

                let now = time::Instant::now();
                let this_dt = now - last;
                dt = this_dt.as_secs_f32();
                times.push(this_dt);
                last = now;
            }
        }
    });
}
