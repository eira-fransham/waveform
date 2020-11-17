use clap::{App, Arg};
use glutin::dpi::LogicalSize;
#[cfg(windows)]
use glutin::platform::windows::WindowBuilderExtWindows;
use glutin::{
    event::{Event, KeyboardInput, VirtualKeyCode, WindowEvent},
    event_loop::{ControlFlow, EventLoop},
    window::WindowBuilder,
    ContextBuilder, GlProfile, GlRequest,
};
use rodio::Source as _;
use rustfft::{num_complex::Complex32, FFTplanner};
use skia_safe::{
    gpu::{gl::FramebufferInfo, BackendRenderTarget, SurfaceOrigin},
    Color, ColorType, Paint, PaintStyle, Path, Surface,
};

use std::{collections::VecDeque, convert::TryInto, fs, ops, sync::mpsc, time};

type WindowedContext = glutin::ContextWrapper<glutin::PossiblyCurrent, glutin::window::Window>;

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

#[derive(Copy, Clone, PartialEq, Debug)]
struct V2(pub f32, pub f32);

impl V2 {
    fn tup(self) -> (f32, f32) {
        (self.0, self.1)
    }

    fn len2(self) -> f32 {
        self.0 * self.0 + self.1 * self.1
    }

    fn len(self) -> f32 {
        self.len2().sqrt()
    }

    fn normalize(self) -> Self {
        self / self.len()
    }
}

impl From<(f32, f32)> for V2 {
    fn from((x, y): (f32, f32)) -> Self {
        Self(x, y)
    }
}

impl From<V2> for skia_safe::Point {
    fn from(other: V2) -> Self {
        other.tup().into()
    }
}

impl ops::Add for V2 {
    type Output = Self;

    fn add(self, other: Self) -> Self {
        V2(self.0 + other.0, self.1 + other.1)
    }
}

impl ops::Sub for V2 {
    type Output = Self;

    fn sub(self, other: Self) -> Self {
        V2(self.0 - other.0, self.1 - other.1)
    }
}

impl ops::Mul for V2 {
    type Output = Self;

    fn mul(self, other: Self) -> Self {
        V2(self.0 * other.0, self.1 * other.1)
    }
}

impl ops::Mul<f32> for V2 {
    type Output = Self;

    fn mul(self, other: f32) -> Self {
        V2(self.0 * other, self.1 * other)
    }
}

impl ops::Mul<V2> for f32 {
    type Output = V2;

    fn mul(self, other: V2) -> V2 {
        other.mul(self)
    }
}

impl ops::Div for V2 {
    type Output = Self;

    fn div(self, other: Self) -> Self {
        V2(self.0 / other.0, self.1 / other.1)
    }
}

impl ops::Div<f32> for V2 {
    type Output = Self;

    fn div(self, other: f32) -> Self {
        V2(self.0 / other, self.1 / other)
    }
}

struct RectF {
    pub x: f32,
    pub y: f32,
    pub w: f32,
    pub h: f32,
}

impl RectF {
    fn new((x, y): (f32, f32), (w, h): (f32, f32)) -> Self {
        RectF { x, y, w, h }
    }

    fn width(&self) -> f32 {
        self.w
    }

    fn height(&self) -> f32 {
        self.h
    }

    fn lower_left(&self) -> V2 {
        V2(self.x, self.y + self.h)
    }
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

        let calc_index = |i: f32| (i / self.len() as f32).powi(2) * (self.levels.len() as f32 / 2.);

        let (lower, higher) = (
            calc_index(i as f32 - 0.5).floor() as usize,
            calc_index(i as f32 + 0.5).ceil() as usize,
        );
        let higher = higher.max(lower + 1);
        let level = self.levels[lower.max(0)..higher.min(self.levels.len())]
            .iter()
            .map(|c| c.norm())
            .sum::<f32>();
        let level = (level.abs() + 1.).log10().powi(4).max(0.).min(100.) / 100.;
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

    fn draw(&mut self, rect: RectF, dt: f32) -> Option<(Path, Path)> {
        const CURVE_CONTROL_DIST: f32 = 1000.0;

        let dist = CURVE_CONTROL_DIST / self.len() as f32;

        self.scratch.clear();
        self.scratch.extend(self.next.iter().copied());

        let cur_count = self.scratch.len();

        self.fft_planner
            .plan_fft(cur_count)
            .process(&mut self.scratch, &mut self.levels[..cur_count]);

        let mut last_path = Path::new();
        let mut highest_path = Path::new();

        let count = self.last_levels.capacity();

        let factor = V2(rect.width() / (count - 1) as f32, -rect.height());

        let (mut last, mut highest_last) = self
            .update_level(0, dt)
            .map(|(last, highest)| (V2(0., last) * factor, V2(0., highest) * factor))?;

        last_path.move_to(rect.lower_left() + last);
        highest_path.move_to(rect.lower_left() + highest_last);

        let (mut cur, mut highest_cur) = self
            .update_level(1, dt)
            .map(|(last, highest)| (V2(1., last) * factor, V2(1., highest) * factor))?;

        let mut last_control = last + (cur - last).normalize() * dist;
        let mut highest_last_control =
            highest_last + (highest_cur - highest_last).normalize() * dist;

        for i in 2..count {
            let (next, highest_next) = self.update_level(i, dt).map(|(last, highest)| {
                (V2(i as f32, last) * factor, V2(i as f32, highest) * factor)
            })?;

            let offset = (next - last).normalize();
            let highest_offset = (highest_next - highest_last).normalize();
            last_path.cubic_to(
                rect.lower_left() + last_control,
                rect.lower_left() + cur - offset * dist,
                rect.lower_left() + cur,
            );
            highest_path.cubic_to(
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

        last_path.cubic_to(
            rect.lower_left() + last_control,
            rect.lower_left() + cur - last.normalize() * dist,
            rect.lower_left() + cur,
        );
        highest_path.cubic_to(
            rect.lower_left() + highest_last_control,
            rect.lower_left() + highest_cur - (highest_cur - highest_last).normalize() * dist,
            rect.lower_left() + highest_cur,
        );

        Some((last_path, highest_path))
    }
}

fn main() {
    const WIDTH: usize = 800;
    const HEIGHT: usize = 600;
    const FFT_BUFSIZE: usize = 44100 / 8;
    const EQ_NODES: usize = 200;

    let fft_buf_size = FFT_BUFSIZE.to_string();
    let eq_nodes = EQ_NODES.to_string();

    let matches = App::new("Equaliser")
        .arg(
            Arg::with_name("fft_buf_size")
                .short("f")
                .long("fft-size")
                .help("Sets the size of the FFT buffer")
                .default_value(&fft_buf_size)
                .takes_value(true),
        )
        .arg(
            Arg::with_name("eq_nodes")
                .short("e")
                .long("eq-nodes")
                .help("Sets the number of nodes on the equalizer")
                .default_value(&eq_nodes)
                .takes_value(true),
        )
        .arg(
            Arg::with_name("INPUT")
                .help("Sets the audio file to play")
                .required(true)
                .index(1),
        )
        .get_matches();
    let filename = std::path::Path::new(matches.value_of_os("INPUT").unwrap());
    let fft_buf_size: usize = matches
        .value_of("fft_buf_size")
        .unwrap()
        .parse()
        .expect("`fft-size`: not a number");
    let eq_nodes: usize = matches
        .value_of("eq_nodes")
        .unwrap()
        .parse()
        .expect("`eq_nodes`: not a number");

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
                self.sender
                    .send(self.accumulator.drain(..).take(channels).sum::<f32>() / channels as f32)
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
    #[cfg(windows)]
    let window_builder = window_builder.with_drag_and_drop(false);

    // Create an OpenGL 3.x context for Pathfinder to use.
    let gl_context = ContextBuilder::new()
        .with_depth_buffer(0)
        .with_stencil_buffer(8)
        .with_pixel_format(24, 8)
        .with_double_buffer(Some(true))
        .build_windowed(window_builder, &event_loop)
        .unwrap();

    // Load OpenGL, and make the context current.
    let gl_context = unsafe { gl_context.make_current().unwrap() };

    gl::load_with(|name| gl_context.get_proc_address(name));

    let mut gr_context = skia_safe::gpu::Context::new_gl(None).unwrap();

    let fb_info = {
        let mut fboid: gl::types::GLint = 0;
        unsafe { gl::GetIntegerv(gl::FRAMEBUFFER_BINDING, &mut fboid) };

        FramebufferInfo {
            fboid: fboid.try_into().unwrap(),
            format: skia_safe::gpu::gl::Format::RGBA8.into(),
        }
    };

    fn create_surface(
        windowed_context: &WindowedContext,
        fb_info: &FramebufferInfo,
        gr_context: &mut skia_safe::gpu::Context,
    ) -> skia_safe::Surface {
        let pixel_format = windowed_context.get_pixel_format();
        let size = windowed_context.window().inner_size();
        let backend_render_target = BackendRenderTarget::new_gl(
            (
                size.width.try_into().unwrap(),
                size.height.try_into().unwrap(),
            ),
            pixel_format.multisampling.map(|s| s.try_into().unwrap()),
            pixel_format.stencil_bits.try_into().unwrap(),
            *fb_info,
        );
        Surface::from_backend_render_target(
            gr_context,
            &backend_render_target,
            SurfaceOrigin::BottomLeft,
            ColorType::RGBA8888,
            None,
            None,
        )
        .unwrap()
    };

    let mut surface = create_surface(&gl_context, &fb_info, &mut gr_context);
    let sf = gl_context.window().scale_factor() as f32;
    surface.canvas().scale((sf, sf));

    let mut eq = Eq::new(eq_nodes, fft_buf_size);

    let mut last = time::Instant::now();

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
        let size = (size.width as f32, size.height as f32);

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
                surface = create_surface(&gl_context, &fb_info, &mut gr_context);
            }
            _ => {
                if let Some((last_path, highest_path)) = eq.draw(RectF::new((0., 0.), size), dt) {
                    let mut paint = Paint::default();

                    let canvas = surface.canvas();
                    canvas.clear(Color::BLACK);

                    paint.set_stroke_width(3.0);
                    paint.set_style(PaintStyle::Stroke);
                    paint.set_color(0xff_ff_00_00);

                    canvas.draw_path(&highest_path, &paint);

                    paint.set_color(0xff_ff_ff_ff);

                    canvas.draw_path(&last_path, &paint);
                }

                surface.canvas().flush();
                gl_context.swap_buffers().unwrap();

                let now = time::Instant::now();
                let this_dt = now - last;
                dt = this_dt.as_secs_f32();
                times.push(this_dt);
                last = now;
            }
        }
    });
}
