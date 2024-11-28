use std::sync::Arc;

use winit::{
    application::ApplicationHandler,
    event::{KeyEvent, WindowEvent},
    event_loop::{ActiveEventLoop, ControlFlow, EventLoop},
    window::Window,
};

use radiance_cascades::{
    drawing::{error::DrawingError, renderer::Renderer},
    math::Vec2,
};

fn new_window(event_loop: &ActiveEventLoop) -> Arc<Window> {
    let window = event_loop
        .create_window(Window::default_attributes())
        .expect("Can't create window");
    Arc::new(window)
}

#[derive(Default)]
struct App {
    window: Option<Arc<Window>>,
    renderer: Option<Renderer>,
    window_resized: bool,
    need_recreate_swapchain: bool,
}
impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &winit::event_loop::ActiveEventLoop) {
        self.init(event_loop);
    }

    fn window_event(
        &mut self,
        event_loop: &winit::event_loop::ActiveEventLoop,
        _window_id: winit::window::WindowId,
        event: winit::event::WindowEvent,
    ) {
        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::RedrawRequested => {
                let res = self.render();
                match res {
                    Err(DrawingError::ObsoleteSwapchain) => self.need_recreate_swapchain = true,
                    _ => {}
                }
            }
            WindowEvent::Resized(_) => self.window_resized = true,
            WindowEvent::KeyboardInput {
                device_id: _,
                event,
                is_synthetic: _,
            } => self.handle_input(event),
            _ => {}
        }

        if self.window_resized {
            self.recreate_swapchain();
            self.resize_viewport();
        }
        if self.need_recreate_swapchain {
            self.recreate_swapchain();
        }
    }
}
impl App {
    fn init(&mut self, event_loop: &ActiveEventLoop) {
        let window = new_window(event_loop);
        let viewport = Vec2::new(1280.0, 720.0);
        let renderer = Renderer::new(viewport, window.clone());

        *self = Self {
            window: Some(window),
            renderer: Some(renderer),
            window_resized: false,
            need_recreate_swapchain: false,
        }
    }

    fn render(&self) -> Result<(), DrawingError> {
        self.renderer.as_ref().unwrap().render()
    }

    fn handle_input(&mut self, _event: KeyEvent) {}

    fn recreate_swapchain(&mut self) {
        self.need_recreate_swapchain = false;
        let window = self.window.as_ref().unwrap();
        self.renderer.as_mut().unwrap().revoke_swapchain(window);
    }

    fn resize_viewport(&mut self) {
        self.window_resized = false;
        let window = self.window.as_ref().unwrap();
        self.renderer.as_mut().unwrap().resize_viewport(window);
    }
}

fn main() {
    let event_loop = EventLoop::new().unwrap();
    let mut app = App::default();
    event_loop.set_control_flow(ControlFlow::Wait);
    event_loop.run_app(&mut app).unwrap();
}
