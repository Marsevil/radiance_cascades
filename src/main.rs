use std::cell::RefCell;
use std::sync::Arc;

use rand::{thread_rng, Rng};
use thiserror::Error;
use vulkano::{
    buffer::{Buffer, BufferCreateInfo, BufferUsage, Subbuffer},
    command_buffer::{
        AutoCommandBufferBuilder, CommandBufferUsage, CopyBufferInfo, PrimaryAutoCommandBuffer,
        PrimaryCommandBufferAbstract, RenderPassBeginInfo, SubpassBeginInfo, SubpassContents,
        SubpassEndInfo,
    },
    memory::allocator::{AllocationCreateInfo, MemoryTypeFilter},
    swapchain::{self, SwapchainPresentInfo},
    sync::{self, GpuFuture},
    Validated, VulkanError,
};
use winit::{
    application::ApplicationHandler,
    event::{KeyEvent, WindowEvent},
    event_loop::{ActiveEventLoop, ControlFlow, EventLoop},
    keyboard::{KeyCode, PhysicalKey},
    window::Window,
};

use radiance_cascades::{
    drawing::{context::DrawingContext, vulkan_helper},
    geometry::{Vertex2D, Vertex2DBuilder},
};

fn new_window(event_loop: &ActiveEventLoop) -> Arc<Window> {
    let window = event_loop
        .create_window(Window::default_attributes())
        .expect("Can't create window");
    Arc::new(window)
}

fn get_command_buffers(
    vk_ctx: &vulkan_helper::VulkanState,
    ctx: &DrawingContext,
    vertex_buf: &Subbuffer<[Vertex2D]>,
) -> Box<[Arc<PrimaryAutoCommandBuffer>]> {
    ctx.framebuffers
        .iter()
        .map(|framebuf| {
            let mut builder = AutoCommandBufferBuilder::primary(
                &ctx.command_buffer_allocator,
                vk_ctx.queue.queue_family_index(),
                CommandBufferUsage::MultipleSubmit,
            )
            .unwrap();

            builder
                .begin_render_pass(
                    RenderPassBeginInfo {
                        clear_values: vec![Some([0.1, 0.1, 0.1, 0.1].into())],
                        ..RenderPassBeginInfo::framebuffer(framebuf.clone())
                    },
                    SubpassBeginInfo {
                        contents: SubpassContents::Inline,
                        ..Default::default()
                    },
                )
                .and_then(|builder| builder.bind_pipeline_graphics(ctx.pipeline.clone()))
                .and_then(|builder| builder.bind_vertex_buffers(0, vertex_buf.clone()))
                .and_then(|builder| builder.draw(vertex_buf.len() as u32, 1, 0, 0))
                .and_then(|builder| builder.end_render_pass(SubpassEndInfo::default()))
                .unwrap();

            builder.build().unwrap()
        })
        .collect()
}

#[derive(Debug, Error)]
enum DrawingError {
    #[error("Swapchain needs to be recreate")]
    ObsoleteSwapchain,
}

#[derive(Default)]
struct App {
    window: Option<Arc<Window>>,
    vk_state: Option<vulkan_helper::VulkanState>,
    ctx: Option<DrawingContext>,
    command_buffers: Option<Box<[Arc<PrimaryAutoCommandBuffer>]>>,
    vertex_buf: Option<Subbuffer<[Vertex2D]>>,
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
        let vk_state = vulkan_helper::VulkanState::new(&window);
        let ctx = DrawingContext::new(&vk_state, &window);

        let vertexes = [
            Vertex2DBuilder::new([-0.5, -0.5])
                .color([1.0, 0.0, 0.0])
                .build(),
            Vertex2DBuilder::new([0.0, 0.5])
                .color([0.0, 1.0, 0.0])
                .build(),
            Vertex2DBuilder::new([0.5, -0.25])
                .color([0.0, 0.0, 1.0])
                .build(),
        ];
        let vertex_buf = Buffer::from_iter(
            ctx.buffer_allocator.clone(),
            BufferCreateInfo {
                usage: BufferUsage::VERTEX_BUFFER | BufferUsage::TRANSFER_DST,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                    | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                ..Default::default()
            },
            vertexes,
        )
        .unwrap();

        let command_buffers = get_command_buffers(&vk_state, &ctx, &vertex_buf);

        self.window = Some(window);
        self.vk_state = Some(vk_state);
        self.ctx = Some(ctx);
        self.command_buffers = Some(command_buffers);
        self.vertex_buf = Some(vertex_buf);
    }

    fn render(&self) -> Result<(), DrawingError> {
        let vk_state = self.vk_state.as_ref().unwrap();
        let ctx = self.ctx.as_ref().unwrap();
        let command_buffers = self.command_buffers.as_ref().unwrap();

        let (image_idx, suboptimal, acquire_future) =
            match swapchain::acquire_next_image(ctx.swapchain.clone(), None)
                .map_err(Validated::unwrap)
            {
                Ok(res) => res,
                Err(VulkanError::OutOfDate) => {
                    return Err(DrawingError::ObsoleteSwapchain);
                }
                Err(e) => panic!("Can't acquire the next image: {e}"),
            };
        if suboptimal {
            return Err(DrawingError::ObsoleteSwapchain);
        }

        let exec = sync::now(vk_state.device.clone())
            .join(acquire_future)
            .then_execute(
                vk_state.queue.clone(),
                command_buffers[image_idx as usize].clone(),
            )
            .map(|fut| {
                fut.then_swapchain_present(
                    vk_state.queue.clone(),
                    SwapchainPresentInfo::swapchain_image_index(ctx.swapchain.clone(), image_idx),
                )
            })
            .map(|fut| fut.then_signal_fence_and_flush())
            .unwrap();

        exec.and_then(|exec| exec.wait(None))
            .expect("Rendering failed");

        Ok(())
    }

    fn handle_input(&mut self, event: KeyEvent) {
        let is_space: bool = event.physical_key == PhysicalKey::Code(KeyCode::Space);
        let is_pressed: bool = event.state.is_pressed();
        if !is_space || !is_pressed {
            return;
        }

        let vk_state = self.vk_state.as_ref().unwrap();
        let ctx = self.ctx.as_ref().unwrap();
        let command_allocator = &ctx.command_buffer_allocator;

        let rng = RefCell::new(thread_rng());
        let gen_coords = || {
            let mut rng = rng.borrow_mut();
            [rng.gen_range(-1.0..1.0), rng.gen_range(-1.0..1.0)]
        };
        let gen_color = || {
            let mut rng = rng.borrow_mut();
            [rng.gen(), rng.gen(), rng.gen()]
        };
        let vertexes = [
            Vertex2DBuilder::new(gen_coords())
                .color(gen_color())
                .build(),
            Vertex2DBuilder::new(gen_coords())
                .color(gen_color())
                .build(),
            Vertex2DBuilder::new(gen_coords())
                .color(gen_color())
                .build(),
        ];
        let vertex_buf = Buffer::from_iter(
            ctx.buffer_allocator.clone(),
            BufferCreateInfo {
                usage: BufferUsage::TRANSFER_SRC,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_HOST
                    | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                ..Default::default()
            },
            vertexes,
        )
        .unwrap();

        let cmd = AutoCommandBufferBuilder::primary(
            command_allocator,
            vk_state.queue.queue_family_index(),
            CommandBufferUsage::OneTimeSubmit,
        )
        .map(|mut builder| {
            builder
                .copy_buffer(CopyBufferInfo::buffers(
                    vertex_buf,
                    self.vertex_buf.as_ref().unwrap().clone(),
                ))
                .unwrap();
            builder
        })
        .and_then(|builder| builder.build())
        .unwrap();

        cmd.execute(vk_state.queue.clone())
            .unwrap()
            .then_signal_fence_and_flush()
            .and_then(|fut| fut.wait(None))
            .unwrap();

        self.render().unwrap();
    }

    fn recreate_swapchain(&mut self) {
        self.need_recreate_swapchain = false;

        let window = self.window.as_ref().unwrap();

        let mut ctx = self.ctx.take().unwrap();
        ctx = ctx.revoke_swapchain(window);
        self.ctx = Some(ctx);
    }

    fn resize_viewport(&mut self) {
        self.window_resized = false;

        let window = self.window.as_ref().unwrap();
        let vk_state = self.vk_state.as_ref().unwrap();
        let vertex_buf = self.vertex_buf.as_ref().unwrap();

        let mut ctx = self.ctx.take().unwrap();
        ctx = ctx.revoke_swapchain(window);
        ctx = ctx.resize_viewport(vk_state, window);

        let new_command_buffers = get_command_buffers(vk_state, &ctx, vertex_buf);

        self.ctx = Some(ctx);
        self.command_buffers = Some(new_command_buffers);
    }
}

fn main() {
    let event_loop = EventLoop::new().unwrap();
    let mut app = App::default();
    event_loop.set_control_flow(ControlFlow::Wait);
    event_loop.run_app(&mut app).unwrap();
}
