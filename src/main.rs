use std::sync::Arc;

use thiserror::Error;
use vulkano::{
    buffer::{Buffer, BufferCreateInfo, BufferUsage, Subbuffer},
    command_buffer::{
        AutoCommandBufferBuilder, CommandBufferUsage, PrimaryAutoCommandBuffer,
        RenderPassBeginInfo, SubpassBeginInfo, SubpassContents, SubpassEndInfo,
    },
    memory::allocator::{AllocationCreateInfo, MemoryTypeFilter},
    swapchain::{self, SwapchainPresentInfo},
    sync::{self, GpuFuture},
    Validated, VulkanError,
};
use winit::{
    application::ApplicationHandler,
    event::WindowEvent,
    event_loop::{ActiveEventLoop, ControlFlow, EventLoop},
    window::Window,
};

use radiance_cascades::{
    drawing::{context::DrawingContext, vulkan_helper},
    geometry::Vertex2D,
};

fn new_window(event_loop: &ActiveEventLoop) -> Arc<Window> {
    let window = Arc::new(
        event_loop
            .create_window(Window::default_attributes())
            .expect("Can't create window"),
    );

    window
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
                vk_ctx.graphics_queue_family_idx,
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
                let res = self.draw();
                match res {
                    Err(DrawingError::ObsoleteSwapchain) => self.need_recreate_swapchain = true,
                    _ => {}
                }
            }
            WindowEvent::Resized(_) => self.window_resized = true,
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
        let vk_state = vulkan_helper::VulkanState::new(event_loop, &window);
        let ctx = DrawingContext::new(&vk_state, &window);

        let vertex_buf = Buffer::from_iter(
            ctx.buffer_allocator.clone(),
            BufferCreateInfo {
                usage: BufferUsage::VERTEX_BUFFER,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                    | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                ..Default::default()
            },
            vec![
                Vertex2D::from([-0.5, -0.5]),
                Vertex2D::from([0.0, 0.5]),
                Vertex2D::from([0.5, -0.25]),
            ],
        )
        .unwrap();

        let command_buffers = get_command_buffers(&vk_state, &ctx, &vertex_buf);

        self.window = Some(window);
        self.vk_state = Some(vk_state);
        self.ctx = Some(ctx);
        self.command_buffers = Some(command_buffers);
        self.vertex_buf = Some(vertex_buf);
    }

    fn draw(&self) -> Result<(), DrawingError> {
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
            .unwrap()
            .then_swapchain_present(
                vk_state.queue.clone(),
                SwapchainPresentInfo::swapchain_image_index(ctx.swapchain.clone(), image_idx),
            )
            .then_signal_fence_and_flush();

        exec.and_then(|exec| exec.wait(None))
            .expect("Rendering failed");

        Ok(())
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
