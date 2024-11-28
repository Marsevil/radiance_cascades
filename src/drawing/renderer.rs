use std::sync::Arc;

use glam::Vec2;
use vulkano::{
    buffer::{Buffer, BufferCreateInfo, BufferUsage, Subbuffer},
    command_buffer::{
        AutoCommandBufferBuilder, CommandBufferExecFuture, CommandBufferUsage,
        PrimaryAutoCommandBuffer, RenderPassBeginInfo, SubpassBeginInfo, SubpassContents,
        SubpassEndInfo,
    },
    device::DeviceOwned,
    memory::allocator::{AllocationCreateInfo, MemoryTypeFilter},
    swapchain::{self, SwapchainAcquireFuture, SwapchainPresentInfo},
    sync::{
        self,
        future::{JoinFuture, NowFuture},
        GpuFuture,
    },
    Validated, VulkanError,
};
use winit::window::Window;

use super::{
    error::DrawingError,
    objects::{
        discretable::{discrete_light, discrete_wall},
        Light, Wall,
    },
    vs,
    vulkan_helper::{VulkanContext, VulkanCore},
};
use crate::geometry::vertex::{Vertex2D, Vertex2DBuilder};

fn get_vertex_buffer(vk_ctx: &VulkanContext, vertexes: Box<[Vertex2D]>) -> Subbuffer<[Vertex2D]> {
    Buffer::from_iter(
        vk_ctx.buffer_allocator.clone(),
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
    .unwrap()
}

fn get_uniform_buffer(
    vk_ctx: &VulkanContext,
    position: Vec2,
    viewport: Vec2,
) -> Subbuffer<vs::Data> {
    let buffer = vk_ctx.uniform_buffer_allocator.allocate_sized().unwrap();
    *buffer.write().unwrap() = vs::Data {
        viewport: viewport.into(),
        position: position.into(),
    };

    buffer
}

fn get_command_buffer(
    vk_core: &VulkanCore,
    vk_ctx: &VulkanContext,
    frame_idx: u32,
    vertex_buf: Subbuffer<[Vertex2D]>,
) -> Arc<PrimaryAutoCommandBuffer> {
    let framebuffer = vk_ctx.framebuffers.get(frame_idx as usize).unwrap();
    let mut builder = AutoCommandBufferBuilder::primary(
        &vk_ctx.command_buffer_allocator,
        vk_core.queue.queue_family_index(),
        CommandBufferUsage::OneTimeSubmit,
    )
    .unwrap();

    let buffer_len = vertex_buf.len();
    builder
        .begin_render_pass(
            RenderPassBeginInfo {
                clear_values: vec![Some([0.1, 0.1, 0.1, 0.1].into())],
                ..RenderPassBeginInfo::framebuffer(framebuffer.clone())
            },
            SubpassBeginInfo {
                contents: SubpassContents::Inline,
                ..Default::default()
            },
        )
        .and_then(|builder| builder.bind_pipeline_graphics(vk_ctx.pipeline.clone()))
        .and_then(|builder| builder.bind_vertex_buffers(0, vertex_buf))
        .and_then(|builder| builder.draw(buffer_len as u32, 1, 0, 0))
        .and_then(|builder| builder.end_render_pass(SubpassEndInfo::default()))
        .unwrap();

    builder.build().unwrap()
}

pub struct Renderer {
    /// Size of the viewport
    viewport: Vec2,
    /// List of all the light geometry
    lights: Vec<Light>,
    /// List of all the wall geometry
    walls: Vec<Wall>,

    vk_core: VulkanCore,
    vk_ctx: VulkanContext,
    // command_buffers: Option<Box<[Arc<PrimaryAutoCommandBuffer>]>>,
    // vertex_buf: Option<Subbuffer<[Vertex2D]>>,
}

impl Renderer {
    pub fn new(viewport: Vec2, window: Arc<Window>) -> Self {
        let vk_core = VulkanCore::new(window.clone());
        let vk_ctx = VulkanContext::new(&vk_core, &window);

        Self {
            viewport,
            lights: Default::default(),
            walls: Default::default(),
            vk_core,
            vk_ctx,
            // command_buffers: None,
            // vertex_buf: None,
        }
    }

    pub fn add_light(&mut self, light: Light) {
        self.lights.push(light);
    }

    pub fn add_walls(&mut self, wall: Wall) {
        self.walls.push(wall);
    }

    pub fn render(&self) -> Result<(), DrawingError> {
        // Acquire next image
        let (image_idx, suboptimal, acquire_future) =
            swapchain::acquire_next_image(self.vk_ctx.swapchain.clone(), None)
                .map_err(Validated::unwrap)
                .map_err(DrawingError::from)?;
        if suboptimal {
            return Err(DrawingError::ObsoleteSwapchain);
        }

        // Process buffers and drawing commands
        let light_cmds = self.render_lights_commands(image_idx);
        let wall_cmds = self.render_walls_commands(image_idx);

        // Execute all the drawing commands
        let mut exec = sync::now(self.vk_core.device.clone())
            .join(acquire_future)
            .boxed();
        let queue = &self.vk_core.queue;
        for cmd in light_cmds {
            exec = exec.then_execute(queue.clone(), cmd).unwrap().boxed();
        }
        for cmd in wall_cmds {
            exec = exec.then_execute(queue.clone(), cmd).unwrap().boxed();
        }
        exec = exec
            .then_swapchain_present(
                queue.clone(),
                SwapchainPresentInfo::swapchain_image_index(
                    self.vk_ctx.swapchain.clone(),
                    image_idx,
                ),
            )
            .boxed();
        exec.then_signal_fence_and_flush()
            .unwrap()
            .wait(None)
            .expect("Rendering failed");

        Ok(())
    }

    pub fn revoke_swapchain(&mut self, window: &Window) {
        let mut vk_ctx = unsafe { std::ptr::read(std::ptr::from_ref(&self.vk_ctx)) };
        vk_ctx = vk_ctx.revoke_swapchain(window);
        unsafe { std::ptr::write(std::ptr::from_mut(&mut self.vk_ctx), vk_ctx) };
    }

    pub fn resize_viewport(&mut self, window: &Window) {
        let mut vk_ctx = unsafe { std::ptr::read(std::ptr::from_ref(&self.vk_ctx)) };
        vk_ctx = vk_ctx.revoke_swapchain(window);
        vk_ctx = vk_ctx.resize_viewport(&self.vk_core, window);
        unsafe { std::ptr::write(std::ptr::from_mut(&mut self.vk_ctx), vk_ctx) };
    }
}

impl Renderer {
    fn render_walls_commands(&self, frame_idx: u32) -> Box<[Arc<PrimaryAutoCommandBuffer>]> {
        let vk_core = &self.vk_core;
        let vk_ctx = &self.vk_ctx;
        let walls = self.walls.iter();

        walls
            .map(|wall| {
                let vertexes = Box::new(discrete_wall(wall));
                let vertex_buf = get_vertex_buffer(vk_ctx, vertexes);
                get_command_buffer(vk_core, vk_ctx, frame_idx, vertex_buf)
            })
            .collect()
    }

    fn render_lights_commands(&self, frame_idx: u32) -> Box<[Arc<PrimaryAutoCommandBuffer>]> {
        let vk_core = &self.vk_core;
        let vk_ctx = &self.vk_ctx;
        let lights = self.lights.iter();

        lights
            .map(|light| {
                let vertexes = Box::new(discrete_light(light));
                let vertex_buf = get_vertex_buffer(vk_ctx, vertexes);
                get_command_buffer(vk_core, vk_ctx, frame_idx, vertex_buf)
            })
            .collect()
    }
}
