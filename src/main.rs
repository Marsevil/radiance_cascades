use std::sync::Arc;

use thiserror::Error;
use vulkano::{
    buffer::{Buffer, BufferContents, BufferCreateInfo, BufferUsage, Subbuffer},
    command_buffer::{
        allocator::{StandardCommandBufferAllocator, StandardCommandBufferAllocatorCreateInfo},
        AutoCommandBufferBuilder, CommandBufferUsage, PrimaryAutoCommandBuffer,
        RenderPassBeginInfo, SubpassBeginInfo, SubpassContents, SubpassEndInfo,
    },
    image::{view::ImageView, Image, ImageUsage},
    memory::allocator::{AllocationCreateInfo, MemoryTypeFilter, StandardMemoryAllocator},
    pipeline::{
        graphics::{
            color_blend::{ColorBlendAttachmentState, ColorBlendState},
            input_assembly::InputAssemblyState,
            multisample::MultisampleState,
            rasterization::RasterizationState,
            vertex_input::{Vertex, VertexDefinition},
            viewport::{Viewport, ViewportState},
            GraphicsPipelineCreateInfo,
        },
        layout::PipelineDescriptorSetLayoutCreateInfo,
        GraphicsPipeline, PipelineLayout, PipelineShaderStageCreateInfo,
    },
    render_pass::{Framebuffer, FramebufferCreateInfo, RenderPass, Subpass},
    shader::ShaderModule,
    swapchain::{self, Swapchain, SwapchainCreateInfo, SwapchainPresentInfo},
    sync::{self, GpuFuture},
    Validated, VulkanError,
};
use winit::{
    application::ApplicationHandler,
    event::WindowEvent,
    event_loop::{ActiveEventLoop, ControlFlow, EventLoop},
    window::Window,
};

mod vulkan_helper;

mod vs {
    vulkano_shaders::shader! {
        ty: "vertex",
        path: "src/shaders/shader.vert",
    }
}
mod fs {
    vulkano_shaders::shader! {
        ty: "fragment",
        path: "src/shaders/shader.frag",
    }
}

#[derive(BufferContents, Vertex)]
#[repr(C)]
struct Vertex2D {
    #[format(R32G32_SFLOAT)]
    position: [f32; 2],
}
impl From<[f32; 2]> for Vertex2D {
    fn from(value: [f32; 2]) -> Self {
        Self { position: value }
    }
}

fn new_window(event_loop: &ActiveEventLoop) -> Arc<Window> {
    let window = Arc::new(
        event_loop
            .create_window(Window::default_attributes())
            .expect("Can't create window"),
    );

    window
}

fn get_swapchain(
    window: &Window,
    vk_ctx: &vulkan_helper::VulkanState,
) -> (Arc<Swapchain>, Box<[Arc<Image>]>) {
    let caps = vk_ctx
        .physical_device
        .surface_capabilities(&vk_ctx.surface, Default::default())
        .expect("Failed to get surface capabilities");
    let dimensions = window.inner_size();
    let composite_alpha = caps.supported_composite_alpha.into_iter().next().unwrap();
    let image_format = vk_ctx
        .physical_device
        .surface_formats(&vk_ctx.surface, Default::default())
        .unwrap()[0]
        .0;
    let (swapchain, images) = Swapchain::new(
        vk_ctx.device.clone(),
        vk_ctx.surface.clone(),
        SwapchainCreateInfo {
            min_image_count: caps.min_image_count,
            image_format,
            image_extent: dimensions.into(),
            image_usage: ImageUsage::COLOR_ATTACHMENT,
            composite_alpha,
            ..Default::default()
        },
    )
    .unwrap();

    (swapchain, images.into_boxed_slice())
}

fn get_framebuffers(
    images: &[Arc<Image>],
    render_pass: &Arc<RenderPass>,
) -> Box<[Arc<Framebuffer>]> {
    images
        .iter()
        .map(|image| {
            let view = ImageView::new_default(image.clone()).unwrap();
            Framebuffer::new(
                render_pass.clone(),
                FramebufferCreateInfo {
                    attachments: vec![view],
                    ..Default::default()
                },
            )
            .unwrap()
        })
        .collect::<Box<_>>()
}

fn get_pipeline(
    viewport: Viewport,
    vk_ctx: &vulkan_helper::VulkanState,
    render_pass: Arc<RenderPass>,
    vs: Arc<ShaderModule>,
    fs: Arc<ShaderModule>,
) -> Arc<GraphicsPipeline> {
    let vs = vs.entry_point("main").unwrap();
    let fs = fs.entry_point("main").unwrap();

    let vertex_input_state = Vertex2D::per_vertex()
        .definition(&vs.info().input_interface)
        .unwrap();
    let stages = [
        PipelineShaderStageCreateInfo::new(vs),
        PipelineShaderStageCreateInfo::new(fs),
    ];
    let layout = PipelineLayout::new(
        vk_ctx.device.clone(),
        PipelineDescriptorSetLayoutCreateInfo::from_stages(&stages)
            .into_pipeline_layout_create_info(vk_ctx.device.clone())
            .unwrap(),
    )
    .unwrap();

    let subpass = Subpass::from(render_pass, 0).unwrap();

    GraphicsPipeline::new(
        vk_ctx.device.clone(),
        None,
        GraphicsPipelineCreateInfo {
            stages: stages.into_iter().collect(),
            vertex_input_state: Some(vertex_input_state),
            input_assembly_state: Some(InputAssemblyState::default()),
            viewport_state: Some(ViewportState {
                viewports: [viewport].into_iter().collect(),
                ..Default::default()
            }),
            rasterization_state: Some(RasterizationState::default()),
            multisample_state: Some(MultisampleState::default()),
            color_blend_state: Some(ColorBlendState::with_attachment_states(
                subpass.num_color_attachments(),
                ColorBlendAttachmentState::default(),
            )),
            subpass: Some(subpass.into()),
            ..GraphicsPipelineCreateInfo::layout(layout)
        },
    )
    .unwrap()
}

fn get_command_buffers(
    command_buffer_allocator: &StandardCommandBufferAllocator,
    vk_ctx: &vulkan_helper::VulkanState,
    pipeline: Arc<GraphicsPipeline>,
    vertex_buf: &Subbuffer<[Vertex2D]>,
    framebuffers: &Box<[Arc<Framebuffer>]>,
) -> Box<[Arc<PrimaryAutoCommandBuffer>]> {
    framebuffers
        .iter()
        .map(|framebuf| {
            let mut builder = AutoCommandBufferBuilder::primary(
                command_buffer_allocator,
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
                .and_then(|builder| builder.bind_pipeline_graphics(pipeline.clone()))
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

struct DrawingContext {
    vk_state: vulkan_helper::VulkanState,
    window: Arc<Window>,
    swapchain: Arc<Swapchain>,
    framebuffers: Box<[Arc<Framebuffer>]>,
    command_buffers: Box<[Arc<PrimaryAutoCommandBuffer>]>,
    render_pass: Arc<RenderPass>,
    vs: Arc<ShaderModule>,
    fs: Arc<ShaderModule>,
    command_buffer_allocator: StandardCommandBufferAllocator,
    vertex_buf: Subbuffer<[Vertex2D]>,
}

#[derive(Default)]
struct App {
    ctx: Option<DrawingContext>,
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
        let vk_ctx = vulkan_helper::VulkanState::new(event_loop, &window);

        let (swapchain, images) = get_swapchain(&window, &vk_ctx);

        let render_pass = vulkano::single_pass_renderpass!(
            vk_ctx.device.clone(),
            attachments: {
                color: {
                    format: swapchain.image_format(),
                    samples: 1,
                    load_op: Clear,
                    store_op: Store,
                },

            },
            pass: {
                color: [color],
                depth_stencil: {},
            },
        )
        .unwrap();

        let framebufs = get_framebuffers(&images, &render_pass);

        let memory_allocator =
            Arc::new(StandardMemoryAllocator::new_default(vk_ctx.device.clone()));
        let vertex_buf = Buffer::from_iter(
            memory_allocator.clone(),
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

        let vs = vs::load(vk_ctx.device.clone()).expect("Can't compile vertex shader");
        let fs = fs::load(vk_ctx.device.clone()).expect("Can't compile fragment shader");

        let viewport = Viewport {
            offset: [0.0, 0.0],
            extent: window.inner_size().into(),
            depth_range: 0.0..=1.0,
        };

        let pipeline = get_pipeline(
            viewport,
            &vk_ctx,
            render_pass.clone(),
            vs.clone(),
            fs.clone(),
        );

        let command_buffer_allocator = StandardCommandBufferAllocator::new(
            vk_ctx.device.clone(),
            StandardCommandBufferAllocatorCreateInfo::default(),
        );

        let command_buffers = get_command_buffers(
            &command_buffer_allocator,
            &vk_ctx,
            pipeline.clone(),
            &vertex_buf,
            &framebufs,
        );

        let ctx = DrawingContext {
            vk_state: vk_ctx,
            window,
            swapchain,
            command_buffers,
            render_pass,
            framebuffers: framebufs,
            vs,
            fs,
            command_buffer_allocator,
            vertex_buf,
        };
        self.ctx = Some(ctx);
    }

    fn draw(&self) -> Result<(), DrawingError> {
        let ctx = self.ctx.as_ref().unwrap();
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

        let exec = sync::now(ctx.vk_state.device.clone())
            .join(acquire_future)
            .then_execute(
                ctx.vk_state.queue.clone(),
                ctx.command_buffers[image_idx as usize].clone(),
            )
            .unwrap()
            .then_swapchain_present(
                ctx.vk_state.queue.clone(),
                SwapchainPresentInfo::swapchain_image_index(ctx.swapchain.clone(), image_idx),
            )
            .then_signal_fence_and_flush();

        exec.and_then(|exec| exec.wait(None))
            .expect("Rendering failed");

        Ok(())
    }

    fn recreate_swapchain(&mut self) {
        self.need_recreate_swapchain = false;
        let ctx = self.ctx.as_mut().unwrap();
        let new_dimensions = ctx.window.inner_size();
        let (new_swapchain, new_images) = ctx
            .swapchain
            .recreate(SwapchainCreateInfo {
                image_extent: new_dimensions.into(),
                ..ctx.swapchain.create_info()
            })
            .expect("Failed to recreate swapchain {e");

        let new_framebuffers = get_framebuffers(&new_images, &ctx.render_pass);

        ctx.swapchain = new_swapchain;
        ctx.framebuffers = new_framebuffers;
    }

    fn resize_viewport(&mut self) {
        self.window_resized = false;
        let ctx = self.ctx.as_mut().unwrap();
        let viewport = Viewport {
            offset: [0.0, 0.0],
            extent: ctx.window.inner_size().into(),
            depth_range: 0.0..=1.0,
        };
        let new_pipeline = get_pipeline(
            viewport,
            &ctx.vk_state,
            ctx.render_pass.clone(),
            ctx.vs.clone(),
            ctx.fs.clone(),
        );
        let new_command_buffers = get_command_buffers(
            &ctx.command_buffer_allocator,
            &ctx.vk_state,
            new_pipeline,
            &ctx.vertex_buf,
            &ctx.framebuffers.clone(),
        );

        ctx.command_buffers = new_command_buffers;
    }
}

fn main() {
    let event_loop = EventLoop::new().unwrap();
    let mut app = App::default();
    event_loop.set_control_flow(ControlFlow::Wait);
    event_loop.run_app(&mut app).unwrap();
}
