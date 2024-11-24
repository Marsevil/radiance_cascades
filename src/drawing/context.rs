use std::sync::Arc;

use vulkano::{
    command_buffer::allocator::{
        StandardCommandBufferAllocator, StandardCommandBufferAllocatorCreateInfo,
    },
    image::{view::ImageView, Image, ImageUsage},
    memory::allocator::StandardMemoryAllocator,
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
    swapchain::{PresentMode, Swapchain, SwapchainCreateInfo},
};
use winit::window::Window;

use crate::{
    drawing::{fs, vs, vulkan_helper},
    geometry::Vertex2D,
};

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
    let image_count = u32::min(
        caps.min_image_count + 1,
        caps.max_image_count.unwrap_or(u32::MAX),
    );
    let present_mode = if caps
        .compatible_present_modes
        .contains(&PresentMode::Mailbox)
    {
        PresentMode::Mailbox
    } else {
        PresentMode::Fifo
    };
    let (swapchain, images) = Swapchain::new(
        vk_ctx.device.clone(),
        vk_ctx.surface.clone(),
        SwapchainCreateInfo {
            min_image_count: image_count,
            image_format,
            image_extent: dimensions.into(),
            image_usage: ImageUsage::COLOR_ATTACHMENT,
            composite_alpha,
            present_mode,
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

pub struct DrawingContext {
    pub command_buffer_allocator: StandardCommandBufferAllocator,
    pub buffer_allocator: Arc<StandardMemoryAllocator>,
    pub swapchain: Arc<Swapchain>,
    pub framebuffers: Box<[Arc<Framebuffer>]>,
    pub render_pass: Arc<RenderPass>,
    pub pipeline: Arc<GraphicsPipeline>,
    pub vs: Arc<ShaderModule>,
    pub fs: Arc<ShaderModule>,
}
impl DrawingContext {
    pub fn new(vk_state: &vulkan_helper::VulkanState, window: &Window) -> Self {
        let memory_allocator = Arc::new(StandardMemoryAllocator::new_default(
            vk_state.device.clone(),
        ));
        let command_buffer_allocator = StandardCommandBufferAllocator::new(
            vk_state.device.clone(),
            StandardCommandBufferAllocatorCreateInfo::default(),
        );
        let vs = vs::load(vk_state.device.clone()).expect("Can't compile vertex shader");
        let fs = fs::load(vk_state.device.clone()).expect("Can't compile fragment shader");

        let (swapchain, images) = get_swapchain(window, vk_state);
        let render_pass = vulkano::single_pass_renderpass!(
            vk_state.device.clone(),
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
        let viewport = Viewport {
            offset: [0.0, 0.0],
            extent: window.inner_size().into(),
            depth_range: 0.0..=1.0,
        };
        let pipeline = get_pipeline(
            viewport,
            vk_state,
            render_pass.clone(),
            vs.clone(),
            fs.clone(),
        );

        DrawingContext {
            command_buffer_allocator,
            buffer_allocator: memory_allocator,
            swapchain,
            pipeline,
            render_pass,
            framebuffers: framebufs,
            vs,
            fs,
        }
    }

    pub fn revoke_swapchain(self, window: &Window) -> Self {
        let new_dimensions = window.inner_size();
        let (new_swapchain, new_images) = self
            .swapchain
            .recreate(SwapchainCreateInfo {
                image_extent: new_dimensions.into(),
                ..self.swapchain.create_info()
            })
            .expect("Failed to recreate swapchain {e");

        let new_framebuffers = get_framebuffers(&new_images, &self.render_pass);

        Self {
            swapchain: new_swapchain,
            framebuffers: new_framebuffers,
            ..self
        }
    }

    pub fn resize_viewport(self, vk_state: &vulkan_helper::VulkanState, window: &Window) -> Self {
        let viewport = Viewport {
            offset: [0.0, 0.0],
            extent: window.inner_size().into(),
            depth_range: 0.0..=1.0,
        };
        let new_pipeline = get_pipeline(
            viewport,
            vk_state,
            self.render_pass.clone(),
            self.vs.clone(),
            self.fs.clone(),
        );

        Self {
            pipeline: new_pipeline,
            ..self
        }
    }
}
